import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from utilities import matching
from utilities import img_transform
from utilities import draw_help_and_fps
from utilities import boxes_filtering
from utilities import visualize_box
from torch.backends import cudnn
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection
from detection.model.yolov3 import Darknet
from detection.utils.commons import non_max_suppression
from re_id.reid import models
from re_id.reid.utils.serialization import load_checkpoint
from re_id.reid.feature_extraction.cnn import extract_cnn_feature


class PersonHandler:
    def __init__(self, args):
        # Tracking Variables
        self.tracking_type = args.tracking_type
        self.matching_threshold = args.matching_threshold
        self.memory = {}

        # Detection Variables
        self.conf_th = args.conf_th
        self.nms_thres = args.nms_thres
        self.img_size = args.img_size

        # Re-ID Variables
        self.img_boxes = []
        self.query_embedding = None
        self.embeddings = None
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 150, 255),
                       (0, 255, 255), (200, 0, 200), (255, 191, 0), (180, 105, 255)]
        self.query = []
        self.query_paths = []

        if len(args.config_path) == 0 or len(args.detection_weight) == 0:
            raise ValueError('Detection model weight does not exist!')

        if len(args.reid_weights) == 0:
            raise ValueError('Person ReID model weight does not exist!')

        self.use_gpu = torch.cuda.is_available()
        if args.use_cpu:
            self.use_gpu = False

        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.use_resize = args.use_resize
        self.show_fps = args.show_fps

        # Load model Detection
        print('Loading detection model ...')
        detect_model = Darknet(args.config_path, img_size=self.img_size, device=self.device)
        detect_model.load_darknet_weights(args.detection_weight)

        self.detect_model = nn.DataParallel(detect_model).cuda() if self.use_gpu else detect_model
        self.detect_model.eval()
        cudnn.benchmark = True

        # Load model Person Re-identification
        print('Loading Re-ID model')
        if args.arch.startswith('resnet'):
            reid_model = models.create(args.arch, num_features=256,
                                       dropout=args.dropout, num_classes=args.num_classes, cut_at_pooling=False,
                                       FCN=True)
        else:
            reid_model = models.create(args.arch, num_classes=args.num_classes, use_gpu=self.use_gpu)

        load_checkpoint(reid_model, args.reid_weights)
        self.reid_model = nn.DataParallel(reid_model).cuda() if self.use_gpu else reid_model

    def loop_and_detect(self, loader, vis, tracker, encoder, img_query):
        """Loop, grab images from camera, do object detection and person re-identification.

        # Arguments
          loader: the camera object (video source).
          vis: visualization tool
          tracker: is to update detection boxes for tracking.
          encoder: is to extract features from image (deep sort)
          img_query: probe image for querying people.
        """
        fps = 0.0
        tic = time.time()
        for i, (img, img0) in enumerate(loader):
            display = np.array(img0)
            output = display.copy()
            box, conf, cls = self.detect_n_track(output, img, encoder, tracker)

            if len(img_query) > 0 and len(box) > 0:
                overlay = display.copy()
                for img_path in img_query:
                    if img_path not in self.query_paths:
                        self.query_paths.append(img_path)
                        self.query.append(cv2.imread(img_path))

                # Person Re-ID
                self.img_boxes = []
                persons = []
                for i in range(len(box)):
                    self.img_boxes.append(box[i])
                    person_slice = img0[box[i][0]:box[i][2], box[i][1]:box[i][3], :]
                    persons += [person_slice]

                query_imgs = img_transform(self.query, (128, 384))
                gallery_imgs = img_transform(persons, (128, 384))

                self.embeddings = self.extract_embeddings(gallery_imgs)
                self.query_embedding = self.extract_embeddings(query_imgs)

                # run matching algorithm
                bb_idx = matching(self.query_embedding, self.embeddings, self.matching_threshold)
                bb_idx = [idx[1] for idx in bb_idx]

                # Draw relevant info on bounding boxes
                for i, bbox in enumerate(self.img_boxes):
                    if i in bb_idx:
                        cv2.putText(output, 'Found One', (bbox[1] + 10, bbox[2] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        color = self.COLORS[i // len(self.COLORS)]
                        cv2.rectangle(overlay, (bbox[1], bbox[0]), (bbox[3], bbox[2]),
                                      color, -1)

                # Add overlay, show image, update fps buffer, and exit
                cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)

            if not self.tracking_type:
                output = vis.draw_bboxes(output, box, conf, cls)

            if self.show_fps:
                draw_help_and_fps(output, fps, True)

            if self.use_resize:
                output = cv2.resize(output, (640, 480), interpolation=cv2.INTER_LINEAR)

            start_time = time.time()
            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

            print('Time to write image', time.time() - start_time)
            # Calculate FPS
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps * 0.9 + curr_fps * 0.1)
            tic = toc

    def detect_n_track(self, origimg, img, encoder, tracker):
        """Do object detection and object tracking (optional) over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # Applies yolov3 detection
        with torch.no_grad():
            detections = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

        if detections is None:
            return [], [], []

        _box, _conf, cls = boxes_filtering(origimg, detections, self.img_size)

        vis_box = visualize_box(_box)

        if len(_box) == 0:
            return [], [], []

        # Applies deep_sort/sort for tracking people
        if self.tracking_type == 'sort':
            dets = []
            for i in range(len(_box)):
                x, y, w, h = _box[i]
                dets.append([x, y, x + w, y + h, _conf[i]])

            dets = np.asarray(dets)
            tracks = tracker.update(dets)

            boxes = []
            indexIDs = []
            previous = self.memory.copy()
            self.memory = {}

            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                self.memory[indexIDs[-1]] = boxes[-1]

            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    # extract the bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in self.COLORS[indexIDs[i] % len(self.COLORS)]]
                    cv2.rectangle(origimg, (x, y), (w, h), color, 2)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                        p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                        cv2.line(origimg, p0, p1, color, 3)
                    text = "{}".format(indexIDs[i])
                    cv2.putText(origimg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    i += 1
        elif self.tracking_type == "deep_sort":
            features = encoder(origimg, _box)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(_box, features)]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
            detections = [detections[i] for i in indices]
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(origimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(origimg, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        return vis_box, _conf, cls

    def extract_embeddings(self, inputs):
        return list(extract_cnn_feature(self.reid_model, inputs))
