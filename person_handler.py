import cv2
import time
import uuid
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
from re_id.reid.utils.data.iotools import mkdir_if_missing
from re_id.reid.utils.serialization import load_checkpoint
from re_id.reid.feature_extraction.cnn import extract_cnn_feature


class PersonHandler:
    def __init__(self, args, encoder=None):
        # Tracking Variables
        self.tracking_type = args.tracking_type
        self.matching_threshold = args.matching_threshold
        self.tracker = None
        self.encoder = encoder

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
        self.out = None
        self.no_frame = 0
        self.save_probe = args.save_probe
        self.freq = args.freq

        # Load model Detection
        print('Loading detection model ...')
        detect_model = Darknet(args.config_path, img_size=self.img_size, device=self.device)
        detect_model.load_darknet_weights(args.detection_weight)
        detect_model.fuse()

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

    def loop_and_detect(self, loader, vis, img_query):
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
            self.no_frame += 1
            display = np.array(img0)
            output = display.copy()
            box, conf, cls = self.detect_n_track(output, img)

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

            if self.out is not None:
                self.out.write(output)

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

    def detect_n_track(self, origimg, img):
        """Do object detection and object tracking (optional) over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        raw_img = origimg.copy()

        # Applies yolov3 detection
        with torch.no_grad():
            detections, _ = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

        if detections is None:
            return [], [], []

        _box, _conf, cls = boxes_filtering(origimg, detections, self.img_size, cls_out=[0])

        if len(_box) == 0:
            return [], [], []

        # Applies deep_sort/sort for tracking people
        if self.tracking_type == 'sort':
            dets = []
            for i in range(len(_box)):
                x, y, w, h = _box[i]
                dets.append([x, y, x + w, y + h, _conf[i]])

            dets = np.asarray(dets)
            tracks = self.tracker.update(dets)

            tmp_box = []
            for track in tracks:
                bbox = np.array(track[:4]).astype(int)

                self.save_probe_dir(int(track[4]), raw_img, bbox)
                cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                cv2.putText(origimg, str(int(track[4])), (bbox[0], bbox[3]), 0, 5e-3 * 200, (0, 255, 0), 2)

                tlwh = bbox.copy()
                tlwh[2:] -= tlwh[:2]
                tmp_box.append(tlwh)

            vis_box = visualize_box(tmp_box)
        elif self.tracking_type == "deep_sort":
            features = self.encoder(origimg, _box)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(_box, features)]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
            detections = [detections[i] for i in indices]
            self.tracker.predict()
            self.tracker.update(detections)

            tmp_box = []
            for track in self.tracker.tracks:
                if track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr().astype(int)
                self.save_probe_dir(track.track_id, raw_img, bbox)
                cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                cv2.putText(origimg, str(track.track_id), (bbox[0], bbox[1]), 0, 5e-3 * 200, (0, 255, 0), 2)
                tmp_box.append(track.to_tlwh().astype(int))

            vis_box = visualize_box(tmp_box)
        else:
            vis_box = visualize_box(_box)

        return vis_box, _conf, cls

    def extract_embeddings(self, inputs):
        return list(extract_cnn_feature(self.reid_model, inputs))

    def set_out(self, out):
        self.out = out

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_probe_dir(self, track_id, raw_img, bbox):
        """ save query images in probe directory"""
        if self.save_probe:
            dir = 'probe/{}'.format(track_id)
            mkdir_if_missing(dir)
            if self.no_frame % self.freq == 0:
                cv2.imwrite('{}/{}.png'.format(dir, str(uuid.uuid4())),
                            raw_img[bbox[1]: bbox[3], bbox[0]: bbox[2], :])
