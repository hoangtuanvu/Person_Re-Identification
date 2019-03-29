from utils import *
from detection.trt_models.detection import obj_det_graph
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from re_id.reid import models
from re_id.reid.utils.serialization import load_checkpoint
from torch.utils.data import DataLoader
from torchvision.transforms import *
from re_id.reid.dataset_loader import RawDatasetImages
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection

MEASURE_MODEL_TIME = False


class PersonHandler:
    def __init__(self, args):
        self.tracking_type = args.tracking_type
        self.matching_threshold = args.matching_threshold
        self.img_boxes = []
        self.conf_th = args.conf_th
        self.query_embedding = None
        self.embeddings = None
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 150, 255),
                       (0, 255, 255), (200, 0, 200), (255, 191, 0), (180, 105, 255)]
        self.query = []
        self.query_paths = []
        self.transform = Compose([
            Resize((384, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.line = [(43, 543), (550, 655)]
        self.memory = {}

        if len(args.model) == 0:
            raise ValueError('Detection model weight does not exist!')

        if len(args.load_weights) == 0:
            raise ValueError('Person ReID model weight does not exist!')

        # Load model Detection
        self.sess, self.scores, self.boxes, self.classes, self.input = obj_det_graph(args.model, 'data',
                                                                                     args.detection_weights)

        # Load model Person Re-identification
        self.use_gpu = torch.cuda.is_available()
        if args.use_cpu:
            self.use_gpu = False

        print('Initializing model: {}'.format(args.arch))
        reid_model = models.create(args.arch, num_features=256,
                                   dropout=args.dropout, num_classes=args.num_classes, cut_at_pooling=False, FCN=True)
        load_checkpoint(reid_model, args.load_weights)
        self.reid_model = nn.DataParallel(reid_model).cuda() if self.use_gpu else reid_model

    def loop_and_detect(self, cam, vis, od_type, tracker, encoder, img_query):
        """Loop, grab images from camera, and do object detection.

        # Arguments
          cam: the camera object (video source).
          tf_sess: TensorFlow/TensorRT session to run SSD object detection.
          conf_th: confidence/score threshold for object detection.
          vis: for visualization.
        """
        show_fps = True
        fps = 0.0
        tic = time.time()
        while True:

            ret, img = cam.read()

            if not ret:
                break

            display = np.array(img)
            output = display.copy()

            # Run Object Detection and Tracking
            box, conf, cls = self.detect(output, encoder, tracker, od_type=od_type)

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
                    person_slice = img[box[i][0]:box[i][2], box[i][1]:box[i][3], :]
                    persons += [person_slice]

                query_loader = DataLoader(
                    RawDatasetImages(self.query, transform=self.transform),
                    batch_size=1, shuffle=False,
                    pin_memory=self.use_gpu, drop_last=False)

                gallery_loader = DataLoader(
                    RawDatasetImages(persons, transform=self.transform),
                    batch_size=1, shuffle=False,
                    pin_memory=self.use_gpu, drop_last=False)

                self.embeddings = inference(gallery_loader, self.reid_model, self.use_gpu)
                self.query_embedding = inference(query_loader, self.reid_model, self.use_gpu)

                # run matching algorithm
                bb_idx = matching(self.query_embedding, self.embeddings, self.matching_threshold)

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

            if show_fps:
                output = draw_help_and_fps(output, fps, True)

            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

            # Calculate FPS
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps * 0.9 + curr_fps * 0.1)
            tic = toc

    def detect(self, origimg, encoder, tracker, od_type='ssd'):
        """Do object detection over 1 image."""
        global avg_time

        if od_type == 'faster_rcnn':
            img = resize(origimg, (1024, 576))
        elif od_type == 'ssd':
            img = resize(origimg, (300, 300))
        else:
            raise ValueError('bad object detector type: $s' % od_type)

        if MEASURE_MODEL_TIME:
            tic = time.time()

        start_time = time.time()
        boxes_out, scores_out, classes_out = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={self.input: img[None, ...]})
        print('Processed time: {}'.format(time.time() - start_time))
        _box, _conf, cls = person_filtering(origimg, boxes_out, scores_out, classes_out, self.conf_th)

        if len(_box) == 0:
            return [], [], []

        if self.tracking_type == 'sort':
            box = [tmp_box.tolist() for tmp_box in _box]
            conf = [tmp_conf.tolist() for tmp_conf in _conf]

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(box, conf, self.conf_th, 0.3)
            dets = []
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    (x, y) = (box[i][1], box[i][0])
                    (w, h) = (box[i][3] - box[i][1], box[i][2] - box[i][0])
                    dets.append([x, y, x + w, y + h, conf[i]])

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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
            boxs = convert_boxes(_box)
            features = encoder(origimg, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
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

        if MEASURE_MODEL_TIME:
            td = (time.time() - tic) * 1000  # in ms
            avg_time = avg_time * 0.9 + td * 0.1
            print('tf_sess.run() took {:.1f} ms on average'.format(avg_time))

        return _box, _conf, cls
