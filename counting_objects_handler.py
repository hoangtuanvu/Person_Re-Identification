import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.track import TrackState
from detection.model.yolov3 import Darknet
from detection.utils.commons import non_max_suppression
from re_id.reid import models
from re_id.reid.utils.serialization import load_checkpoint
from utilities import visualize_box
from utilities import boxes_filtering

font = cv2.FONT_HERSHEY_PLAIN
line = cv2.LINE_AA


class PersonHandler:
    def __init__(self, args):
        # Tracking Variables
        self.matching_threshold = args.matching_threshold

        # Detection Variables
        self.conf_th = args.conf_th
        self.nms_thres = args.nms_thres
        self.img_size = args.img_size

        # Re-ID Variables
        self.counting_use_reid = args.counting_use_reid

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

    def process(self, loader, tracker, encoder, out):
        """Loop, grab images from camera, and do object detection.

        # Arguments
          cam: the camera object (video source).
          tf_sess: TensorFlow/TensorRT session to run SSD object detection.
          conf_th: confidence/score threshold for object detection.
          vis: for visualization.
        """

        for i, (img, img0) in enumerate(loader):
            display = np.array(img0)
            output = display.copy()
            self.detect_n_counting(output, img, encoder, tracker)

            if out is not None:
                out.write(output)

            if self.use_resize:
                output = cv2.resize(output, (1280, 960), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

    def detect_n_counting(self, origimg, img, encoder, tracker):
        """Do object detection over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # Applies yolov3 detection
        with torch.no_grad():
            detections = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

        if detections is None:
            return [], [], []

        box, conf, cls = boxes_filtering(origimg, detections, self.img_size)
        vis_box = visualize_box(box)

        if len(box) == 0:
            return [], [], []

        # Applies deep sort for tracking people and person re-id for counting people
        features = encoder(origimg, box)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(box, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        if self.counting_use_reid:
            tracker.update(detections, self.reid_model, origimg, self.matching_threshold)
        else:
            tracker.update(detections)

        total = 0
        for track in tracker.tracks:
            if track.state == TrackState.Confirmed:
                total += 1
                if track.time_since_update > 1 and track.hits <= 5:
                    total -= 1

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(origimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        self.draw_counting_numbers(origimg, len(detections), total)

        return vis_box, conf, cls

    @staticmethod
    def draw_counting_numbers(img, current, total):
        """
        Draw number of current people and total of people in a video
        :param img: frame of video
        :param current: number of people in current frame
        :param total: number of people in consecutive frames
        """
        cv2.putText(img, 'CURRENT: {}'.format(current), (11, 50), font, 2.0, (32, 32, 32), 4, line)
        cv2.putText(img, 'CURRENT: {}'.format(current), (10, 50), font, 2.0, (240, 240, 240), 1, line)
        cv2.putText(img, 'TOTAL: {}'.format(total), (11, 100), font, 2.0, (32, 32, 32), 4, line)
        cv2.putText(img, 'TOTAL: {}'.format(total), (10, 100), font, 2.0, (240, 240, 240), 1, line)
