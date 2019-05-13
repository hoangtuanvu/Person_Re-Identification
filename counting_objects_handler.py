import os
import cv2
import numpy as np
import json
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
from tracking.sort.sort import Sort
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.track import TrackState
from tracking.deep_sort.tracker import Tracker
from detection.model.yolov3 import Darknet
from detection.utils.commons import non_max_suppression
from detection.utils.visualization import gen_colors
from re_id.reid import models
from re_id.reid.utils.serialization import load_checkpoint
from utilities import boxes_filtering
from utilities import read_counting_gt

font = cv2.FONT_HERSHEY_PLAIN
line = cv2.LINE_AA


class PersonHandler:
    def __init__(self, args, encoder=None, cls_out=None, metric=None):
        # Tracking Variables
        self.matching_threshold = args.matching_threshold
        self.encoder = encoder
        self.tracking_type = args.tracking_type
        self.tracker = None
        self.cls_out = cls_out
        self.other_trackers = {}
        self.metric = metric

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
        self.out = None
        self.colors = {}
        self.output_name = args.output_name
        self.gt = args.gt

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

    def online_process(self, loader):
        """Loop, grab images from camera, and do count number of objects in Online mode."""

        for i, (_, img, img0) in enumerate(loader):
            display = np.array(img0)
            output = display.copy()
            self.detect_n_counting(output, img)

            if self.out is not None:
                self.out.write(output)

            if self.use_resize:
                output = cv2.resize(output, (1280, 960), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

    def offline_process(self, loader):
        """Loop, grab images from images/videos, and do count number of objects in Offline mode."""

        object_cnt_all = []
        total_objects = {}
        start_time = time.time()
        for i, (path, img, img0) in enumerate(loader):
            if loader.frame == 1:
                print('Processing {}'.format(path))

            display = np.array(img0)
            output = display.copy()
            self.detect_n_counting(output, img, total_objects=total_objects)

            if loader.frame == loader.nframes:
                # the last frame on each video
                res = []
                for cls in self.cls_out:
                    if cls in total_objects:
                        res.append(total_objects[cls])
                    else:
                        res.append(0)

                res.insert(1, 0)
                object_cnt = {"id": int(os.path.basename(path).split('.')[0].split('_')[-1]), "objects": res}
                object_cnt_all.append(object_cnt)

                # Reset trackers
                self.init_tracker()
                self.init_other_trackers()

                # clear total of objects of previous video
                total_objects.clear()

        with open('{}.txt'.format(self.output_name), 'w') as file:
            file.write(json.dumps(object_cnt_all))

        # Calculate Errors
        gt = read_counting_gt(self.gt)

        avg = [0] * 6
        for i in range(len(object_cnt_all)):
            sub_gt = gt[i]
            sub_pred = object_cnt_all[i]

            avg = [avg[j] + abs(sub_gt['objects'][j] - sub_pred['objects'][j]) for j in
                   range(len(sub_gt['objects']))]

        avg = [item / len(object_cnt_all) for item in avg]

        avg_err = {"err": avg}
        object_cnt_all.append(avg_err)

        with open('{}_extend.txt'.format(self.output_name), 'w') as file:
            file.write(json.dumps(object_cnt_all))

        print('Time to process', time.time() - start_time)

    def detect_n_counting(self, origimg, img, total_objects=None):
        """Do object detection over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # Applies yolov3 detection
        with torch.no_grad():
            detections, _ = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

        if detections is None:
            return [], [], []

        box, conf, cls = boxes_filtering(origimg, detections, self.img_size, cls_out=self.cls_out)

        if len(box) == 0:
            return [], [], []

        cls_out_dict = {}
        for i in range(len(box)):
            if cls[i] not in cls_out_dict:
                cls_out_dict[cls[i]] = [[box[i]], [conf[i]]]
            else:
                cls_out_dict[cls[i]][0].append(box[i])
                cls_out_dict[cls[i]][1].append(conf[i])

        for cls in cls_out_dict:
            cls_boxes = cls_out_dict[cls][0]
            cls_conf = cls_out_dict[cls][1]

            total = 0
            if cls == 0:
                # person counting
                features = self.encoder(origimg, cls_boxes)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(cls_boxes, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                detections = [detections[i] for i in indices]

                self.tracker.predict()
                if self.counting_use_reid:
                    self.tracker.update(detections, self.reid_model, origimg, self.matching_threshold)
                else:
                    self.tracker.update(detections)

                for track in self.tracker.tracks:

                    if track.state == TrackState.Confirmed:
                        total += 1
                        if track.time_since_update > 1 and track.hits <= 5:
                            total -= 1

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr().astype(int)
                    cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 2)
                    cv2.putText(origimg, str(track.track_id), (bbox[0], bbox[1]), 0, 5e-3 * 200, (0, 255, 0), 2)

            else:
                # other objects counting
                dets = []
                for i in range(len(cls_boxes)):
                    x, y, w, h = cls_boxes[i]
                    dets.append([x, y, x + w, y + h, cls_conf[i]])

                dets = np.asarray(dets)
                self.other_trackers[cls].update(dets)

                for track in self.other_trackers[cls].trackers:
                    bbox = np.array(track.get_state()[0]).astype(int)
                    total += 1
                    if track.time_since_update > 1 and track.hits < 5:
                        total -= 1

                    if (track.time_since_update > 1) or \
                            (track.hit_streak < 3):
                        continue

                    cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 2)
                    cv2.putText(origimg, str(int(track.id)), (bbox[0], bbox[3]), 0, 5e-3 * 200, (0, 255, 0), 2)

            if total_objects is not None:
                total_objects[cls] = total

    def set_out(self, out):
        self.out = out

    def init_tracker(self):
        self.tracker = Tracker(self.metric)

    def init_other_trackers(self):
        for cls in self.cls_out:
            if cls != 0:
                self.other_trackers[cls] = Sort(max_age=300)

    def set_colors(self):
        colors = gen_colors(len(self.cls_out))

        i = 0
        for cls in self.cls_out:
            self.colors[cls] = colors[i]
            i += 1
