from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
from tracking.sort.sort import Sort
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort import nn_matching
from detection.model.yolov3 import Darknet
from detection.utils.commons import non_max_suppression
from detection.utils.visualization import gen_colors
from processor.utilities import read_counting_gt
from processor.utilities import convert_number_to_image_form
from processor.utilities import rms
from processor.post_process import boxes_filtering
from processor.post_process import gen_report
from processor.post_process import save_probe_dir
from processor.post_process import gen_total_objects
from processor.post_process import ct_boxes_filer
from detectors.detector_factory import detector_factory
from utils.debugger import coco_class_name


class PersonHandler:
    null_values = [], [], []

    def __init__(self, args, p_encoder=None, v_encoder=None, cls_out=None, coordinates_out=None):
        # Tracking Variables
        self.p_encoder = p_encoder
        self.v_encoder = v_encoder
        self.tracker = dict()
        self.other_trackers = dict()

        # Detection Variables
        self.conf_th = args.conf_th
        self.nms_thres = args.nms_thres
        self.img_size = args.img_size
        self.resize_mode = args.mode

        if len(args.config_path) == 0 or len(args.detection_weight) == 0:
            raise ValueError('Detection model weight does not exist!')

        self.use_gpu = torch.cuda.is_available()
        if args.use_cpu:
            self.use_gpu = False

        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.use_resize = args.use_resize
        self.out = None
        self.colors = dict()
        self.output_name = args.output_name
        self.gt = args.gt
        self.coordinates_out = coordinates_out
        self.saved_dir = None
        self.track_dir = None
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.save_probe = args.save_probe
        self.od_model = args.od_model
        self.min_shake_point = args.min_shake_point
        self.stable_point = args.stable_point
        self.shake_camera = False
        self.prev_bboxes = -1
        self.cons_frames = list()
        self.max_cosine_distance = args.max_cosine_distance

        # Load model Detection
        print('Loading detection model ...')
        if self.od_model == 'yolo':
            detect_model = Darknet(args.config_path, img_size=self.img_size, device=self.device)
            if args.detection_weight.endswith('.pt'):
                detect_model.load_state_dict(torch.load(args.detection_weight, map_location=self.device)['model'])
            else:
                detect_model.load_darknet_weights(args.detection_weight)

            self.detect_model = nn.DataParallel(detect_model).cuda() if self.use_gpu else detect_model
            self.detect_model.eval()
            self.cls_out = cls_out
        else:
            Detector = detector_factory[args.task]
            self.detect_model = Detector(args)
            self.cls_out = [cls_id for cls_id in range(len(coco_class_name) - 1)]

        cudnn.benchmark = True

    def online_process(self, loader):
        """Loop, grab images from camera, and do count number of objects in Online mode."""

        for i, (_, img, img0) in enumerate(loader):
            display = np.array(img0)
            output = display.copy()
            self.detect_n_counting(output, img, loader=loader)

            if self.out is not None:
                self.out.write(output)

            if self.use_resize:
                output = cv2.resize(output, (1280, 960), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

    def offline_process(self, loader):
        """Loop, grab images from images/videos, and do count number of objects in Offline mode."""

        start_time = time.time()
        object_cnt_all = []
        total_objects = {}

        # Load ground truth
        gt = read_counting_gt(self.gt)

        out = None
        for i, (path, img, img0) in enumerate(loader):
            if self.saved_dir is not None and out is None:
                out = cv2.VideoWriter('{}/{}.avi'.format(self.saved_dir, os.path.basename(path).split('.')[0]),
                                      cv2.VideoWriter_fourcc(*'XVID'), 10, (self.image_width, self.image_height), True)

            display = np.array(img0)
            output = display.copy()

            self.detect_n_counting(output, img, loader=loader, out=out)

            if out is not None:
                out.write(output)

            if loader.frame == loader.nframes:
                # Process counts and tracks
                self.write_tracks_n_counts(loader=loader, total_objects=total_objects)

                # the last frame on each video
                object_cnt = {"name": os.path.basename(path).split('.')[0],
                              "objects": gen_total_objects(self.cls_out, total_objects, self.od_model),
                              "rms": rms(gt[loader.count]["objects"],
                                         gen_total_objects(self.cls_out, total_objects, self.od_model))}
                object_cnt_all.append(object_cnt)

                # clear total of objects of previous video
                total_objects.clear()

                # Reset out
                out = None

                # Reset trackers
                self.init_tracker()
                # self.init_other_trackers()

        # Generate Report
        gen_report(gt, object_cnt_all)

        print('Time to process', time.time() - start_time)

    def detect_n_counting(self, origimg, img, loader=None, out=None):
        """Do object detection over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        raw_img = origimg.copy()

        # Apply Object Detection models
        with torch.no_grad():
            if self.od_model == 'yolo':
                detections, _ = self.detect_model(input_imgs)
                detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

                if detections is None:
                    return self.null_values

                box, conf, cls = boxes_filtering(raw_img, detections, self.img_size, cls_out=self.cls_out,
                                                 mode=self.resize_mode)
            else:
                detections = self.detect_model.run(img, loader.frame, vid_writer=out)
                if not bool(detections):
                    return self.null_values

                box, conf, cls = ct_boxes_filer(detections['results'], self.cls_out, self.conf_th)

            # Identify shake point
            if loader.frame > 1:
                if abs(len(box) - self.prev_bboxes) >= self.min_shake_point:
                    self.shake_camera = True
                    self.cons_frames.clear()
                else:
                    self.cons_frames.append(True)

            if len(self.cons_frames) >= self.stable_point:
                self.shake_camera = False

            self.prev_bboxes = len(box)

        if len(box) == 0:
            return self.null_values

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

            if cls in self.tracker:
                # People and Vehicle Tracking
                if cls == 0:
                    features = self.p_encoder(raw_img, cls_boxes)
                else:
                    features = self.v_encoder(raw_img, cls_boxes)

                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(cls_boxes, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                detections = [detections[i] for i in indices]

                self.tracker[cls].predict()
                self.tracker[cls].update(detections, self.shake_camera)

                for track in self.tracker[cls].tracks:
                    bbox = track.to_tlbr().astype(int)

                    # save tracked list
                    if self.save_probe:
                        save_probe_dir(video_id=os.path.basename(loader.path).split('.')[0][1:],
                                       track_id=track.track_id, raw_img=raw_img, bbox=bbox)

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 2)
                    cv2.putText(origimg, str(track.track_id), (bbox[0], bbox[1]), 0, 5e-3 * 200, (0, 255, 0), 2)

                    # write coordinates
                    self.write_coordinates(loader=loader, x=bbox[0], y=bbox[1], w=bbox[2] - bbox[0],
                                           h=bbox[3] - bbox[1], cls=cls, track_id=track.track_id)
            else:
                # Other objects Tracking
                dets = []
                for i in range(len(cls_boxes)):
                    x, y, w, h = cls_boxes[i]
                    dets.append([x, y, x + w, y + h, cls_conf[i]])

                dets = np.asarray(dets)
                self.other_trackers[cls].update(dets)

                for track in self.other_trackers[cls].trackers:
                    bbox = np.array(track.get_state()[0]).astype(int)
                    if (track.time_since_update > 1) or \
                            (track.hit_streak < 3):
                        continue

                    cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 2)
                    cv2.putText(origimg, str(int(track.id)), (bbox[0], bbox[3]), 0, 5e-3 * 200, (0, 255, 0), 2)

                    # write coordinates
                    self.write_coordinates(loader=loader, x=bbox[0], y=bbox[1], w=bbox[2] - bbox[0],
                                           h=bbox[3] - bbox[1], cls=cls, track_id=int(track.id))

    def set_out(self, out):
        self.out = out

    def set_saved_dir(self, saved_dir):
        self.saved_dir = saved_dir

    def set_track_dir(self, track_dir):
        self.track_dir = track_dir

    def init_tracker(self):
        for cls in self.cls_out:
            if self.od_model == 'yolo':
                if cls in [0, 2, 5, 7]:
                    self.tracker[cls] = Tracker(
                        nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance))
                else:
                    self.other_trackers[cls] = Sort(max_age=300)
            else:
                if cls in [0, 1, 4, 5]:
                    self.tracker[cls] = Tracker(
                        nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance))
                else:
                    self.other_trackers[cls] = Sort(max_age=300)

    def set_colors(self):
        colors = gen_colors(len(self.cls_out))

        i = 0
        for cls in self.cls_out:
            self.colors[cls] = colors[i]
            i += 1

    def write_coordinates(self, loader, x, y, w, h, cls, track_id):
        if self.coordinates_out is not None:
            self.coordinates_out.write('{},{},{},{},{},{},{}\n'.format(
                '{}_{}.jpg'.format(os.path.basename(loader.path).split('.')[0],
                                   convert_number_to_image_form(loader.frame)),
                x, y, w, h, self.cls_out.index(cls) + 1, track_id
            ))

    def write_tracks_n_counts(self, loader, total_objects=None):
        if self.track_dir is not None:
            f = open('{}/{}.txt'.format(self.track_dir, os.path.basename(loader.path).split('.')[0]), 'w')

        # Process other objects like cars, truck,...
        for cls in self.other_trackers:
            total = 0
            for track in self.other_trackers[cls].trackers:
                bbox = np.array(track.tlbr).astype(int)

                if self.track_dir is not None:
                    f.write('{},{}, xmin={}, ymin={}, xmax={}, ymax={}, width={}, height={}\n'.format(
                        cls, track, bbox[0], bbox[1], bbox[2], bbox[3], bbox[2] - bbox[0], bbox[3] - bbox[1]))

                if track.hits >= 6:
                    total += 1

            total_objects[cls] = total

        for cls in self.tracker:
            # Process people and Vehicle
            total = 0

            for track in self.tracker[cls].tracks:
                bbox = track.init_bbox.astype(int)

                if self.track_dir is not None:
                    f.write('{},{}, xmin={}, ymin={}, xmax={}, ymax={}, width={}, height={}\n'.format(
                        cls, track, bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1], bbox[2], bbox[3]))

                if cls == 0:
                    if track.hits >= 5:
                        total += 1
                else:
                    if track.hits >= 6:
                        total += 1

            total_objects[cls] = total
