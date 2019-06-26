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
from detection.model.yolov3 import Darknet
from detection.utils.commons import non_max_suppression
from detection.utils.visualization import gen_colors
from re_id.reid.utils.data.iotools import mkdir_if_missing
from utilities import boxes_filtering
from utilities import read_counting_gt
from utilities import convert_number_to_image_form
from utilities import rms
from utilities import gen_report

font = cv2.FONT_HERSHEY_PLAIN
line = cv2.LINE_AA


class PersonHandler:
    def __init__(self, args, encoder=None, cls_out=None, metric=None, coordinates_out=None):
        # Tracking Variables
        self.encoder = encoder
        self.tracker = None
        self.cls_out = cls_out
        self.other_trackers = {}
        self.metric = metric

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
        self.colors = {}
        self.output_name = args.output_name
        self.gt = args.gt
        self.coordinates_out = coordinates_out
        self.saved_dir = None
        self.track_dir = None
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.save_probe = args.save_probe

        # Load model Detection
        print('Loading detection model ...')
        detect_model = Darknet(args.config_path, img_size=self.img_size, device=self.device)
        if args.detection_weight.endswith('.pt'):
            detect_model.load_state_dict(torch.load(args.detection_weight, map_location=self.device)['model'])
        else:
            detect_model.load_darknet_weights(args.detection_weight)

        self.detect_model = nn.DataParallel(detect_model).cuda() if self.use_gpu else detect_model
        self.detect_model.eval()
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

            self.detect_n_counting(output, img, loader=loader)

            if out is not None:
                out.write(output)

            if loader.frame == loader.nframes:
                # Process counts and tracks
                self.write_tracks_n_counts(loader=loader, total_objects=total_objects)

                # the last frame on each video
                object_cnt = {"name": os.path.basename(path).split('.')[0],
                              "objects": self.gen_total_objects(total_objects),
                              "rms": rms(gt[loader.count]["objects"], self.gen_total_objects(total_objects))}
                object_cnt_all.append(object_cnt)

                # clear total of objects of previous video
                total_objects.clear()

                # Reset out
                out = None

                # Reset trackers
                self.init_tracker()
                self.init_other_trackers()

        # Generate Report
        gen_report(gt, object_cnt_all)

        print('Time to process', time.time() - start_time)

    def detect_n_counting(self, origimg, img, loader=None):
        """Do object detection over 1 image."""
        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        raw_img = origimg.copy()

        # Applies yolov3 detection
        with torch.no_grad():
            detections, _ = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]

        if detections is None:
            return [], [], []

        box, conf, cls = boxes_filtering(raw_img, detections, self.img_size, cls_out=self.cls_out,
                                         mode=self.resize_mode)

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

            if cls == 0:
                # person counting
                features = self.encoder(raw_img, cls_boxes)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(cls_boxes, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                detections = [detections[i] for i in indices]

                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    bbox = track.to_tlbr().astype(int)

                    # save tracked list
                    self.save_probe_dir(video_id=os.path.basename(loader.path).split('.')[0][1:],
                                        track_id=track.track_id, raw_img=raw_img, bbox=bbox)

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    cv2.rectangle(origimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[cls], 2)
                    cv2.putText(origimg, str(track.track_id), (bbox[0], bbox[1]), 0, 5e-3 * 200, (0, 255, 0), 2)

                    # write coordinates
                    self.write_coordinates(loader=loader, x=bbox[0], y=bbox[1], w=bbox[2] - bbox[0],
                                           h=bbox[3] - bbox[1], cls=cls, track_id=track.track_id)
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

                if cls in [2, 5, 7]:
                    if track.hits >= 6:
                        total += 1
                else:
                    if track.hits >= 4:
                        total += 1

            total_objects[cls] = total
        else:
            # Process people
            total = 0

            for track in self.tracker.tracks:
                bbox = track.init_bbox.astype(int)

                if self.track_dir is not None:
                    f.write('{},{}, xmin={}, ymin={}, xmax={}, ymax={}, width={}, height={}\n'.format(
                        0, track, bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1], bbox[2], bbox[3]))

                if track.hits >= 6:
                    total += 1

            total_objects[0] = total

    def gen_total_objects(self, total_objects):
        """Generate total number of objects of each output class"""
        res = []

        for cls in self.cls_out:
            if cls not in total_objects:
                total_objects[cls] = 0

        # for Person
        res.append(total_objects[0])
        # for Fire extinguisher
        res.append(0)
        # for Fire hydrant
        res.append(total_objects[10])
        # for Vehicles
        res.append(total_objects[2] + total_objects[5] + total_objects[7])
        # for bicycle
        res.append(total_objects[1])
        # for motorbike
        res.append(total_objects[3])

        return res

    def save_probe_dir(self, video_id, track_id, raw_img, bbox):
        """ save query images in probe directory"""
        if self.save_probe:
            new_track_id = convert_number_to_image_form(int(track_id), start_digit='2', max_length=3)
            dir = 'tracking_images/{}/{}'.format(video_id, new_track_id)

            # create folder if not exist
            mkdir_if_missing(dir)

            # write images
            obj_img = raw_img[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
            h, w, _ = obj_img.shape

            if h == 0 or w == 0:
                return

            obj_img = cv2.resize(obj_img, (128, 256), interpolation=cv2.INTER_LINEAR)

            cur_idx = len(os.listdir(dir))
            cv2.imwrite('{}/{}C1T{}F{}.jpg'.format(dir, new_track_id, new_track_id,
                                                   convert_number_to_image_form(cur_idx, start_digit='', max_length=3)),
                        obj_img)
