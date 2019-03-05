from detection.trt_models.detection import obj_det_graph
import cv2
import time
import numpy as np
from tracking.deep_sort import preprocessing
from tracking.deep_sort.detection import Detection

WINDOW_NAME = 'CameraDemo'
DEFAULT_LABELMAP = 'third_party/models/research/object_detection/' \
                   'data/mscoco_label_map.pbtxt'
MEASURE_MODEL_TIME = False


class PersonHandler:
    def __init__(self, args):
        self.use_tracking = args.use_tracking
        self.sess, self.scores, self.boxes, self.classes, self.input = self.load_model(args.model)

    @staticmethod
    def load_model(model):
        tf_sess, tf_scores, tf_boxes, tf_classes, tf_input = obj_det_graph(model, 'data', 'data/{}.pb'.format(model))
        return tf_sess, tf_scores, tf_boxes, tf_classes, tf_input

    def loop_and_detect(self, cam, conf_th, vis, od_type, tracker, encoder):
        """Loop, grab images from camera, and do object detection.

        # Arguments
          cam: the camera object (video source).
          tf_sess: TensorFlow/TensorRT session to run SSD object detection.
          conf_th: confidence/score threshold for object detection.
          vis: for visualization.
        """
        show_fps = True
        full_scrn = False
        fps = 0.0
        tic = time.time()
        while cam.thread_running:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                # Check to see if the user has closed the display window.
                # If yes, terminate the while loop.
                break

            img = cam.read()
            if img is not None:
                box, conf, cls = self.detect(img, conf_th, encoder, tracker, od_type=od_type)

                img = vis.draw_bboxes(img, box, conf, cls)
                if show_fps:
                    img = self.draw_help_and_fps(img, fps)
                cv2.imshow(WINDOW_NAME, img)
                toc = time.time()
                curr_fps = 1.0 / (toc - tic)
                # calculate an exponentially decaying average of fps number
                fps = curr_fps if fps == 0.0 else (fps * 0.9 + curr_fps * 0.1)
                tic = toc

            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
            elif key == ord('H') or key == ord('h'):  # Toggle help/fps
                show_fps = not show_fps
            elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
                full_scrn = not full_scrn
                self.set_full_screen(full_scrn)

    def detect(self, origimg, conf_th, encoder, tracker, od_type='ssd'):
        """Do object detection over 1 image."""
        global avg_time

        if od_type == 'faster_rcnn':
            img = self.resize(origimg, (1024, 576))
        elif od_type == 'ssd':
            img = self.resize(origimg, (300, 300))
        else:
            raise ValueError('bad object detector type: $s' % od_type)

        if MEASURE_MODEL_TIME:
            tic = time.time()

        boxes_out, scores_out, classes_out = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={self.input: img[None, ...]})

        h, w, _ = origimg.shape
        boxs = self.convert_boxes(boxes_out[0], h, w, classes_out[0])
        if len(boxs) == 0:
            return [], [], []

        if self.use_tracking:
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

        box, conf, cls = self.person_filtering(origimg, boxes_out, scores_out, classes_out, conf_th)

        return box, conf, cls

    @staticmethod
    def open_display_window(width, height):
        """Open the cv2 window for displaying images with bounding boxeses."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, width, height)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, 'Camera TF-TRT Person Detection Demo')

    @staticmethod
    def draw_help_and_fps(img, fps):
        """Draw help message and fps number at top-left corner of the image."""
        help_text = "'Esc' to Quit, 'H' for FPS & Help, 'F' for Fullscreen"
        font = cv2.FONT_HERSHEY_PLAIN
        line = cv2.LINE_AA

        fps_text = 'FPS: {:.1f}'.format(fps)
        cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
        cv2.putText(img, fps_text, (11, 50), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 50), font, 1.0, (240, 240, 240), 1, line)
        return img

    @staticmethod
    def set_full_screen(full_scrn):
        """Set display window to full screen or not."""
        prop = cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

    @staticmethod
    def resize(src, shape=None, to_rgb=True):
        """Preprocess input image for the TF-TRT object detection model."""
        img = src.astype(np.uint8)
        if shape:
            img = cv2.resize(img, shape)
        if to_rgb:
            # BGR to RGB
            img = img[..., ::-1]
        return img

    @staticmethod
    def convert_boxes(boxs, h_, w_, classes):
        return_boxs = []
        for i in range(len(boxs)):
            if classes[i] != 1:
                continue
            box = boxs[i]
            x = int(box[1] * w_)
            y = int(box[0] * h_)
            w = int((box[3] - box[1]) * w_)
            h = int((box[2] - box[0]) * h_)
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            if x == 0 and y == 0 and w == 0 and h == 0:
                continue
            return_boxs.append([x, y, w, h])
        return return_boxs

    @staticmethod
    def person_filtering(img, boxes, scores, classes, conf_th):
        """Process ouput of the TF-TRT person detector."""
        h, w, _ = img.shape
        out_box = boxes[0] * np.array([h, w, h, w])
        out_box = out_box.astype(np.int32)
        out_conf = scores[0]
        out_cls = classes[0].astype(np.int32)

        _out_box = []
        _out_conf = []
        _out_cls = []
        for i in range(len(out_conf)):
            if out_conf[i] > conf_th and out_cls[i] == 1:
                _out_box.append(out_box[i])
                _out_conf.append(out_conf[i])
                _out_cls.append(out_cls[i])

        return _out_box, _out_conf, _out_cls
