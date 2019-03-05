from detection.trt_models.detection import read_label_map
import cv2
import sys
import argparse
from detection.utils.camera import Camera
from detection.utils.visualization import BBoxVisualization
from detection.trt_models.person_handler import PersonHandler

from tracking.tools import generate_detections as gdet
from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort import nn_matching

WINDOW_NAME = 'CameraDemo'
DEFAULT_LABELMAP = 'detection/third_party/models/research/object_detection/' \
                   'data/mscoco_label_map.pbtxt'
DEFAULT_TRACKER_PATH = 'tracking/model_data/mars-small128.pb'


def parse_args():
    """Parse input arguments."""
    desc = ('This script captures and displays live camera video, '
            'and does real-time person detection with TF-TRT model')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file', dest='use_file',
                        help='use a video file as input (remember to '
                             'also set --filename)',
                        action='store_true')
    parser.add_argument('--image', dest='use_image',
                        help='use an image file as input (remember to '
                             'also set --filename)',
                        action='store_true')
    parser.add_argument('--filename', dest='filename',
                        help='video file name, e.g. test.mp4',
                        default=None, type=str)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--onboard', dest='use_onboard',
                        help='use USB webcams',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format(DEFAULT_LABELMAP),
                        default=DEFAULT_LABELMAP, type=str)
    parser.add_argument('--tracker-path', dest='tracker_weights',
                        help='[{}]'.format(DEFAULT_TRACKER_PATH),
                        default=DEFAULT_TRACKER_PATH, type=str)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    parser.add_argument('--model', dest='model',
                        help='tf-trt object detection model '
                             '[{}]'.format('ssd_mobilenet_v1_coco'),
                        default='ssd_mobilenet_v1_coco', type=str)
    parser.add_argument('--build', dest='do_build',
                        help='re-build TRT pb file (instead of using'
                             'the previously built version)',
                        action='store_true')
    parser.add_argument('--tensorboard', dest='do_tensorboard',
                        help='write optimized graph summary to TensorBoard',
                        action='store_true')
    parser.add_argument('--num-classes', dest='num_classes',
                        help='(deprecated and not used) number of object '
                             'classes', type=int)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.3, type=float)
    parser.add_argument('--max-cosine-distance', dest='max_cosine_distance',
                        help='maximun cosine distance [0.3]',
                        default=0.3, type=float)
    parser.add_argument('--tracking', dest='use_tracking',
                        help='use or not use tracking',
                        default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    print('Load Input Arguments')
    args = parse_args()

    print('Load Camera Type')
    cam = Camera(args)
    cam.open()

    print('Load Tracker ...')
    encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)
    tracker = Tracker(metric)

    print('Load Object Detection model ...')
    person_handler = PersonHandler(args)
    person_handler.open_display_window(cam.img_width, cam.img_height)

    print('Load Label Map')
    cls_dict = read_label_map(args.labelmap_file)

    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'
    cam.start()  # ask the camera to start grabbing images

    # grab image and do object detection (until stopped by user)
    print('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)

    person_handler.loop_and_detect(cam, args.conf_th, vis, od_type, tracker, encoder)

    print('cleaning up')
    cam.stop()  # terminate the sub-thread in camera
    person_handler.sess.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
