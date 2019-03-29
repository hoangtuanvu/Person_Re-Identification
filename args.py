import argparse


def parse_args():
    """Parse input arguments."""
    # ******************************************************************************************************************
    # Datasets (general)
    # ******************************************************************************************************************
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
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    # ******************************************************************************************************************
    # Detection
    # ******************************************************************************************************************
    parser.add_argument('--model', dest='model',
                        help='tf-trt object detection model '
                             '[{}]'.format('faster_rcnn_inception_v2'),
                        default='faster_rcnn_inception_v2', type=str)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.3, type=float)
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format('detection/third_party/models/research/object_detection/'
                                           'data/mscoco_label_map.pbtxt'),
                        default='detection/third_party/models/research/object_detection/'
                                'data/mscoco_label_map.pbtxt', type=str)
    parser.add_argument('--detection-path', dest='detection_weights',
                        help='Person detection model',
                        default='data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb', type=str)
    # ******************************************************************************************************************
    # Tracking
    # ******************************************************************************************************************
    parser.add_argument('--tracker-path', dest='tracker_weights',
                        help='[{}]'.format('tracking/model_data/mars-small128.pb'),
                        default='tracking/model_data/mars-small128.pb', type=str)
    parser.add_argument('--tracking-type', dest='tracking_type',
                        help='use sort or deep sort for tracking person',
                        default='', type=str)
    parser.add_argument('--max-cosine-distance', dest='max_cosine_distance',
                        help='maximun cosine distance [0.3]',
                        default=0.3, type=float)
    # ******************************************************************************************************************
    # Re-Identification
    # ******************************************************************************************************************
    parser.add_argument('--use-cpu', action='store_true', help='use cpu')
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--load-weights', type=str,
                        default='/mnt/sda3/PersonREID/checkpoint/reid/market_duke_cuhk03_full_not_pretrained/checkpoint.pth.tar',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--matching-threshold', dest='matching_threshold',
                        help='distance threshold between query image and gallery',
                        default=7.0, type=float)
    parser.add_argument('--num-classes', dest='num_classes', default=751,
                        help='(deprecated and not used) number of object classes', type=int)
    args = parser.parse_args()
    return args
