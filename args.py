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
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    parser.add_argument('--use-resize', dest='use_resize',
                        action='store_true', help='resize output image for improving FPS')
    parser.add_argument('--show-fps', dest='show_fps',
                        action='store_true', help='show FPS on each frame of video or camera streaming')
    parser.add_argument('--save-probe', dest='save_probe',
                        action='store_true', help='save query images automatically')
    parser.add_argument('--freq', default=5,
                        type=int, help='save query images by frequency')
    # ******************************************************************************************************************
    # Detection
    # ******************************************************************************************************************
    parser.add_argument('--confidence', dest='conf_th',
                        help='object confidence threshold',
                        default=0.2, type=float)
    parser.add_argument('--nms-thres', dest='nms_thres',
                        help='iou threshold for non-maximum suppression',
                        default=0.3, type=float)
    parser.add_argument('--config-path', dest='config_path',
                        default='detection/config/yolov3.cfg', type=str)
    parser.add_argument('--detection-weight', dest='detection_weight',
                        help='Person detection model',
                        default='detection/weights/yolov3.weights', type=str)
    parser.add_argument('--data-path', dest='data_path',
                        default='detection/data/coco.names', type=str)
    parser.add_argument('--img-size', dest='img_size',
                        type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use-cpu', dest='use_cpu',
                        action='store_true', help='force model use cpu for inference')
    parser.add_argument('--images', dest='images',
                        type=str, default='data/samples', help='path to images')
    parser.add_argument('--port', dest='port', help='opening port for application',
                        default=7000, type=int)
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
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--reid-weights', type=str,
                        default='re_id/weights/checkpoint.pth.tar',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--matching-threshold', dest='matching_threshold',
                        help='distance threshold between query image and gallery',
                        default=3.7, type=float)
    parser.add_argument('--num-classes', dest='num_classes', default=9511,
                        help='(deprecated and not used) number of object classes', type=int)
    # ******************************************************************************************************************
    # Counting
    # ******************************************************************************************************************
    parser.add_argument('--counting-use-reid', dest='counting_use_reid',
                        action='store_true', help='use person re-identification for counting person')
    parser.add_argument('--save-dir', dest='save_dir',
                        default='.', type=str)
    parser.add_argument('--is-saved', dest='is_saved',
                        action='store_true', help='saves video output for person counting problem')
    parser.add_argument('--cls-out', dest='cls_out',
                        default='cls_out.txt', type=str)
    parser.add_argument('--inputs', type=str,
                        help='input for counting objects. It can be videos, images, directory of videos/images')
    parser.add_argument('--output-name', dest='output_name', type=str,
                        default='predict', help='name of output file')
    parser.add_argument('--gt', default='t1_gt.json',
                        help='Ground truth json file', type=str)
    parser.add_argument('--save-coordinates',
                        action='store_true', help='saves coordinates of all objects in a frame with tracking ID')
    args = parser.parse_args()
    return args
