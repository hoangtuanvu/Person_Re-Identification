import argparse


def parse_args():
    """Parse input arguments."""
    # ******************************************************************************************************************
    # Datasets (general)
    # ******************************************************************************************************************
    desc = ('This script captures and displays live camera video, '
            'and does real-time person detection with TF-TRT model')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epochs', dest='epochs',
                        type=int, default=270, help='number of epochs')
    parser.add_argument('--accumulate', dest='accumulate',
                        type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--multi-scale', dest='multi_scale',
                        action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--resume', dest='resume',
                        action='store_true', help='resume training flag')
    parser.add_argument('--transfer', dest='transfer',
                        action='store_true', help='transfer learning flag')
    parser.add_argument('--dist-url', dest='dist_url',
                        default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', dest='rank',
                        default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', dest='world_size',
                        default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', dest='backend',
                        default='nccl', type=str, help='distributed backend')
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
                        default='config/yolov3.cfg', type=str)
    parser.add_argument('--weight-path', dest='weight_path',
                        help='Person detection model',
                        default='weights/yolov3.weights', type=str)
    parser.add_argument('--data-path', dest='data_path',
                        default='config/coco.data', type=str)
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=1, help='size of the batches')
    parser.add_argument('--n-cpu', dest='n_cpu',
                        type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', dest='img_size',
                        type=int, default=416, help='size of each image dimension')
    parser.add_argument('--iou-thres', dest='iou_thres',
                        type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--save-json', dest='save_json',
                        action='store_true', help='save a coco api-compatible JSON results file')
    parser.add_argument('--use-cpu', dest='use_cpu',
                        action='store_true', help='force model use cpu for inference')
    parser.add_argument('--images', dest='images',
                        type=str, default='data/samples', help='path to images')
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
    args = parser.parse_args()
    return args
