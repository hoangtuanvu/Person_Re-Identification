from tracking.deep_sort import nn_matching
from tracking.tools import generate_detections as gdet
from detection.utils.commons import load_cls_dict
from detection.utils.datasets import LoadImages
from counting_objects_handler import PersonHandler
from processor.utilities import load_cls_out
from args import parse_args
from re_id.reid.utils.data.iotools import mkdir_if_missing

print('Load Input Arguments')
args = parse_args()

print('Load Tracker ...')
encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)

print('Load Label Map')
cls_dict = load_cls_dict(args.data_path)
cls_out = load_cls_out(args.cls_out, cls_dict)
print(cls_out)

coordinates_out = None
if args.save_coordinates:
    coordinates_out = open('coordinates.txt', 'w')

print('Load Object Detection model ...')
person_handler = PersonHandler(args, encoder=encoder, cls_out=cls_out, metric=metric, coordinates_out=coordinates_out)
if args.is_saved:
    mkdir_if_missing(args.save_dir)
    person_handler.set_saved_dir(args.save_dir)

if args.save_tracks:
    mkdir_if_missing(args.track_dir)
    person_handler.set_track_dir(args.track_dir)

# person_handler.set_out(out)
person_handler.set_colors()
person_handler.init_tracker()
person_handler.init_other_trackers()

print('Process Videos in Offline Mode')
loader = LoadImages(args.inputs, img_size=args.img_size, resize_mode=args.mode)
person_handler.offline_process(loader)