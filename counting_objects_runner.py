from tracking.deep_sort import nn_matching
from tracking.tools import generate_detections as gdet
from detection.utils.commons import load_cls_dict
from detection.utils.datasets import LoadImages
from counting_objects_handler import PersonHandler
from utilities import load_cls_out
from args import parse_args

print('Load Input Arguments')
args = parse_args()

print('Load Tracker ...')
encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)

print('Load Label Map')
cls_dict = load_cls_dict(args.data_path)
cls_out = load_cls_out(args.cls_out, cls_dict)
print(cls_out)

print('Load Object Detection model ...')
person_handler = PersonHandler(args, encoder=encoder, cls_out=cls_out, metric=metric)
person_handler.set_colors()
person_handler.init_tracker()
person_handler.init_other_trackers()

print('Process Videos in Offline Mode')
loader = LoadImages(args.inputs, args.img_size)
person_handler.offline_process(loader)
