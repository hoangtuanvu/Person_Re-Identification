from tracking.tools import generate_detections as gdet
from detection.utils.commons import load_cls_dict
from detection.utils.datasets import LoadImages
from counting_objects_handler import PersonHandler
from processor.utilities import load_cls_out
from re_id.reid.utils.data.iotools import mkdir_if_missing

from opts import opts

print('Load Input Arguments')
args = opts().init()

print('Load Person Appearance ...')
p_encoder = gdet.create_box_encoder(model_filename=args.p_tracker_weights, batch_size=8)

print('Load Vehicle Appearance ...')
v_encoder = gdet.create_box_encoder(model_filename=args.v_tracker_weights, batch_size=8)

print('Load Label Map')
cls_dict = load_cls_dict(args.data_path)
cls_out = load_cls_out(args.cls_out, cls_dict)
print(cls_out)

coordinates_out = None
if args.save_coordinates:
    coordinates_out = open('coordinates.txt', 'w')

print('Load Object Detection model ...')
person_handler = PersonHandler(args, p_encoder=p_encoder, v_encoder=v_encoder, cls_out=cls_out,
                               coordinates_out=coordinates_out)
if args.is_saved:
    mkdir_if_missing(args.save_vid)
    person_handler.set_saved_dir(args.save_vid)

if args.save_tracks:
    mkdir_if_missing(args.track_dir)
    person_handler.set_track_dir(args.track_dir)

person_handler.set_colors()
person_handler.init_tracker()

print('Process Videos in Offline Mode')
loader = LoadImages(args.inputs, img_size=args.img_size, resize_mode=args.mode, od_model=args.od_model)
person_handler.offline_process(loader)