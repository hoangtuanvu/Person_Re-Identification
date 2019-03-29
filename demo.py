from detection.trt_models.detection import read_label_map
import cv2
from detection.utils.camera import Camera
from detection.utils.visualization import BBoxVisualization
from person_handler_with_opencv import PersonHandler
from tracking.tools import generate_detections as gdet
from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort import nn_matching
from args import parse_args


def main():
    print('Load Input Arguments')
    args = parse_args()

    print('Load Camera Type')
    # cam = Camera(args)
    # # args.use_file = True
    # # args.filename = 'TownCentreXVID.avi'
    # cam.open()
    cam = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_5.avi', fourcc, 20, (1920, 1080), True)

    print('Load Tracker ...')
    encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)
    tracker = Tracker(metric)

    print('Load Object Detection model ...')
    person_handler = PersonHandler(args)
    # person_handler.open_display_window(cam.img_width, cam.img_height)

    print('Load Label Map')
    cls_dict = read_label_map(args.labelmap_file)

    # if not cam.is_opened():
    #     sys.exit('Failed to open camera!')

    od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'
    # cam.start()  # ask the camera to start grabbing images

    # grab image and do object detection (until stopped by user)
    print('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)

    person_handler.loop_and_detect(cam, vis, od_type, tracker, encoder, out=out)

    print('cleaning up')
    # cam.stop()  # terminate the sub-thread in camera
    person_handler.sess.close()
    cam.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
