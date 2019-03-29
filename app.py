from flask import Flask, render_template, Response, request, flash, redirect
import os
import cv2
import re
from tracking.deep_sort import nn_matching
from tracking.deep_sort.tracker import Tracker
from tracking.sort.sort import *
from tracking.tools import generate_detections as gdet
from detection.trt_models.detection import read_label_map
from detection.utils.visualization import BBoxVisualization
from person_handler import PersonHandler
from args import parse_args

MEASURE_MODEL_TIME = False

app = Flask(__name__)

cap = None
file_path = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('index.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request

        if 'files' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        file_obj = request.files.getlist("files")

        global file_path
        file_path = []

        for file in file_obj:
            # if filename is empty, then assume no upload
            if file.filename == '':
                flash('No file was uploaded.')
                return redirect(request.url)

            try:
                filename = file.filename
                file_path.append(os.path.join('./uploads', filename))
                if not os.path.exists('uploads'):
                    os.mkdir('uploads')

                file.save(file_path[-1])
                passed = True
            except ValueError:
                passed = False

            if not passed:
                flash('An error occurred, try again.')
                return redirect(request.url)

        if re.findall('([-\w]+\.(?:jpg|gif|png))', os.path.basename(file_path[0].lower())):
            return render_template('person_reid.html')
        else:
            return render_template('person_det.html')


print('Load Input Arguments')
args = parse_args()

print('Load Tracker ...')
tracker = None
encoder = None

if args.tracking_type == "sort":
    tracker = Sort()
elif args.tracking_type == "deep_sort":
    encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)
    tracker = Tracker(metric)

print('Load Object Detection model ...')
person_handler = PersonHandler(args)

print('Load Label Map')
cls_dict = read_label_map(args.labelmap_file)

od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'

# grab image and do object detection (until stopped by user)
print('starting to loop and detect')
vis = BBoxVisualization(cls_dict)


@app.route('/person_reid', methods=['GET'])
def person_reid():
    return Response(person_handler.loop_and_detect(cap, vis, od_type, tracker, encoder, file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    global cap
    cap = cv2.VideoCapture(file_path[0])
    return Response(person_handler.loop_and_detect(cap, vis, od_type, tracker, encoder, ''),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=7000, host="localhost")
