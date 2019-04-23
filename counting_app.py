from flask import Flask, render_template, Response, request, flash, redirect
from tracking.deep_sort import nn_matching
from tracking.deep_sort.tracker import Tracker
from tracking.tools import generate_detections as gdet
from detection.utils.datasets import LoadCamera
from detection.utils.commons import load_cls_dict
from detection.utils.visualization import BBoxVisualization
from counting_objects_handler import PersonHandler
from args import parse_args
import cv2
import os
from re_id.reid.utils.data.iotools import mkdir_if_missing

app = Flask(__name__)

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

        return render_template('person_counting.html')


print('Load Input Arguments')
args = parse_args()

print('Load Tracker ...')
encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)

print('Load Object Detection model ...')
person_handler = PersonHandler(args)

print('Load Label Map')
cls_dict = load_cls_dict(args.data_path)

# grab image and do object detection (until stopped by user)
print('starting to loop and detect')
vis = BBoxVisualization(cls_dict)


@app.route('/video_feed', methods=['GET'])
def video_feed():
    global loader
    loader = LoadCamera(file_path[0], args.img_size)

    out = None
    if args.is_saved:
        mkdir_if_missing(args.save_dir)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('{}/{}.avi'.format(args.save_dir, os.path.basename(file_path[0][:-4])), fourcc, 10,
                              (args.image_width, args.image_height), True)
    tracker = Tracker(metric)

    return Response(person_handler.process(loader, tracker, encoder, out),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=7000, host="localhost")
