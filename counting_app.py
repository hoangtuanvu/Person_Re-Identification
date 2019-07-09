from flask import Flask, render_template, Response, request, flash, redirect
from tracking.deep_sort import nn_matching
from tracking.tools import generate_detections as gdet
from detection.utils.datasets import LoadCamera
from detection.utils.commons import load_cls_dict
from counting_objects_handler import PersonHandler
from processor.utilities import load_cls_out
from args import parse_args
from opts import opts
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
args = opts().init()

print('Load Tracker ...')
encoder = gdet.create_box_encoder(model_filename=args.tracker_weights, batch_size=8)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance)

print('Load Label Map')
cls_dict = load_cls_dict(args.data_path)
cls_out = load_cls_out(args.cls_out, cls_dict)

print('Load Object Detection model ...')
person_handler = PersonHandler(args, encoder=encoder, cls_out=cls_out, metric=metric)
person_handler.set_colors()


@app.route('/video_feed', methods=['GET'])
def video_feed():
    global loader
    loader = LoadCamera(file_path[0], img_size=args.img_size, resize_mode=args.mode)

    out = None
    if args.is_saved:
        mkdir_if_missing(args.save_dir)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('{}/{}.avi'.format(args.save_dir, os.path.basename(file_path[0][:-4])), fourcc, 10,
                              (args.image_width, args.image_height), True)

    person_handler.set_out(out)
    person_handler.init_tracker()
    person_handler.init_other_trackers()

    return Response(person_handler.online_process(loader),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=args.port, host="localhost")
