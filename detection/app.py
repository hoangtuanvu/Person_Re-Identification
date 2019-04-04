import os
import re
from handler import PersonHandler
from args import parse_args
from utils.datasets import LoadCamera
from utils.commons import load_cls_dict
from utils.parse_config import parse_data_config
from utils.visualization import BBoxVisualization
from flask import Flask, render_template, Response, request, flash, redirect

MEASURE_MODEL_TIME = False

app = Flask(__name__)

file_path = []
loader = None


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

print('Load Object Detection model ...')
person_handler = PersonHandler(args)

print('Load Label Map')
cls_dict = load_cls_dict(parse_data_config(args.data_path)['names'])

# grab image and do object detection (until stopped by user)
print('starting to loop and detect')
vis = BBoxVisualization(cls_dict)


@app.route('/video_feed', methods=['GET'])
def video_feed():
    global loader
    loader = LoadCamera(file_path[0], args.img_size)
    return Response(person_handler.loop_and_detect(loader, vis), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=7000, host="localhost")
