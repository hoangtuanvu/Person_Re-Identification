from model.yolov3 import Darknet
from utils.commons import *
from utils.datasets import *
from torch.backends import cudnn
import time

MEASURE_MODEL_TIME = False


class PersonHandler:
    def __init__(self, args):
        self.tracking_type = args.tracking_type
        self.conf_th = args.conf_th
        self.nms_thres = args.nms_thres
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 150, 255),
                       (0, 255, 255), (200, 0, 200), (255, 191, 0), (180, 105, 255)]
        self.img_size = args.img_size
        self.memory = {}

        # Load model Detection
        self.use_gpu = torch.cuda.is_available()
        if args.use_cpu:
            self.use_gpu = False

        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        detect_model = Darknet(args.config_path, img_size=self.img_size, device=self.device)
        detect_model.load_darknet_weights(args.weight_path)

        self.detect_model = nn.DataParallel(detect_model).cuda() if self.use_gpu else detect_model
        self.detect_model.eval()
        cudnn.benchmark = True

    def loop_and_detect(self, loader, vis):
        """Loop, grab images from camera, and do object detection.

        # Arguments
          cam: the camera object (video source).
          tf_sess: TensorFlow/TensorRT session to run SSD object detection.
          conf_th: confidence/score threshold for object detection.
          vis: for visualization.
        """
        show_fps = True
        fps = 0.0
        tic = time.time()

        for i, (img, img0) in enumerate(loader):
            display = np.array(img0)
            output = display.copy()
            box, conf, cls = self.detect(output, img)

            if not self.tracking_type:
                output = vis.draw_bboxes(output, box, conf, cls)

            if show_fps:
                output = self.draw_help_and_fps(output, fps, True)

            cv2.imwrite('demo.jpg', output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

            # Calculate FPS
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps * 0.9 + curr_fps * 0.1)
            tic = toc

    def detect(self, origimg, img):
        """Do object detection over 1 image."""
        global avg_time

        if MEASURE_MODEL_TIME:
            tic = time.time()

        input_imgs = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # Get detections
        with torch.no_grad():
            start_time = time.time()
            detections = self.detect_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_th, self.nms_thres)[0]
            print('Time to predict', time.time() - start_time)

        if detections is None:
            return [], [], []

        _box, _conf, cls = boxes_filtering(origimg, detections, self.img_size)

        vis_box = []
        for i in range(len(_box)):
            x1, y1, w, h = _box[i]
            vis_box.append([y1, x1, y1 + h, x1 + w])

        if len(_box) == 0:
            return [], [], []

        if MEASURE_MODEL_TIME:
            td = (time.time() - tic) * 1000  # in ms
            avg_time = avg_time * 0.9 + td * 0.1
            print('tf_sess.run() took {:.1f} ms on average'.format(avg_time))

        return vis_box, _conf, cls

    @staticmethod
    def draw_help_and_fps(img, fps, only_fps=False):
        """Draw help message and fps number at top-left corner of the image."""
        help_text = "'Esc' to Quit, 'H' for FPS & Help, 'F' for Fullscreen"
        font = cv2.FONT_HERSHEY_PLAIN
        line = cv2.LINE_AA

        fps_text = 'FPS: {:.1f}'.format(fps)

        cv2.putText(img, fps_text, (11, 50), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 50), font, 1.0, (240, 240, 240), 1, line)

        if not only_fps:
            cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
            cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)

        return img
