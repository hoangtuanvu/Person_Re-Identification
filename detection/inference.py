import shutil
import time
import torch
import os
import cv2
from pathlib import Path
from sys import platform
from args import parse_args
from model.yolov3 import Darknet
from utils.commons import load_cls_dict
from utils.commons import non_max_suppression
from utils.commons import boxes_filtering
from utils.datasets import LoadImages
from utils.parse_config import parse_data_config
from utils.visualization import BBoxVisualization


def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_images=True,
        use_cpu=True
):
    use_gpu = torch.cuda.is_available()
    if use_cpu:
        use_gpu = False

    device = torch.device('cuda' if use_gpu else 'cpu')

    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size, device)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        model.load_darknet_weights(weights)

    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    print('Load Label Map')
    cls_dict = load_cls_dict(parse_data_config(data_cfg)['names'])

    # grab image and do object detection (until stopped by user)
    print('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        pred = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if detections is not None and len(detections) > 0:
            box, conf, cls = boxes_filtering(im0, detections, img_size)

            vis_box = []
            for i in range(len(box)):
                x1, y1, w, h = box[i]
                vis_box.append([y1, x1, y1 + h, x1 + w])

            im0 = vis.draw_bboxes(im0, vis_box, conf, cls)

        print('Done. (%.3fs)' % (time.time() - t))

        if save_images:  # Save generated image with detections
            if dataloader.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
                vid_writer.write(im0)

            else:
                cv2.imwrite(save_path, im0)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    args = parse_args()

    with torch.no_grad():
        detect(
            args.config_path,
            args.data_path,
            args.weight_path,
            args.images,
            img_size=args.img_size,
            conf_thres=args.conf_th,
            nms_thres=args.nms_thres,
            use_cpu=args.use_cpu
        )
