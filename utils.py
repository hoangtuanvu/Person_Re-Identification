import cv2
import numpy as np
import torch
from re_id.reid.feature_extraction import extract_cnn_feature


def matching(qf, gf, threshold=3.0):
    """
    Do matching algorithm to compute distance between query embedding and gallery embeddings
    :param threshold: threshold for filtering
    :param qf: query embedding
    :param gf: gallery embeddings
    :return: index of the most matching embedding of gallery embeddings
    """
    x = torch.cat([qf[i].unsqueeze(0) for i in range(len(qf))], 0)
    y = torch.cat([gf[i].unsqueeze(0) for i in range(len(gf))], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    bb_ret = []
    for query_idx in range(len(dist)):
        res = [[dist[query_idx][bb_idx], bb_idx] for bb_idx in range(len(dist[query_idx]))]
        print(res)
        res = sorted(res, key=lambda x: x[0])

        if float(res[0][0]) < threshold:
            if res[0][1] not in bb_ret:
                bb_ret.append(res[0][1])

    return bb_ret


def open_display_window(width, height, window_name):
    """Open the cv2 window for displaying images with bounding boxeses."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowTitle(window_name, 'Camera Person Re-Identification Demo')


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


def set_full_screen(full_scrn, window_name):
    """Set display window to full screen or not."""
    prop = cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)


def resize(src, shape=None, to_rgb=True):
    """Preprocess input image for the TF-TRT object detection model."""
    img = src.astype(np.uint8)
    if shape:
        img = cv2.resize(img, shape)
    if to_rgb:
        # BGR to RGB
        img = img[..., ::-1]
    return img


def convert_boxes(boxs):
    """Convert boxes into another representation"""
    return_boxs = [[box[1], box[0], box[3] - box[1], box[2] - box[0]] for box in boxs]
    return return_boxs


def person_filtering(img, boxes, scores, classes, conf_th):
    """
    Process output of the Person Detection model.
    :param img: input image
    :param boxes: output boxes from
    :param scores:
    :param classes:
    :param conf_th:
    :return:
    """
    """"""
    h, w, _ = img.shape
    out_box = boxes[0] * np.array([h, w, h, w])
    out_box = out_box.astype(np.int32)
    out_conf = scores[0]
    out_cls = classes[0].astype(np.int32)

    _out_box = []
    _out_conf = []
    _out_cls = []
    for i in range(len(out_conf)):
        if out_conf[i] > conf_th and out_cls[i] == 1:
            _out_box.append(out_box[i])
            _out_conf.append(out_conf[i])
            _out_cls.append(out_cls[i])

    return _out_box, _out_conf, _out_cls


def inference(loader, model, use_gpu):
    with torch.no_grad():
        embedding = []

        dataloader_iterator = iter(loader)
        for i in range(len(loader)):
            try:
                imgs = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(loader)
                imgs = next(dataloader_iterator)

            if use_gpu:
                imgs = imgs.cuda()

            features = extract_cnn_feature(model, imgs)
            embedding.extend(list(features))
    return embedding
