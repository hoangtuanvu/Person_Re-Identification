import cv2
import numpy as np
import torch
import os
import json
import math


def img_transform(imgs, new_shape):
    """
    Transform images by applying some operators like normalize, resize, ...
    :param imgs: input images
    :param new_shape: new size for resize images
    :return:
    """
    aug_imgs = []
    for img in imgs:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        aug_imgs.append(img)

    _ = torch.from_numpy(np.stack(aug_imgs)).float().to(torch.device('cpu'))

    return _


def normalize(img, mean=None, std=None):
    """
    Normalize single image by standard and mean
    :param img: input image in numpy array
    :param mean:
    :param std:
    :return:
    """
    if mean is None or std is None:
        return img

    assert len(mean) == 3 and len(std) == 3

    def sub(x, value):
        return x - value

    def div(x, value):
        return x / value

    c0 = div(sub(img[0], mean[0]), std[0])
    c1 = div(sub(img[1], mean[1]), std[1])
    c2 = div(sub(img[2], mean[2]), std[2])
    img = np.stack([c0, c1, c2], axis=0)
    return img


def matching(qf, gf, threshold=3.7):
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
                bb_ret.append([query_idx, res[0][1]])

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

    cv2.putText(img, fps_text, (11, 50), font, 2.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 50), font, 2.0, (240, 240, 240), 1, line)

    if not only_fps:
        cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)


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


def visualize_box(box):
    """
    Convert box with (x, y, w, h) format to (y, x, y + h, x + w) and update current box for negative coordinates
    :param box: output box from detection model
    :return: visualization box
    """
    vis_box = []
    for i in range(len(box)):
        x = box[i][0] if box[i][0] >= 0 else 0
        box[i][0] = x
        y = box[i][1] if box[i][1] >= 0 else 0
        box[i][1] = y
        w = box[i][2]
        h = box[i][3]
        vis_box.append([y, x, y + h, x + w])

    return vis_box


def load_cls_out(file_path, cls_dict):
    if not os.path.exists(file_path):
        raise ValueError('File does not exist')

    with open(file_path) as f:
        classes = [cls.replace("\n", "") for cls in f.readlines()]

    cls = []
    for cls_name in classes:
        if cls_name in cls_dict.values():
            cls.append(list(cls_dict.values()).index(cls_name))
        else:
            print('Class {} does not exist in Coco Classes'.format(cls_name))

    return cls


def read_counting_gt(gt_file):
    if not os.path.exists(gt_file):
        raise ValueError('Ground Truth file is not exist!')

    return json.load(open(gt_file))['track1_GT']


def rms(gt, pred):
    person_err = 5 * (abs(gt[0] - pred[0]) ** 2)
    other_errs = sum([abs(gt[i] - pred[i]) ** 2 for i in range(1, len(gt))])
    return math.sqrt(person_err + other_errs)


def convert_number_to_image_form(numb, start_digit='0', max_length=4):
    assert len(str(numb)) < max_length + 1

    return '{}{}'.format(str(start_digit) + '0' * (max_length - len(str(numb))), numb)


def filter_negative_values(ret):
    for i in range(len(ret)):
        ret[i] = ret[i] if ret[i] >= 0 else 0
