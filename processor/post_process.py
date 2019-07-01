import xlwt
import cv2
import os
import numpy as np
from .utilities import filter_negative_values
from .utilities import convert_number_to_image_form
from re_id.reid.utils.data.iotools import mkdir_if_missing


def save_probe_dir(video_id, track_id, raw_img, bbox):
    """ save query images in probe directory"""
    new_track_id = convert_number_to_image_form(int(track_id), start_digit='2', max_length=3)
    dir = 'tracking_images/{}/{}'.format(video_id, new_track_id)

    # create folder if not exist
    mkdir_if_missing(dir)

    # write images
    obj_img = raw_img[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
    h, w, _ = obj_img.shape

    if h == 0 or w == 0:
        return

    obj_img = cv2.resize(obj_img, (128, 256), interpolation=cv2.INTER_LINEAR)

    cur_idx = len(os.listdir(dir))
    cv2.imwrite('{}/{}C1T{}F{}.jpg'.format(dir, new_track_id, new_track_id,
                                           convert_number_to_image_form(cur_idx, start_digit='', max_length=3)),
                obj_img)


def ct_boxes_filer(dets, cls_out, conf_th):
    out_box = []
    out_conf = []
    out_cls = []

    for cls_id in dets:
        for x1, y1, x2, y2, conf in dets[cls_id]:
            if conf < conf_th:
                continue

            if (cls_id - 1) in cls_out and len(dets[cls_id]) > 0:
                w = x2 - x1
                h = y2 - y1

                # Class 0 (pedestrian)
                if cls_id - 1 == 0 and min(w, h) <= 10:
                    continue

                # Class Vehicle (Car, Truck, Bus)
                if cls_id - 1 in [1, 4, 5] and min(w, h) < 30:
                    continue

                box = [int(x1), int(y1), int(w), int(h)]
                filter_negative_values(box)
                out_box.append(box)
                out_conf.append(conf)
                out_cls.append(cls_id - 1)

    return out_box, out_conf, out_cls


def gen_report(gt, pred):
    book = xlwt.Workbook(encoding="utf-8")

    sheet1 = book.add_sheet("Sheet 1")

    sheet1.write(0, 0, "Folder")
    sheet1.write(0, 1, "Ground Truth")
    sheet1.write(0, 2, "Detected object")
    sheet1.write(0, 3, "Error Score")

    for row in range(0, len(pred)):
        sheet1.write(row + 1, 0, pred[row]["name"])
        sheet1.write(row + 1, 1, ','.join(str(item) for item in gt[row]["objects"]))
        sheet1.write(row + 1, 2, ','.join(str(item) for item in pred[row]["objects"]))
        sheet1.write(row + 1, 3, pred[row]["rms"])

    book.save("statistics.xls")


def gen_total_objects(cls_out, total_objects, od_model):
    """Generate total number of objects of each output class"""
    res = []

    for cls in cls_out:
        if cls not in total_objects:
            total_objects[cls] = 0

    # for Person
    res.append(total_objects[0])

    # for Fire extinguisher
    res.append(0)

    if od_model == 'yolo':
        # for Fire hydrant
        res.append(total_objects[10])

        # for Vehicles
        res.append(total_objects[2] + total_objects[5] + total_objects[7])

        # for bicycle
        res.append(total_objects[1])

        # for motorbike
        res.append(total_objects[3])
    else:
        # for Fire hydrant
        res.append(0)

        # for Vehicles
        res.append(total_objects[1] + total_objects[4] + total_objects[5])

        # for bicycle
        res.append(total_objects[3])

        # for motorbike
        res.append(total_objects[2])

    return res


def boxes_filtering(img, detections, img_size, cls_out, mode='auto'):
    """
    Resize output boxes to original and get class = 0 (person)
    :param img: frame from video or camera live stream
    :param detections: output boxes from darknet yolov3
    :param img_size: resize of image
    :param cls_out: list of categories need to get
    :param mode: resize mode
    """
    out_box = []
    out_conf = []
    out_cls = []
    h, w, _ = img.shape
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections.cpu().numpy():
        if int(cls_pred) in cls_out:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * h
            box_w = ((x2 - x1) / unpad_w) * w
            if mode == 'square':
                y1 = ((y1 - pad_y // 2) / unpad_h) * h
                x1 = ((x1 - pad_x // 2) / unpad_w) * w
            else:
                y1 = ((y1 - np.mod(pad_y, 32) // 2) / unpad_h) * h
                x1 = ((x1 - np.mod(pad_x, 32) // 2) / unpad_w) * w

            if cls_pred == 0 and (conf <= 0.6 or min(box_h, box_w) <= 10):
                continue

            if cls_pred in [2, 5, 7] and (conf <= 0.55 or box_w < 30 or box_h < 30):
                continue

            box = [int(x1), int(y1), int(box_w), int(box_h)]
            filter_negative_values(box)
            out_box.append(box)
            out_conf.append(conf)
            out_cls.append(int(cls_pred))

    return out_box, out_conf, out_cls
