import torch
import json
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from args import parse_args
from model.yolov3 import Darknet
from torch.utils.data import DataLoader
from utils.parse_config import parse_data_config
from utils.datasets import LoadImagesAndLabels
from utils.commons import scale_coords
from utils.commons import xyxy2xywh
from utils.commons import xywh2xyxy
from utils.commons import float3
from utils.commons import bbox_iou
from utils.commons import ap_per_class
from utils.commons import load_classes
from utils.commons import coco80_to_coco91_class
from utils.commons import non_max_suppression


def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.1,
        nms_thres=0.5,
        save_json=False,
        use_cpu=True,
        model=None
):
    if model is None:
        use_gpu = torch.cuda.is_available()
        if use_cpu:
            use_gpu = False

        device = torch.device('cuda' if use_gpu else 'cpu')

        # Initialize model
        model = Darknet(cfg, img_size, device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            model.load_darknet_weights(weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg = parse_data_config(data_cfg)
    test_path = data_cfg['valid']

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size=img_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    model.eval()
    seen = 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAP, mAPj = 0.0, 0.0, 0.0, 0.0
    jdict, tdict, stats, AP, AP_class = [], [], [], [], []

    coco91class = coco80_to_coco91_class()
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Calculating mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        output = model(imgs)
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            correct, detected = [], []
            tcls = torch.Tensor()
            seen += 1

            if pred is None:
                if len(labels):
                    tcls = labels[:, 0].cpu()  # target classes
                    stats.append((correct, torch.Tensor(), torch.Tensor(), tcls))
                continue

            if save_json:  # add to json pred dictionary
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            if len(labels):
                # Extract target boxes as (x1, y1, x2, y2)
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
                tcls = labels[:, 0]  # target classes

                for *pbox, pconf, pcls_conf, pcls in pred:
                    if pcls not in tcls:
                        correct.append(0)
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)
            else:
                # If no labels add number of detections as incorrect
                correct.extend([0] * len(pred))

            # Append Statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls.cpu()))

    # Compute means
    stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
    if len(stats_np):
        AP, AP_class, R, P = ap_per_class(*stats_np)
        mP, mR, mAP = P.mean(), R.mean(), AP.mean()

    # Print P, R, mAP
    print(('%11s%11s' + '%11.3g' * 3) % (seen, len(dataset), mP, mR, mAP))

    # Print mAP per class
    if len(stats_np):
        print('\nmAP Per Class:')
        names = load_classes(data_cfg['names'])
        for c, a in zip(AP_class, AP):
            print('%15s: %-.4f' % (names[c], a))

    # Save JSON
    if save_json and mAP and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return mAP
    return mP, mR, mAP


if __name__ == '__main__':
    args = parse_args()

    with torch.no_grad():
        mAP = test(
            args.config_path,
            args.data_cfg,
            args.weight_path,
            args.batch_size,
            args.img_size,
            args.iou_thres,
            args.conf_th,
            args.nms_thres,
            args.save_json
        )
