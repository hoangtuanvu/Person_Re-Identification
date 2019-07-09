import numpy as np
import glob
import os
from processor.post_process import gen_total_objects
from processor.utilities import read_counting_gt
from processor.utilities import rms
from processor.post_process import gen_report


def offline_extraction(file, cls_thres):
    counter = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            values = line.strip().split(',')

            hits = -1
            for val in values:
                if 'hits' in val:
                    hits = int(val.split('=')[1])

            if hits == -1:
                raise ValueError('Negative hits')

            cls_id = int(values[0])
            if cls_id not in counter:
                counter[cls_id] = [hits]
            else:
                counter[cls_id].append(hits)

    for cls in counter:
        np_hits = np.asarray(counter[cls])
        if cls in cls_thres:
            counts = len(np.where(np_hits >= cls_thres[cls])[0])
        else:
            counts = len(np_hits)
        counter[cls] = counts

    return counter


def offline_counter(pred_path, gt_path, classes, od_model, cls_thres):
    object_cnt_all = []
    gt = read_counting_gt(gt_path)
    print(gt)

    count_id = 0
    error_score = 0
    for path in sorted(glob.glob('{}/*.txt'.format(pred_path))):
        counter = offline_extraction(path, cls_thres)
        print(counter)
        error = rms(gt[count_id]["objects"],
                    gen_total_objects(classes, counter, od_model))

        object_cnt = {"name": os.path.basename(path).split('.')[0],
                      "objects": gen_total_objects(classes, counter, od_model),
                      "rms": error}
        object_cnt_all.append(object_cnt)
        error_score += error
        count_id += 1

    # Generate Report
    gen_report(gt, object_cnt_all)

    return error_score / count_id


# print(offline_counter('/home/brian/Downloads/tracks_070309', '../t1_gt.json', [0, 1, 2, 3, 4, 5], 'centernet',
#                       {0:6, 1: 5, 2: 4, 3: 1, 4: 5, 5: 4}))

print(offline_counter('/home/brian/Downloads/tracks_070818', '../t1_gt.json', [0, 10, 2, 5, 7, 1, 3], 'yolo',
                      {0:5, 1: 4, 3: 4, 2: 6, 5: 6, 7: 6, 10: 4}))

# print(offline_counter('/home/brian/Downloads/tracks_070220', '../t1_gt.json', [0, 1, 2, 3, 4, 5], 'centernet',
#                       {0:5, 1: 3, 2: 4, 3: 1, 4: 5, 5: 4}))

# print(offline_counter('/home/brian/Downloads/tracks_070215', '../t1_gt.json', [0, 1, 2, 3, 4, 5], 'centernet',
#                       {0: 18, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4}))
