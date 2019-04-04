## Installation
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh

## Inference
There are three pre-trained models such as yolo3, yolo3-spp and yolo3-tiny.

    $ python3 inference.py --config-path config/[yolo_version_cfg] --detection-path weights/[yolo_version_weights] --images [path_to_images_or_videos]

## Train
Data augmentation as well as additional training tricks remains to be implemented. PRs are welcomed!
```
    train.py [-h] [--epochs EPOCHS] [--data-path IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--config-path MODEL_CONFIG_PATH]
                [--weight-path WEIGHTS_PATH]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
```


## Test
Evaluates the model on COCO test.

    $ python3 test.py --config-path config/[yolo_version_cfg] --detection-path weights/[yolo_version_weights]


## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
