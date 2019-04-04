# Person Re-ID system using Detection, Tracking and Person ReID architecture (PCB-RPP)
 

## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python 3, Pytorch 0.4+, Tensorflow 1.8+**

1. Install dependencies
```
pip3 install -r requirements.txt
```

2. Run application with video model
```
python app.py --reid-weights [path/to/person_reid_weights] --detection-weight [path/to/detection_weights] --num-classes [number_classes_of_person_reid]
```
