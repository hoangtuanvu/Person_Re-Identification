TensorFlow/TensorRT Models for Person Detection
====================================

<p align="center">
<img src="data/landing_graphic.jpg" alt="landing graphic" height="300px"/>
</p>

This repository contains scripts and documentation to use TensorFlow image classification and object detection models.  The models are sourced from the [TensorFlow models repository](https://github.com/tensorflow/models)
and optimized using TensorRT.

* [Setup](#setup)
* [Object Detection](#od)
  * [Download pretrained model](#od_download)
  * [Build TensorRT compatible graph](#od_build)
  * [Optimize with TensorRT](#od_trt)
  * [Jupyter Notebook Sample](#od_notebook)
  * [Train for custom task](#od_train)

<a name="setup"></a>
Setup
-----

1. Install TensorRT by the following instruction
   ```
   https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html
   ```
2. Install dependencies

   ```
   sudo apt-get install python3-pip python3-matplotlib python3-pil
   ```
   
3. Install TensorFlow 1.7+ (with TensorRT support).

    ```
    pip install tensorflow-gpu
    ```
    
    or if you're using Python 3.
    
    ```
    pip3 install tensorflow-gpu
    ```


4. Run the installation script

    ```
    ./install.sh
    ```
    
    or if you want to specify python interpreter
    
    ```
    ./install.sh python3
    ```

<a name="od"></a>
Object Detection 
----------------

<a name="od_download"></a>
### Download pretrained model

As a convenience, we provide a script to download pretrained model weights and config files sourced from the
TensorFlow models repository.  

```python
from detection.trt_models.detection import download_detection_model

config_path, checkpoint_path = download_detection_model('ssd_inception_v2_coco')
```
To manually download the pretrained models, follow the links [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

> **Important:** Some of the object detection configuration files have a very low non-maximum suppression score threshold (ie. 1e-8).
> This can cause unnecessarily large CPU post-processing load.  Depending on your application, it may be advisable to raise 
> this value to something larger (like 0.3) for improved performance.  We do this for the above benchmark timings.  This can be done by modifying the configuration
> file directly before calling build_detection_graph.  The parameter can be found for example in this [line](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config#L130).

<a name="od_build"></a>
### Build TensorRT compatible graph

```python
from detection.trt_models.detection import build_detection_graph

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path
)
```

<a name="od_trt"></a>
### Optimize with TensorRT

```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```

<a name="od_notebook"></a>
### Jupyter Notebook Sample

For a comprehensive example of performing the above steps and executing on a real
image, see the [jupyter notebook sample](examples/detection/detection.ipynb).

<a name="od_train"></a>
### Train for custom task

Follow the documentation from the [TensorFlow models repository](https://github.com/tensorflow/models/tree/master/research/object_detection).
Once you have obtained a checkpoint, proceed with building the graph and optimizing
with TensorRT as shown above.  Please note that all models are not tested so 
you should use an object detection
config file during training that resembles one of the ssd_mobilenet_v1_coco or
ssd_inception_v2_coco models.  Some config parameters may be modified, such as the number of
classes, image size, non-max supression parameters, but the performance may vary.
