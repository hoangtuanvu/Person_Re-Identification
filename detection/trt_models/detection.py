from object_detection.protos import pipeline_pb2
from object_detection import exporter
from object_detection.utils import label_map_util
import os
import subprocess
from collections import namedtuple
from google.protobuf import text_format
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from .graph_utils import force_nms_cpu as f_force_nms_cpu
from .graph_utils import replace_relu6 as f_replace_relu6
from .graph_utils import remove_assert as f_remove_assert

# from .graph_utils import force_2ndstage_cpu as f_force_2ndstage_cpu

Model = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt'

MODELS = {
    'ssd_mobilenet_v1_coco':
        Model(
            'ssd_mobilenet_v1_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
            'ssd_mobilenet_v1_coco_2018_01_28',
        ),
    'ssd_mobilenet_v1_0p75_depth_quantized_coco':
        Model(
            'ssd_mobilenet_v1_0p75_depth_quantized_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz',
            'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18'
        ),
    'ssd_mobilenet_v1_ppn_coco':
        Model(
            'ssd_mobilenet_v1_ppn_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
            'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
        ),
    'ssd_mobilenet_v1_fpn_coco':
        Model(
            'ssd_mobilenet_v1_fpn_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
            'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
        ),
    'ssd_mobilenet_v2_coco':
        Model(
            'ssd_mobilenet_v2_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
            'ssd_mobilenet_v2_coco_2018_03_29',
        ),
    'ssdlite_mobilenet_v2_coco':
        Model(
            'ssdlite_mobilenet_v2_coco',
            'http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz',
            'ssdlite_mobilenet_v2_coco_2018_05_09'),
    'ssd_inception_v2_coco':
        Model(
            'ssd_inception_v2_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
            'ssd_inception_v2_coco_2018_01_28',
        ),
    'ssd_resnet_50_fpn_coco':
        Model(
            'ssd_resnet_50_fpn_coco',
            'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
            'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        ),
    'faster_rcnn_resnet50_coco':
        Model(
            'faster_rcnn_resnet50_coco',
            'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
            'faster_rcnn_resnet50_coco_2018_01_28',
        ),
    'faster_rcnn_nas':
        Model(
            'faster_rcnn_nas',
            'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
            'faster_rcnn_nas_coco_2018_01_28',
        ),
    'mask_rcnn_resnet50_atrous_coco':
        Model(
            'mask_rcnn_resnet50_atrous_coco',
            'http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz',
            'mask_rcnn_resnet50_atrous_coco_2018_01_28',
        ),
    'facessd_mobilenet_v2_quantized_open_image_v4':
        Model(
            'facessd_mobilenet_v2_quantized_open_image_v4',
            'http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz',
            'facessd_mobilenet_v2_quantized_320x320_open_image_v4')
}


def get_output_names(model):
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    if model == 'mask_rcnn_resnet50_atrous_coco':
        output_names.append(MASKS_NAME)
    return output_names


def download_detection_model(model, output_dir='.'):
    """Downloads a pre-trained object detection model
    or use your own model
    """
    global MODELS

    model_name = model
    if model_name in MODELS.keys():
        model = MODELS[model_name]
        subprocess.call(['mkdir', '-p', output_dir])
        tar_file = os.path.join(output_dir, os.path.basename(model.url))
        config_path = os.path.join(output_dir, model.extract_dir,
                                   PIPELINE_CONFIG_NAME)
        checkpoint_path = os.path.join(output_dir, model.extract_dir,
                                       CHECKPOINT_PREFIX)
        if not os.path.exists(os.path.join(output_dir, model.extract_dir)):
            subprocess.call(['wget', model.url, '-O', tar_file])
            subprocess.call(['tar', '-xzf', tar_file, '-C', output_dir])
            # hack fix to handle mobilenet_v2 config bug
            subprocess.call(['sed', '-i', '/batch_norm_trainable/d', config_path])
    else:
        # assuming user is querying a self-trained 'egohands' model
        if not os.path.exists(os.path.join(output_dir, model_name + '.config')):
            raise FileNotFoundError
        if not os.path.exists(os.path.join(output_dir, model_name)):
            raise FileNotFoundError
        config_path = os.path.join(output_dir, model_name + '.config')
        checkpoint_path = os.path.join(output_dir, model_name,
                                       CHECKPOINT_PREFIX)
    return config_path, checkpoint_path


def build_detection_graph(config, checkpoint,
                          batch_size=1,
                          score_threshold=None,
                          force_nms_cpu=True,
                          force_frcn2_cpu=True,
                          replace_relu6=True,
                          remove_assert=True,
                          input_shape=None,
                          output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""

    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config):
        with tf.Graph().as_default():
            exporter.export_inference_graph(
                'image_tensor',
                config,
                checkpoint_path,
                output_dir,
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    # if force_frcn2_cpu:
    #     if 'faster_rcnn_' in config_path or 'rfcn_' in config_path:
    #         frozen_graph = f_force_2ndstage_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        if 'ssd_' in config_path or 'ssdlite_' in config_path:
            frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    # TODO: handle mask_rcnn
    input_names = [INPUT_NAME]
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, input_names, output_names


def build_trt_pb(model_name, pb_path, download_dir='data'):
    """Build TRT model from the original TF model, and save the graph
    into a pb file for faster access in the future.
    The code was mostly taken from the following example by NVIDIA.
    https://github.com/NVIDIA-Jetson/tf_trt_models/blob/master/examples/detection/detection.ipynb
    """

    config_path, checkpoint_path = download_detection_model(model_name, download_dir)

    frozen_graph_def, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path
    )

    trt_graph_def = trt.create_inference_graph(
        input_graph_def=frozen_graph_def,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 26,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with open(pb_path, 'wb') as pf:
        pf.write(trt_graph_def.SerializeToString())


def load_trt_pb(pb_path):
    """Load the TRT graph from the pre-build pb file."""
    trt_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as pf:
        trt_graph_def.ParseFromString(pf.read())
    # force CPU device placement for NMS ops
    for node in trt_graph_def.node:
        if 'rfcn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'faster_rcnn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    with tf.Graph().as_default() as trt_graph:
        tf.import_graph_def(trt_graph_def, name='')
    return trt_graph


def write_graph_tensorboard(sess, log_path):
    """Write graph summary to log_path, so TensorBoard could display it."""
    writer = tf.summary.FileWriter(log_path)
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()


def obj_det_graph(obj_model, obj_model_dir, trt_graph_path):
    if os.path.exists(trt_graph_path):
        trt_graph = tf.GraphDef()
        with open(trt_graph_path, 'rb') as f:
            trt_graph.ParseFromString(f.read())
        for node in trt_graph.node:
            if 'NonMaxSuppression' in node.name:
                node.device = '/device:CPU:0'
        input_names = ['image_tensor']

        for node in trt_graph.node:
            if INPUT_NAME == node.name:
                node.attr['shape'].shape.dim[0].size = -1
    else:
        config_path, checkpoint_path = download_detection_model(obj_model, obj_model_dir)
        frozen_graph, input_names, output_names = build_detection_graph(
            config=config_path, checkpoint=checkpoint_path, score_threshold=0.3)

        print('Input_names: {}'.format(input_names))
        print("Making a TRT graph for the object detection model")
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50)

        with open(trt_graph_path, 'wb') as f:
            f.write(trt_graph.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.333
    tf_sess = tf.Session(config=tf_config)

    tf.import_graph_def(trt_graph, name='')
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    return tf_sess, tf_scores, tf_boxes, tf_classes, tf_input


def read_label_map(path_to_labels):
    """Read from the label map file and return a class dictionary which
    maps class id (int) to the corresponding display name (string).
    Reference:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    """

    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels)
    cls_dict = {int(x['id']): x['name'] for x in category_index.values()}
    num_classes = max(c for c in cls_dict.keys()) + 1
    # add missing classes as, say,'CLS12' if any
    return {i: cls_dict.get(i, 'CLS{}'.format(i)) for i in range(num_classes)}
