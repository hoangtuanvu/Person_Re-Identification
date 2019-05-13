import os
import sys
import torch
import numpy as np
import torch.nn as nn
from .layers import Upsample
from .layers import EmptyLayer
from .layers import YOLOLayer

sys.path.append('..')
from detection.utils.parse_config import parse_model_config


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416, device=torch.device('cpu'), onnx_export=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.onnx_export = onnx_export
        self.hyperparams, self.module_list = self.create_modules(self.module_defs, device)

    def forward(self, x, var=None):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if self.training:
            return output
        elif self.onnx_export:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            return output[5:85].t(), output[:4].t()  # ONNX scores, boxes
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

        # return output if self.training else torch.cat(output, 1)

    def create_modules(self, module_defs, device):
        """
        Constructs module list of layer blocks from module configuration in module_defs
        """
        hyperparams = module_defs.pop(0)
        output_filters = [int(hyperparams['channels'])]
        module_list = nn.ModuleList()
        yolo_layer_count = 0
        for i, module_def in enumerate(module_defs):
            modules = nn.Sequential()

            if module_def['type'] == 'convolutional':
                bn = int(module_def['batch_normalize'])
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

            elif module_def['type'] == 'maxpool':
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                modules.add_module('maxpool_%d' % i, maxpool)

            elif module_def['type'] == 'upsample':
                # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
                upsample = Upsample(scale_factor=int(module_def['stride']))
                modules.add_module('upsample_%d' % i, upsample)

            elif module_def['type'] == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())

            elif module_def['type'] == 'shortcut':
                filters = output_filters[int(module_def['from'])]
                modules.add_module('shortcut_%d' % i, EmptyLayer())

            elif module_def['type'] == 'yolo':
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                nc = int(module_def['classes'])  # number of classes
                img_size = hyperparams['height']

                # Define detection layer
                yolo_layer = YOLOLayer(anchors, nc, img_size, yolo_layer_count, cfg=hyperparams['cfg'], device=device,
                                       onnx_export=self.onnx_export)
                modules.add_module('yolo_%d' % i, yolo_layer)
                yolo_layer_count += 1

            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return hyperparams, module_list

    def load_darknet_weights(self, weights, cutoff=-1):
        # Parses and loads the weights stored in 'weights'
        # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
        weights_file = weights.split(os.sep)[-1]

        # Try to download weights if not available locally
        if not os.path.isfile(weights):
            try:
                os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
            except IOError:
                print(weights + ' not found')

        # Establish cutoffs
        if weights_file == 'darknet53.conv.74':
            cutoff = 75
        elif weights_file == 'yolov3-tiny.conv.15':
            cutoff = 15

        # Open the weights file
        fp = open(weights, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header
        self.seen = header[3]  # number of images seen during training
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

        return cutoff

    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen  # number of images seen during training
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    fused = self.fuse_conv_and_bn(conv, b)
                    a = nn.Sequential(fused, *list(a.children())[i + 1:])
                    break
            fused_list.append(a)
        self.module_list = fused_list

    @staticmethod
    def fuse_conv_and_bn(conv, bn):
        # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        with torch.no_grad():
            # init
            fusedconv = torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=True
            )

            # prepare filters
            w_conv = conv.weight.clone().view(conv.out_channels, -1)
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
            fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

            # prepare spatial bias
            if conv.bias is not None:
                b_conv = conv.bias
            else:
                b_conv = torch.zeros(conv.weight.size(0))
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            fusedconv.bias.copy_(b_conv + b_bn)

            return fusedconv
