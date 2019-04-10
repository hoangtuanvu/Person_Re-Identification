from __future__ import absolute_import
import torch


def extract_cnn_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        tmp = model(inputs)
        outputs = tmp[0]
        outputs = outputs.data.cpu()
        return outputs
