from __future__ import absolute_import
import torch
from ..utils import to_torch


def extract_cnn_feature(model, inputs):
    model.eval()
    inputs = to_torch(inputs)
    with torch.no_grad():
        tmp = model(inputs)
        outputs = tmp[0]
        outputs = outputs.data.cpu()
        return outputs
