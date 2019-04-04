import sys
sys.path.append('..')
from detection.utils.commons import *
from .utils import create_grids
import torch.nn.functional as F


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, device):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        create_grids(self, 32, 1, device)

    def forward(self, p, img_size, var=None):
        bs, nG = p.shape[0], p.shape[-1]
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            p[..., 0:2] = torch.sigmoid(p[..., 0:2]) + self.grid_xy  # xy
            p[..., 2:4] = torch.exp(p[..., 2:4]) * self.anchor_wh  # wh yolo method
            # p[..., 2:4] = ((torch.sigmoid(p[..., 2:4]) * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., 5:] = torch.sigmoid(p[..., 5:])  # p_class
            # p[..., 5:] = F.softmax(p[..., 5:], dim=4)  # p_class
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
