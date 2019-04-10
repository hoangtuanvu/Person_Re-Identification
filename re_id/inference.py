from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import models
from torchvision.transforms import *
import glob
from reid.feature_extraction import extract_cnn_feature
from reid.utils.meters import AverageMeter
import time
from torch.utils.data import Dataset
import warnings
from reid.utils.serialization import load_checkpoint
from reid.utils.data.transforms import RectScale

from PIL import Image


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(path))
    return isfile


class RawDatasetPath(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)
    query_img = args.query
    gallery_path = args.gallery

    transform = Compose([
        RectScale(args.height, args.width),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    query_loader = DataLoader(
        RawDatasetPath(glob.glob(query_img + '/*.jpg'), transform=transform),
        batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=use_gpu, drop_last=False
    )

    gallery_loader = DataLoader(
        RawDatasetPath(glob.glob(gallery_path + '/*.jpg'), transform=transform),
        batch_size=16, shuffle=False, num_workers=args.workers,
        pin_memory=use_gpu, drop_last=False
    )

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=751, cut_at_pooling=False, FCN=True)

    # Load from checkpoint
    print('Loading model ...')
    if args.load_weights and check_isfile(args.load_weights):
        load_checkpoint(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    distmat = inference(model, query_loader, gallery_loader, use_gpu)

    if args.visualize_ranks:
        # Do some visualize ranks
        pass


def inference(model, query_loader, gallery_loader, use_gpu):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf = []
        for batch_idx, (imgs, _) in enumerate(query_loader):
            if use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = extract_cnn_feature(model, imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.extend(list(features))

        gf, g_paths = [], []
        for batch_idx, (imgs, path) in enumerate(gallery_loader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = extract_cnn_feature(model, imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.extend(list(features))
            g_paths.extend(list(path))

    print('=> BatchTime(s): {:.3f}'.format(batch_time.avg))

    x = torch.cat([qf[i].unsqueeze(0) for i in range(len(qf))], 0)
    y = torch.cat([gf[i].unsqueeze(0) for i in range(len(gf))], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # ************************************************************
    # Model
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # ************************************************************
    # Optimizer
    # ************************************************************
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # ************************************************************
    # Training Configs
    # ************************************************************
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # ************************************************************
    # Data augmentation
    # ************************************************************
    parser.add_argument('--random-erase', action='store_true',
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', action='store_true',
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', action='store_true',
                        help='randomly alter the intensities of RGB channels')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--query', type=str, default='', help='load query images')
    parser.add_argument('--gallery', type=str, default='', help='load gallery images')
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
