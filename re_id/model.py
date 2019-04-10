from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import warnings
import numpy as np
import sys
import torch
import os
from torch import nn
from torch.backends import cudnn

from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import save_checkpoint
from reid.data_manager import ImageDataManager
from reid.utils.data.reidtools import visualize_ranked_results
from reid.utils.data.torchtools import load_checkpoint


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'combineall': parsed_args.combineall,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
        'cuhk03_labeled': parsed_args.cuhk03_labeled,
        'cuhk03_classic_split': parsed_args.cuhk03_classic_split,
        'market1501_500k': parsed_args.market1501_500k,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else (256, 128)

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    train_loader, testloader_dict = dm.return_dataloaders()

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=dm.num_train_pids, cut_at_pooling=False, FCN=True)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = nn.DataParallel(model).cuda() if use_gpu else model

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Evaluate give datasets:")
        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            query_loader = testloader_dict[name]['query']
            gallery_loader = testloader_dict[name]['gallery']
            distmat = evaluator.evaluate(query_loader, gallery_loader)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    topk=20
                )
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, use_gpu)
        is_best = True
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    print("Evaluate give datasets:")
    for name in args.target_names:
        print('Evaluating {} ...'.format(name))
        query_loader = testloader_dict[name]['query']
        gallery_loader = testloader_dict[name]['gallery']
        distmat = evaluator.evaluate(query_loader, gallery_loader)

        if args.visualize_ranks:
            visualize_ranked_results(
                distmat, dm.return_testdataset_by_name(name),
                save_dir=osp.join(args.save_dir, 'ranked_results', name),
                topk=20
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='data',
                        help='root path to data directory')
    parser.add_argument('-s', '--source-names', type=str, nargs='+',
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--target-names', type=str, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split-id', type=int, default=0,
                        help='split index (note: 0-based)')
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--train-sampler', type=str, default='RandomSampler',
                        help='sampler for trainloader')
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--combineall', action='store_true',
                        help='combine all data in a dataset (train+query+gallery) for training')
    # ************************************************************
    # Dataset-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help='use labeled images, if false, use detected images')
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help='use classic split by Li et al. CVPR\'14')
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help='use cuhk03\'s metric for evaluation')
    parser.add_argument('--market1501-500k', action='store_true',
                        help='add 500k distractors to the gallery set for market1501')
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--train-batch-size', default=64, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help='test batch size')
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
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--visualize-ranks', action='store_true',
                        help='visualize ranked results, only available in evaluation mode')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    main(parser.parse_args())
