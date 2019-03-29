from __future__ import print_function, absolute_import
import time
import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []
    pIDs = []
    camIDs = []

    end = time.time()
    for i, (imgs, pids, camids, fnames) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for camid, output, pid in zip(camids, outputs, pids):
            features.append(output)
            pIDs.append(pid)
            camIDs.append(camid)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, pIDs, camIDs


def pairwise_distance(query_features, gallery_features):
    x = torch.cat([query_features[i].unsqueeze(0) for i in range(len(query_features))], 0)
    y = torch.cat([gallery_features[i].unsqueeze(0) for i in range(len(gallery_features))], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    assert (query_ids is not None and gallery_ids is not None
            and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader):
        print('extracting query features\n')
        query_features, q_ids, q_camids = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, g_ids, g_camids = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features)
        evaluate_all(distmat, query_ids=q_ids, gallery_ids=g_ids, query_cams=q_camids, gallery_cams=g_camids)

        return distmat
