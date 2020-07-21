import torch
import torch.nn as nn

import numpy as np
from sympy.utilities.iterables import multiset_permutations
from torchvision import transforms

from score_functions_east import *
from data_processing import *
from training_east import *

from tqdm import tqdm

def dist_tensor(p1, p2):
    """ Return the euclidean distance between two points.

    :param p1: First point.
    :param p2: Second point.
    :return: Euclidean distance.
    """
    return torch.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def dist_tensor(p1, p2):
    """ Return the euclidean distance between two boints.

    :param p1: First point.
    :param p2: Second point.
    :return: Euclidean distance.
    """
    return torch.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def quad_geo_loss(self, score, geo, pred_geo, edge):

        def smoothed_l1(x):
            return torch.where(torch.abs(x) < 1.0, 0.5 * x ** 2, torch.abs(x) - 0.5)

        idx = score.nonzero()
        deltas = geo[:, idx[:, 0], idx[:, 1]]
        pred_deltas = pred_geo[:, idx[:, 0], idx[:, 1]]

        pix_loss = torch.sum(smoothed_l1(deltas - pred_deltas), dim=0) / (8.0 * edge[idx[:, 0], idx[:, 1]])

        return torch.mean(pix_loss)



    def dice_loss(self, score, score_pred):
        return 1.0 - (2.0 * torch.sum(score * score_pred) / (torch.sum(score) + torch.sum(score_pred)))

    def forward(self, score, pred_score, geo, pred_geo, edge):

        if len(score.shape) == 2:
            geo_loss = self.quad_geo_loss(score, geo, pred_geo)
            score_loss = self.dice_loss(score, pred_score)

        else:
            geo_loss = 0
            score_loss = 0

            for batch in range(score.shape[0]):
                geo_loss += self.quad_geo_loss(score[batch], geo[batch], pred_geo[batch], edge[batch])
                score_loss += self.dice_loss(score[batch], pred_score[batch])

        score_loss /= score.shape[0]
        geo_loss /= score.shape[0]

        return score_loss + geo_loss


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = ReceiptDataLoaderRam("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=8, shuffle=False)

    total_loss = 0

    loss = CustomLoss()

    for step, (images, score_map, geo_map, edge) in enumerate(tqdm(data_loader, desc='Batch')):

        score_map_pred = score_map + np.random.normal(0.0, 2.0, size=score_map.shape)
        geo_map_pred = geo_map + np.random.normal(0.0, 2.0, size=geo_map.shape)

        total_loss += loss.forward(score_map, score_map_pred, geo_map, geo_map_pred, edge)

    print("loss", total_loss)
