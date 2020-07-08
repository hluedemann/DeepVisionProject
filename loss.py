import torch
import torch.nn as nn

import numpy as np
from sympy.utilities.iterables import multiset_permutations
from torchvision import transforms

from score_functions import *
from data_processing import *
from training import *

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

        """
        def shortest_edge(deltas):
            n1 = dist_tensor(*deltas[[0, 1, 2, 3]].reshape(2, 2))
            n2 = dist_tensor(*deltas[[2, 3, 4, 5]].reshape(2, 2))
            n3 = dist_tensor(*deltas[[4, 5, 6, 7]].reshape(2, 2))
            n4 = dist_tensor(*deltas[[6, 7, 0, 1]].reshape(2, 2))

            if np.min([n1, n2, n3, n4]) == 0:
                print("####################################################")
                print(deltas)
                print("####################################################")
            return np.min([n1, n2, n3, n4])
        """

        ## shape is (8, 128, 128)
        c, x, y = geo.shape
        geo_loss = 0
        pixel_count = 1

        idx = score.nonzero()
        deltas = geo[:, idx[:, 0], idx[:, 1]]
        pred_deltas = pred_geo[:, idx[:, 0], idx[:, 1]]
        #print("idx: ",idx.shape)
        #print("ds: ",ds.shape)
        #print("pds: ",pds.shape)

        pix_loss = torch.sum(smoothed_l1(deltas - pred_deltas), dim=0) / (8.0 * edge[idx[:, 0], idx[:, 1]])

        return torch.mean(pix_loss)

        """
        for i in range(x):
            for j in range(y):

                if score[i, j] == 0.0:
                    continue
                deltas = geo[:, i, j]
                # print(f"deltas: {deltas}")
                pred_deltas = pred_geo[:, i, j]
                # print(f"pred_deltas: {pred_deltas}")

                # iterate over all possible permutations of deltas

                # pixel_loss = np.zeros(4*3*2*1)
                # for i, p in enumerate(multiset_permutations(np.array([0,2,4,6]))):
                #    permuted_deltas = deltas[[p[0], p[0]+1, p[1], p[1]+1, p[2], p[2]+1, p[3], p[3]+1]]
                #    pixel_loss[i] = torch.sum(smoothed_l1(pred_deltas - permuted_deltas))

                pixel_loss = torch.sum(smoothed_l1(pred_deltas - deltas))
                # print(f"pixel_loss: {pixel_loss}")
                #if edge[i, j] != 0.0:
                geo_loss += pixel_loss / (8.0 * edge[i, j])  # np.min(pixel_loss) / (8.0 * edge)
                pixel_count += 1
                #else:
                #    print(f"no egde")

        # return geo_loss / (x * y)
        geo_loss_norm = geo_loss / pixel_count
        #print(f"geo_loss_norm: {geo_loss_norm}")
        return geo_loss_norm
        """

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

        # score_loss /= (128.0 * 128.0)
        # print("score loss: ", score_loss)
        print("geo loss: ", geo_loss)
        print("score loss: ", score_loss)
        print("loss: ", score_loss + geo_loss)

        return score_loss + geo_loss


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = ReceiptDataLoaderRam("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=8, shuffle=False)

    total_loss = 0

    loss = CustomLoss()

    for step, (images, score_map, geo_map, edge) in enumerate(tqdm(data_loader, desc='Batch')):
        #images, score, geo = images.to(device), score_map.to(device), geo_map.to(device)

        #print("score map: ", score_map.shape)
        #print("score map: ", geo_map.shape)

        score_map_pred = score_map + np.random.normal(0.0, 2.0, size=score_map.shape)
        geo_map_pred = geo_map + np.random.normal(0.0, 2.0, size=geo_map.shape)

        total_loss += loss.forward(score_map, score_map_pred, geo_map, geo_map_pred, edge)

    print("loss", total_loss)
