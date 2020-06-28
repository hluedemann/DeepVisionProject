import torch
import torch.nn as nn



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, score, pred_score, geo, pred_geo):

        a = torch.sum(score * pred_score)
        b = torch.sum(score) + torch.sum(pred_score) + 1e-5

        classify_loss = 1.0 - (2.0 * a / b)

        return classify_loss