import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
from torchvision.models.vgg import VGG, make_layers, model_urls
from torchvision import transforms

from data_processing import *
from score_functions_east import *
from data_loader_east import *

import os

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Extractor(nn.Module):
    def __init__(self, batch_norm=True, pretrained=True):
        super(Extractor, self).__init__()
        vgg = VGG(make_layers(cfg['D'], batch_norm))

        if pretrained:
            if batch_norm:
                vgg.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
            else:
                vgg.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

        self.features = vgg.features


    def forward(self, x):
        f = []
        # we only need the output of the CNN
        for feature in self.features:
            x = feature(x)
            # check if we are after a pooling layer
            if isinstance(feature, nn.MaxPool2d):
                f.append(x)

        # return values after pooling 2-5
        return f[1:]


class Merge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y


class Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_score = nn.Conv2d(32, 1, 1)
        self.sigmoid_score = nn.Sigmoid()
        self.conv_quad = nn.Conv2d(32, 8, 1)
        self.sigmoid_quad = nn.Sigmoid()

    def forward(self, x):
        score = self.conv_score(x)
        score = self.sigmoid_score(score)

        quad = self.conv_quad(x)
        #quad = self.sigmoid_quad(quad)

        return score, quad


class EAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = Extractor()
        self.merge = Merge()
        self.output = Output()

    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))


if __name__ == "__main__":

    example_image = "data/train_data/X00016469612.jpg"
    example_annotation = "data/train_data/X00016469612.txt"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    data = ReceiptDataLoader("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=16, shuffle=True)


    east = EAST()
    # east.to(device)

    with torch.no_grad():
        for step, data in enumerate(data_loader):

            print(step)
            images, score_map, score_geo = data

            print(images.shape)
            print(score_map.shape)
            print(score_geo.shape)

            s, g = east.forward(images)









