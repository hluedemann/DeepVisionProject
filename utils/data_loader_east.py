###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   Data loader for the training and evaluation of the EAST model.
###################################################################################################


import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data_processing import parse_annotation, resize_image_and_boxes, get_files_with_extension
from utils.score_functions_east import get_score_values


class ReceiptDataLoaderTrain(Dataset):
    """ Data loader for the EAST model.

    This data loader loads all the data into the RAM. Also all the data needed for the labels in computed beforehand
    in order to allow faster training.
    """

    def __init__(self, dir, transform=None):
        super().__init__()

        self.names = os.listdir(dir)
        self.images_names = get_files_with_extension(dir, self.names, ".jpg")
        self.annotaion_names = get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

        self.new_size = (512, 512)

        self.images = np.zeros((len(self.images_names), *self.new_size, 3))
        self.scores = np.zeros((len(self.images_names), 128, 128))
        self.geos = np.zeros((len(self.images_names), 128, 128, 8))
        self.shortest_edges = np.zeros((len(self.images_names), 128, 128))

        print("Loading train data:")
        for idx in tqdm(range(len(self.images_names))):

            image, score, geo, boxes = self.load_train_data(idx)

            mask = np.argwhere(score != 0.0)
            deltas = geo[mask[:, 0], mask[:, 1], :]

            self.images[idx] = image
            self.scores[idx] = score
            self.geos[idx] = geo
            self.shortest_edges[idx, mask[:, 0], mask[:, 1]] = self.get_shortest_edge(deltas)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.images[idx]))
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1)

        return tensor.type(torch.FloatTensor),\
               torch.tensor(self.scores[idx]).type(torch.FloatTensor),\
               torch.tensor(self.geos[idx]).permute(2, 0, 1).type(torch.FloatTensor),\
               torch.tensor(self.shortest_edges[idx]).type(torch.FloatTensor)

    def get_shortest_edge(self, deltas):
        """ Get the shortest edge of the bounding boxe.
        This required for the normalization of the loss calculation.
        :param deltas: Offsets to corners for every pixel.
        :return: Shortest edge for every pixel.
        """

        n1 = np.sqrt((deltas[:, 0] - deltas[:, 2]) ** 2 + (deltas[:, 1] - deltas[:, 3]) ** 2)
        n2 = np.sqrt((deltas[:, 2] - deltas[:, 4]) ** 2 + (deltas[:, 3] - deltas[:, 5]) ** 2)
        n3 = np.sqrt((deltas[:, 4] - deltas[:, 6]) ** 2 + (deltas[:, 5] - deltas[:, 7]) ** 2)
        n4 = np.sqrt((deltas[:, 6] - deltas[:, 0]) ** 2 + (deltas[:, 7] - deltas[:, 1]) ** 2)

        return np.min([n1, n2, n3, n4])

    def load_train_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotaion_names[idx])

        image_resized, boxes, scale = resize_image_and_boxes(image, boxes, (512, 512))
        score_map, geo_map = get_score_values(image_resized, boxes, scale=0.25)

        return image_resized, score_map, geo_map, boxes


class ReceiptDataLoaderEval(Dataset):
    """ Data loader for the evaluation of the EAST model.

    This data loader loads the data on the fly. This is useful if not all the data is needed. It can also load the test
    data for evaluation.
    """
    def __init__(self, dir, transform=None, type="train"):
        super().__init__()

        self.names = os.listdir(dir)
        self.type = type
        self.images_names = get_files_with_extension(dir, self.names, ".jpg")
        self.annotation_names = get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        if self.type == "train":
            return self.get_train_data(idx)
        else:
            return self.get_test_data(idx)

    def get_train_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotation_names[idx])

        image_resized, boxes, scale = resize_image_and_boxes(image, boxes, (512, 512))

        if self.transform is not None:
            tensor = self.transform(image_resized)
        else:
            tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)

        score_map, geo_map = get_score_values(image_resized, boxes, scale=0.25)

        return tensor, torch.tensor(score_map), torch.tensor(geo_map).permute(2, 0, 1)

    def get_test_data(self, idx):
        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotation_names[idx])
        image_resized = image.resize((512, 512))

        scale = np.zeros(2)
        scale[0] = image_resized.size[0] / image.size[0]
        scale[1] = image_resized.size[1] / image.size[1]

        if self.transform is not None:
            tensor = self.transform(image_resized)
        else:
            tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)
        b = np.array(boxes).reshape((-1, 8))
        return tensor, scale, b

