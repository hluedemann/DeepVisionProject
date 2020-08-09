###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   This script implements functions to evaluate the performance of the EAST model on the test
#   data set.
###################################################################################################


import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cv2 import fillPoly

from models.east import EAST, load_east_model
from utils.data_loader_east import ReceiptDataLoaderEval
from utils.data_processing import scale_bounding_box
from utils.score_functions_east import get_bounding_boxes_from_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)


def calculate_IoU(gt_bboxs, pred_bboxs, image_size):
    """ Calculate the Intersection over Union score.

    :param gt_bboxs: Ground truth bounding boxes.
    :param pred_bboxs: Predicted bounding boxes.
    :param image_size: Size of the image.
    :return: IoU score.
    """

    gt_mask = np.zeros((int(image_size[1]), int(image_size[0])))
    pred_mask = np.zeros((int(image_size[1]), int(image_size[0])))

    gt_bboxs = gt_bboxs.reshape(-1, 4, 2).astype(np.int32)
    pred_bboxs = pred_bboxs.reshape(-1, 4, 2).astype(np.int32)

    for box in gt_bboxs:
        fillPoly(gt_mask, [box], 1)
    for box in pred_bboxs:
        fillPoly(pred_mask, [box], 1)

    intersection = np.argwhere(gt_mask + pred_mask == 2).shape[0]
    union = np.sum(gt_mask + pred_mask - gt_mask * pred_mask)

    return intersection / union


def evaluate_east(model_path, data_loader):
    """ Evaluate the EAST model on the test data set.

    :param model_path: Path to the trained model.
    :param data_loader: Data loader to load the test data.
    :return: IoU scores for every test data image.
    """

    east = load_east_model(weight=model_path)
    east.to(device)

    data_iter = iter(data_loader)

    i = 0
    IoUs = np.zeros(len(data))

    with torch.no_grad():
        for img, scale, boxes in tqdm(data_iter):

            boxes = boxes.numpy()[0]
            scale = scale.numpy()[0]

            img = img.to(device)
            score, geo = east(img)

            score = score.cpu()
            geo = geo.cpu()

            restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(),
                                                             geo[0].permute(1, 2, 0).detach().numpy())
            restored_bboxes_scaled = scale_bounding_box(restored_bboxes[:, :8], 4.0 / scale).reshape(-1, 8)

            image_size_resized = [img.shape[2], img.shape[3]]
            IoU = calculate_IoU(boxes, restored_bboxes_scaled, image_size_resized / scale)
            IoUs[i] = IoU

            i += 1
    return IoUs


if __name__ == "__main__":

    test_data = "data/test_data"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ReceiptDataLoaderEval(test_data, transform=transform, type="test")
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    model_path = "check_points_final/model_east_395.ckpt"

    print("Evaluating EAST model")
    IoUs = evaluate_east(model_path, data_loader)

    print("\nMin IoU score: ", np.min(IoUs))
    print("Max IoU score: ", np.max(IoUs))
    print("Average IoU score: ", np.mean(IoUs))
