###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   This script performs text detection with a trained EAST model.
###################################################################################################


import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from models.east import EAST, load_east_model, device
from utils.score_functions_east import get_bounding_boxes_from_output
from utils.data_processing import scale_bounding_box
from utils.plot_utils import add_bounding_box, plot_image


def text_detection(model, image_path):
    """ Determine the text bounding boxes of an image using the EAST model.

    :param model: Trained EAST model.
    :param image_path: Path to the image.
    :return: Text bounding boxes.
    """

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    image_resized = image.resize((512, 512))
    image_output = image.resize((128, 128))

    scale = np.zeros(2)
    scale[0] = image.size[0] / image_output.size[0]
    scale[1] = image.size[1] / image_output.size[1]

    image_tensor = transform(image_resized)
    image_tensor = image_tensor.unsqueeze(0)

    model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        score, geo = model(image_tensor)
        score = score.cpu()
        geo = geo.cpu()

    restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(),
                                                     geo[0].permute(1, 2, 0).detach().numpy())
    restored_bboxes_scaled = scale_bounding_box(restored_bboxes[:, :8], scale)
    restored_bboxes_scaled = restored_bboxes_scaled.reshape(-1, 8)

    return restored_bboxes_scaled[:, :8]


if __name__ == "__main__":
    east = load_east_model(weight="trained_models/model_east.ckpt")
    # image_path = "data/test_data/X51006438346.jpg"
    image_path = "testImages/white.jpg"
    bboxes = text_detection(east, image_path)

    # Plot the detected bounding boxes
    image = Image.open(image_path)
    add_bounding_box(image, bboxes, "red")
    plot_image(image, "text_detection_text")
