import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import copy

import cv2
import lanms

from data_processing import *




def get_score_values(image, boxes, scale=0.25):
    """ Return the ground truth score map and geometry map of an image and its bounding boxes.

    :param image: Image.
    :param boxes: Boxes.
    :param scale: Scale factor to scale the boxes with. Required because the network outputs results scaled by
                  factor of 0.25.
    :return: Scaled score and geometry map.
    """

    def get_bounding_rectangle(vertices):
        vertices_T = vertices.T
        x_min = np.min(vertices_T[1])
        x_max = np.max(vertices_T[1])
        y_min = np.min(vertices_T[0])
        y_max = np.max(vertices_T[0])
        return [[x_min, x_max], [y_min, y_max]]

    def get_distances(point, vertices):
        distances = np.zeros(8)
        for i in range(4):
            distances[i * 2] = point[0] - vertices[i, 0]
            distances[i * 2 + 1] = point[1] - vertices[i, 1]

        return distances

    #boxes = np.around(np.array(boxes)).astype(np.int32)
    boxes = np.array(boxes)
    #shrinked_boxes = np.around(np.array(get_shrinked_bboxes(boxes))).astype(np.int32)
    shrinked_boxes = np.array(get_shrinked_bboxes(boxes))

    score_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0])), dtype=np.float64)
    geo_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0]), 8), dtype=np.float64)


    for i in range(boxes.shape[0]):

        shrinked_box = np.around(scale * shrinked_boxes[i]).astype(np.int32)
        # box = np.around(scale * boxes[i]).astype(np.int32)
        #print("boxi: ", boxes[i])
        box = scale * boxes[i]

        mask = np.zeros((int(scale * image.size[1]), int(scale * image.size[0])), dtype=np.float64)
        cv2.fillPoly(mask, [shrinked_box], 1)
        score_map += mask

        idx = np.argwhere(mask == 1)

        for y, x in idx:
            #print("Box", box)
            #print(f"x/y: {x}, {y}")
            #print("dist: ", get_distances((x, y), box))
            geo_map[y, x] = get_distances((x, y), box)

        """    
        bounding_rectangle = get_bounding_rectangle(box)
        xmin, xmax = bounding_rectangle[0]
        ymin, ymax = bounding_rectangle[1]
        for x in np.arange(xmin, xmax):
            for y in np.arange(ymin, ymax):
                if score_map[x, y] == 1:
                    geo_map[x, y] = get_distances((x, y), box)
        """
    """
    print("######## Score count ##########")
    print("score: ", np.argwhere(score_map == 1.0).shape)
    print("geo: ", np.argwhere(np.sum(np.abs(geo_map), axis=2) != 0.0).shape)
    print("###############################")
    """

    return score_map, geo_map


def restore_bounding_box(point, distances):
    p1 = (point[0] - distances[0], point[1] - distances[1])
    p2 = (point[0] - distances[2], point[1] - distances[3])
    p3 = (point[0] - distances[4], point[1] - distances[5])
    p4 = (point[0] - distances[6], point[1] - distances[7])

    return np.array([p1, p2, p3, p4]).reshape(-1)


def get_bounding_boxes_from_output(score_map, geo_map):
    """ Recreate the boxes from score map and geometry map.

    :param score_map: Score map.
    :param geo_map: Geometry map.
    :return: Restored boxes
    """
    index_text = np.argwhere(score_map > 0.9)
    #index_text = index_text[np.argsort(index_text[:, 0])]


    restored_bounding_boxes = np.zeros((index_text.shape[0], 8))
    for i in range(index_text.shape[0]):
        indices = index_text[i]   # [y, x]
        restored_bounding_boxes[i] = restore_bounding_box([indices[1], indices[0]], geo_map[indices[0], indices[1], :])

    boxes = np.zeros((restored_bounding_boxes.shape[0], 9))
    boxes[:, :8] = restored_bounding_boxes
    boxes[:, 8] = score_map[index_text[:, 0], index_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.2)

    return boxes



if __name__ == "__main__":

    example_image = "data/train_data/X00016469612.jpg"
    example_annotation = "data/train_data/X00016469612.txt"

    example = Image.open(example_image)

    boxes, texts = parse_annotation(example_annotation)

    example, boxes, scale = resize_image_and_boxes(example, boxes, (512, 512))
    score_map, geo_map = get_score_values(example, boxes, scale=0.25)

    add_bounding_box(example, convert_vert_list_to_tuple(boxes), "red")
    example.show()

    restored_bboxes = get_bounding_boxes_from_output(score_map, geo_map)


    ## Resize image and plot bounding boxes
    new_x = np.around(int(0.25 * example.size[0]))
    new_y = np.around(int(0.25 * example.size[1]))
    example = example.resize((new_x, new_y))
    add_bounding_box(example, restored_bboxes[:, :8], "red")
    plot_image(example, "restored_boxes")
    example.show()

    ## Resize bouning boxes and plot them
    restored_boxes_scales = scale_bounding_box(restored_bboxes[:, :8], 4/scale)
    restored_boxes_scales = restored_boxes_scales.reshape(-1, 8)
    example = Image.open(example_image)
    add_bounding_box(example, restored_boxes_scales, "red")
    plot_image(example, "resized_boxes")
    example.show()
