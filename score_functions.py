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

    boxes = np.around(np.array(boxes)).astype(np.int32)
    shrinked_boxes = np.around(np.array(get_shrinked_bboxes(boxes))).astype(np.int32)

    score_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0])), dtype=np.float64)
    geo_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0]), 8), dtype=np.float64)

    for i in range(boxes.shape[0]):
        shrinked_box = np.around(scale * shrinked_boxes[i]).astype(np.int32)
        box = np.around(scale * boxes[i]).astype(np.int32)
        cv2.fillPoly(score_map, [shrinked_box], 1)
        bounding_rectangle = get_bounding_rectangle(box)
        for x in np.arange(*bounding_rectangle[0]):
            for y in np.arange(*bounding_rectangle[1]):
                if score_map[x, y] == 1:
                    geo_map[x, y] = get_distances((x, y), box)

    return score_map, geo_map


def restore_bounding_box(point, distances):
    p1 = (point[0] - distances[0], point[1] - distances[1])
    p2 = (point[0] - distances[2], point[1] - distances[3])
    p3 = (point[0] - distances[4], point[1] - distances[5])
    p4 = (point[0] - distances[6], point[1] - distances[7])

    return np.array([p1, p2, p3, p4]).reshape(-1)


def get_bounding_boxes_from_output(score_map, geo_map):
    index_text = np.argwhere(score_map > 0.9)
    index_text = index_text[np.argsort(index_text[:, 0])]

    restored_bounding_boxes = np.zeros((index_text.shape[0], 8))
    for i in range(index_text.shape[0]):
        indices = index_text[i]
        restored_bounding_boxes[i] = restore_bounding_box(indices, geo_map[indices[0], indices[1], :])

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

    example, boxes = resize_image_and_boxes(example, boxes, (512, 512))

    score_map, geo_map = get_score_values(example, boxes, scale=0.25)
    print("score_map ", score_map.shape)
    print("geo_map ", geo_map.shape)


    restored_bboxes = get_bounding_boxes_from_output(score_map, geo_map)

    new_x = np.around(int(0.25 * example.size[0]))
    new_y = np.around(int(0.25 * example.size[1]))

    example = example.resize((new_x, new_y))

    add_bounding_box(example, restored_bboxes[0:44], "blue")
    plot_image(example, "restored_boxes")
    print(restored_bboxes.shape)