###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   Utility functions for the training of the EAST model.
###################################################################################################


import numpy as np
import cv2
import lanms

from utils.data_processing import get_shrinked_bboxes, parse_annotation, resize_image_and_boxes


def get_score_values(image, boxes, scale=0.25):
    """ Return the ground truth score map and geometry map of an image and its bounding boxes.

    :param image: Image.
    :param boxes: Boxes.
    :param scale: Scale factor to scale the boxes with. Required because the network outputs results scaled by
                  factor of 0.25.
    :return: Scaled score and geometry map.
    """

    def get_distances(point, vertices):
        distances = np.zeros(8)
        for i in range(4):
            distances[i * 2] = point[0] - vertices[i, 0]
            distances[i * 2 + 1] = point[1] - vertices[i, 1]

        return distances

    boxes = np.array(boxes)
    shrinked_boxes = np.array(get_shrinked_bboxes(boxes))

    score_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0])), dtype=np.float64)
    geo_map = np.zeros((int(scale * image.size[1]), int(scale * image.size[0]), 8), dtype=np.float64)

    for i in range(boxes.shape[0]):

        shrinked_box = np.around(scale * shrinked_boxes[i]).astype(np.int32)
        box = scale * boxes[i]

        mask = np.zeros((int(scale * image.size[1]), int(scale * image.size[0])), dtype=np.float64)
        cv2.fillPoly(mask, [shrinked_box], 1)
        score_map += mask

        idx = np.argwhere(mask == 1)

        for y, x in idx:
            geo_map[y, x] = get_distances((x, y), box)

    return score_map, geo_map


def restore_bounding_box(point, distances):
    """ Recreate bounding boxes from offset values.

    :param point: Point of the pixel.
    :param distances: Offsets form point to four corners of bounding box.
    :return: Return bounding box in the form [(x1, y1), (x2, y2),...]
    """
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

    restored_bounding_boxes = np.zeros((index_text.shape[0], 8))
    for i in range(index_text.shape[0]):
        indices = index_text[i]   # [y, x]
        restored_bounding_boxes[i] = restore_bounding_box([indices[1], indices[0]], geo_map[indices[0], indices[1], :])

    boxes = np.zeros((restored_bounding_boxes.shape[0], 9))
    boxes[:, :8] = restored_bounding_boxes
    boxes[:, 8] = score_map[index_text[:, 0], index_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.2)

    return boxes