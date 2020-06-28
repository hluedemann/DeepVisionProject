import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import tqdm.auto as tqdm
import os
from shutil import copyfile


def clean_data(in_dir, out_dir):
    """
        Copy image from in_dir to out_dir and remove duplicates
    """
    names = os.listdir(in_dir)
    for name in names:
        n = os.path.splitext(name)[0]
        if not n[-1] == ")":
            copyfile(os.path.join(in_dir, name), os.path.join(out_dir, name))


def parse_annotation(filename):
    f = open(filename, "r")
    lines = f.readlines()
    boxes = []
    texts = []

    for line in lines:
        split = line.split(",", 8)
        vertices = [(float(split[0]), float(split[1])), (float(split[2]), float(split[3])),
                    (float(split[4]), float(split[5])), (float(split[6]), float(split[7]))]
        boxes.append(vertices)
        texts.append(split[8].rstrip("\n"))

    f.close()
    return boxes, texts


def plot_image(image, outfile):
    fig, ax = plt.subplots(1)
    plt.axis("off")
    ax.imshow(image)
    plt.savefig(f"output/{outfile}.png")
    plt.show()


def add_bounding_box(image, boxes, color):
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        draw.polygon(boxes[i], outline=color)
    del draw


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def shrink_bbox(vertices):
    def move_points(p1, p2, r1, r2):
        x_dist = p2[0] - p1[0]
        y_dist = p2[1] - p1[1]
        length = dist(p1, p2)

        coeff_1 = 0.3 * r1 / length
        coeff_2 = -0.3 * r2 / length

        p1 = (p1[0] + coeff_1 * x_dist, p1[1] + coeff_1 * y_dist)
        p2 = (p2[0] + coeff_2 * x_dist, p2[1] + coeff_2 * y_dist)

        return p1, p2

    p1, p2, p3, p4 = vertices

    # calculate reference length
    r1 = min(dist(p1, p2), dist(p1, p4))
    r2 = min(dist(p2, p3), dist(p2, p1))
    r3 = min(dist(p3, p4), dist(p3, p2))
    r4 = min(dist(p4, p1), dist(p4, p3))

    # next we need to dermine the longer edges
    if dist(p1, p2) + dist(p3, p4) > dist(p1, p4) + dist(p2, p3):
        p1, p2 = move_points(p1, p2, r1, r2)
        p3, p4 = move_points(p3, p4, r3, r4)
        p1, p4 = move_points(p1, p4, r1, r4)
        p2, p3 = move_points(p2, p3, r2, r3)
    else:
        p1, p4 = move_points(p1, p4, r1, r4)
        p2, p3 = move_points(p2, p3, r2, r3)
        p1, p2 = move_points(p1, p2, r1, r2)
        p3, p4 = move_points(p3, p4, r3, r4)

    return [p1, p2, p3, p4]


def get_shrinked_bboxes(boxes):
    shrinked_boxes = []

    for box in boxes:
        shrinked_boxes.append(shrink_bbox(box))

    return shrinked_boxes

def convert_vert_list_to_tuple(vertices):
    vert = []
    for list in vertices.reshape(-1, 8):
        vert.append([(list[0], list[1]), (list[2], list[3]), (list[4], list[5]), (list[6], list[7])])

    return vert


def resize_image_and_boxes(image, boxes, new_size):

    resized_image = image.resize(new_size)
    scale_x = resized_image.size[0] / image.size[0]
    scale_y = resized_image.size[1] / image.size[1]

    b = np.array(np.copy(boxes))

    b[:, :, 0] = np.around(b[:, :, 0] * scale_x).astype(np.int32)
    b[:, :, 1] = np.around(b[:, :, 1] * scale_y).astype(np.int32)

    return resized_image, b


if __name__ == "__main__":


    clean_data("data/task1_train", "data/train_data")

    example_image = "data/train_data/X00016469612.jpg"
    example_annotation = "data/train_data/X00016469612.txt"

    # Plot original boxes
    boxes, texts = parse_annotation(example_annotation)
    print(boxes)
    example = Image.open(example_image)
    add_bounding_box(example, boxes, "blue")
    plot_image(example, "original_boxes")

    # Plot schinked bboxes
    shrinked_boxes = get_shrinked_bboxes(boxes)
    print(shrinked_boxes)
    add_bounding_box(example, shrinked_boxes, "red")
    plot_image(example, "shrinked_boxes")