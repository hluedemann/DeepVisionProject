###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   Utility functions to plot the results of the text detection and text recognition.
###################################################################################################



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_image(image, outfile):
    """ Plot PIL image.

    :param image: Image to plot.
    :param outfile: File name for saving.
    """
    image.show()
    image.save(f"output/{outfile}.png")


def add_bounding_box(image, boxes, color):
    """ Add the bouning boxes to an image.

    :param image: Image to draw to.
    :param boxes: Boxes to draw. Boxes can have the form [[(x1, y1), (x2, y2) ..],[..]] or
                  [[x1, y1, x2, y2, ...], [...]. They can be either lists or numpy arrays.
    :param color: Color to use for the boxes.
    """

    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        if isinstance(boxes[i], np.ndarray):
            draw.polygon(boxes[i].tolist(), outline=color)
        elif isinstance(boxes[i], list):
            draw.polygon(boxes[i], outline=color)
    del draw


def plot_loss_east(path, out_path):
    """ Create plot of loss for EAST model.

    :param path: File containing the loss values.
    :param out_path: Output path of plot
    """

    x, y = np.loadtxt(path, delimiter=",").T

    plt.title("Loss EAST Model")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"$L_{score}+L_{geo}$", fontsize=14)
    plt.grid()
    plt.plot(x, y)
    plt.savefig(out_path)
    plt.show()


def plot_score_east(in_path, out_path):
    """ Create plot of score for EAST model.

    :param in_path: File containing the score values.
    :param out_path: Output path of plot.
    """
    epoch, iou = np.loadtxt(in_path, delimiter=",").T

    plt.title("IoU score on test set")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("IoU", fontsize=14)
    plt.grid()
    plt.plot(epoch, iou, label="IoU")
    plt.savefig(out_path)
    plt.show()


def plot_scores_text_recognition():
    """ Create plot of score metrics for text recognition.
    """

    d_32_epochs, d_32_score, d_32_lev = np.loadtxt("check_points_for_loss/dense_32/eval.txt", delimiter=",").T
    d_64_epochs, d_64_score, d_64_lev = np.loadtxt("check_points_for_loss/dense_64/eval.txt", delimiter=",").T
    crnn_32_epochs, crnn_32_score, crnn_32_lev = np.loadtxt("check_points_for_loss/rcnn_32/eval.txt", delimiter=",").T
    crnn_64_epochs, crnn_64_score, crnn_64_lev = np.loadtxt("check_points_for_loss/rcnn_64/eval.txt", delimiter=",").T

    plt.figure(figsize=(8, 8))
    plt.title("Word Accuracy Different Models", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Word Accuracy", fontsize=15)
    plt.plot(d_32_epochs, d_32_score, label="Dense Net 32")
    plt.plot(d_64_epochs, d_64_score, label="Dense Net 64")
    plt.plot(crnn_32_epochs, crnn_32_score, label="CRNN 32")
    plt.plot(crnn_64_epochs, crnn_64_score, label="CRNN 64")
    plt.grid()
    plt.legend(loc="lower right", fontsize=15)
    plt.xscale("log")
    plt.ylim(0.1, 0.7)
    plt.savefig("output/score_recognition.pdf")

    plt.figure(figsize=(8, 8))
    plt.title("Levenshtein Score Different Models", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Levenshtein Score", fontsize=15)
    plt.plot(d_32_epochs, d_32_lev, label="Dense Net 32")
    plt.plot(d_64_epochs, d_64_lev, label="Dense Net 64")
    plt.plot(crnn_32_epochs, crnn_32_lev, label="CRNN 32")
    plt.plot(crnn_64_epochs, crnn_64_lev, label="CRNN 64")
    plt.grid()
    plt.ylim(0.65, 1.0)
    plt.legend(loc="lower right", fontsize=15)
    plt.xscale("log")
    plt.savefig("output/levenshtein_recognition.pdf")
    plt.show()


def plot_loss_text_recognition():
    """ Create plot of loss for text recognition.
    """

    d_32_epochs, d_32_loss = np.loadtxt("check_points_for_loss/dense_32/loss.txt", delimiter=",").T
    d_64_epochs, d_64_loss = np.loadtxt("check_points_for_loss/dense_64/loss.txt", delimiter=",").T
    crnn_32_epochs, crnn_32_loss = np.loadtxt("check_points_for_loss/rcnn_32/loss.txt", delimiter=",").T
    crnn_64_epochs, crnn_64_loss = np.loadtxt("check_points_for_loss/rcnn_64/loss.txt", delimiter=",").T

    plt.title("Loss Text Recognition Models")
    plt.plot(d_32_epochs, d_32_loss, label="Dense Net 32")
    plt.plot(d_64_epochs, d_64_loss, label="Dense Net 64")
    plt.plot(crnn_32_epochs, crnn_32_loss, label="CRNN 32")
    plt.plot(crnn_64_epochs, crnn_64_loss, label="CRNN 64")

    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("CTC Loss", fontsize=14)
    plt.yscale("log")

    plt.grid()
    plt.legend()
    plt.savefig("output/loss_recognition_models.pdf")
    plt.show()
