import numpy as np
import matplotlib.pyplot as plt


def plot_loss(path, out_path):

    x, y = np.loadtxt(path, delimiter=",").T


    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(x, y, label="Loss")

    plt.savefig(out_path)
    plt.show()


def plot_score(out_path):
    check_points = [25, 50, 75, 100, 125, 150, 200, 225, 250, 275, 300, 325, 350, 375]
    avg_IoU = [0.7602516841642742, 0.7878658391045087, 0.7978997332602118, 0.7930503005765398, 0.8057804028332763,
               0.8074126371756708, 0.8168595300056662, 0.8044971537799073, 0.8119545779629576, 0.8088774673968238,
               0.812576274257848, 0.8108968157135467, 0.8010301182699692, 0.817425015613535]

    #[25, 50, 75, 100, 125, 150, 200, 250]
    #[0.7725300907644802, 0.7904123828428118, 0.796177305573827, 0.8026781692908358, 0.7938339271978643,
     #0.8043857464640476, 0.8087711312677133, 0.8178284154944252]

    plt.title("IoU score on test set")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.plot(check_points, avg_IoU, label="IoU")
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":

    path = "check_points/loss_east.txt"
    out = "output/loss.pdf"

    plot_loss(path, out)
    plot_score("output/IoU.pdf")