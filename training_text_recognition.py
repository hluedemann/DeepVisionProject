###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   This script implements a function to train the text recognition models.
###################################################################################################


import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from models.text_recognition_net import load_text_recognition_model
from utils.data_loader_text_recognition import ReceiptDataLoaderTextRecognition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)


def train_text_recognition(data_loader, model_name="DenseNetLinear", model_path=None, out_put_size=64):
    """ Train the text recognition networks.

    :param data_loader: Data loader to load the train data.
    :param model_name: Name of the model to train. One of "DenseNetLinear", "DenseNetRNN", "CRNN"
    :param model_path: Checkpoint of model.
    :param out_put_size: Maximal word length that can be recognized. This is also the input size for the CTC loss.
                             Currently 32 and 64 are supported.
    """

    assert model_name == "DenseNetLinear" or model_name == "DenseNetRNN" or model_name == "CRNN", "Unsupported model " \
                                                                                                  "name specified! "
    assert out_put_size == 64 or out_put_size == 32, "Unsupported output size specified!"

    criterion = nn.CTCLoss(reduction="sum", zero_infinity=True)

    model = load_text_recognition_model(model_name, model_path, out_put_size)
    model.to(device)

    lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in tqdm(range(30, 60), desc="Epoch"):

        model.train()

        epoch_loss = 0

        for step, (images, texts_lengths, texts_encoded, texts) in enumerate(tqdm(data_loader, desc=None)):
            images = images.to(device)

            # reshape to (T, N, C)
            log_pred = model(images).permute(1, 0, 2)
            # create input_lengths with T
            batch_size = log_pred.shape[1]

            if model_name == "DenseNetLinear" or model_name == "DenseNetRNNN":
                if out_put_size == 32:
                    input_lengths = torch.full(size=(batch_size,), fill_value=32, dtype=torch.int)
                else:
                    input_lengths = torch.full(size=(batch_size,), fill_value=64, dtype=torch.int)
            else:
                if out_put_size == 32:
                    input_lengths = torch.full(size=(batch_size,), fill_value=31, dtype=torch.int)
                else:
                    input_lengths = torch.full(size=(batch_size,), fill_value=63, dtype=torch.int)

            loss = criterion(log_pred, texts_encoded, input_lengths, texts_lengths) / batch_size
            epoch_loss += loss / len(data_loader) * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("### Epoch Loss: ", epoch_loss)
        torch.save(model.state_dict(), "check_points_text_recognition/model_{}_{}_{}.ckpt".format(model_name, out_put_size, epoch))

        with open(f"check_points_text_recognition/loss_{model_name}.txt", "a+") as f:
            f.write(f"{epoch}, {epoch_loss}\n")


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])

    data = ReceiptDataLoaderTextRecognition("data/train_data", transform)
    batch_size = 16
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    train_text_recognition(data_loader,
                           model_name="DenseNetRNN",
                           model_path=None,
                           out_put_size=64)
