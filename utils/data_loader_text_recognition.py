###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   Data loader for the training and evaluation of the text recognition models.
###################################################################################################



import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from trdg.generators import GeneratorFromDict

from utils.data_processing import get_files_with_extension, parse_annotation, get_brectangle_from_bbox
from models.text_recognition_net import char_list


def add_padding_to_image(image, new_size):
    """ Add padding to image if it is smaller as target size.

    :param image: Image to pad.
    :param new_size: Target size.
    :return: Return padded image.
    """

    size = image.size
    x = int((new_size[0] - size[0]) / 2)
    y = int((new_size[1] - size[1]) / 2)
    image_np = np.array(image).swapaxes(1, 0)
    new_image_np = np.ones(new_size) * 255
    new_image_np[x: x + size[0], y: y + size[1]] = image_np
    new_image = Image.fromarray(new_image_np.swapaxes(1, 0))

    return new_image


def resize_image_with_aspect_ratio(image, new_size):
    """ Resize image while keeping the aspect ratio.
    This is done in order not to distort the text for recognition.

    :param image: Image.
    :param new_size: Target size.
    :return: Resized and padded iamge.
    """

    size = image.size
    if size[0] > new_size[0] or size[1] > new_size[1]:
        x_scale = new_size[0] / size[0]
        y_scale = new_size[1] / size[1]
        if x_scale < y_scale:
            y_scale = x_scale
        else:
            x_scale = y_scale
        new_x = max(int(size[0] * x_scale), 1)
        new_y = max(int(size[1] * y_scale), 1)
        image = image.resize((new_x, new_y))
        image = add_padding_to_image(image, (new_size[0], new_size[1]))
        return image
    else:
        return add_padding_to_image(image, (new_size[0], new_size[1]))


def encode_to_labels(txt):
    """ Encode text to generate labels for text recognition.
    The text is just encoded with its index in the char list plus one. Zero is reserved for the blank.

    :param txt: Text to encode.
    :return: Encoded text.
    """

    dig_lst = []
    for index, char in enumerate(txt):
        try:
            # blank is at 0
            dig_lst.append(char_list.index(char) + 1)
        except:
            print(f"cant parse: {char}")

    return dig_lst


class ReceiptDataLoaderTextRecognition(Dataset):
    """ Data loader for the text recognition.
    """

    def __init__(self, dir, transform=None, artificial=False):
        super().__init__()

        self.names = os.listdir(dir)
        self.images_names = get_files_with_extension(dir, self.names, ".jpg")
        self.annotation_names = get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

        self.images = []
        self.texts = []
        self.texts_length = []
        self.texts_encoded = []

        print("Loading train data ...")
        for (image_name, annotation_name) in tqdm(zip(self.images_names, self.annotation_names)):
            images, texts, texts_length, texts_encoded = self.load_train_data(image_name, annotation_name)
            self.images += images
            self.texts_length += texts_length
            self.texts += texts
            self.texts_encoded += texts_encoded

        # generate additional train data
        print(f"Train data: {len(self.texts)}")
        if artificial:
            generator = GeneratorFromDict(count=len(self.images), width=256)

            for image, text in generator:
                self.texts.append(text)
                text = ''.join(c for c in text if c in char_list)
                self.texts_length.append(len(text))
                self.texts_encoded.append(encode_to_labels(text))

                image = image.convert("L")
                self.images.append(image)

            print(f"Train data with artificial data: {len(self.texts)}")

        max_text_length = max(map(len, self.texts_encoded))
        print(f"max_length: {max_text_length}")
        # do not pad with zero
        self.texts_encoded = np.array([t + [1] * (max_text_length - len(t)) for t in self.texts_encoded])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1)

        return tensor.type(torch.FloatTensor), \
               torch.tensor(self.texts_length[idx]).type(torch.IntTensor), \
               torch.tensor(self.texts_encoded[idx]).type(torch.IntTensor), \
               self.texts[idx]

    def load_train_data(self, image_path, annotation_path):
        boxes, texts = parse_annotation(annotation_path)

        encoded_texts = []
        texts_length = []

        for text in texts:
            text = ''.join(c for c in text if c in char_list)
            texts_length.append(len(text))
            encoded_texts.append(encode_to_labels(text))

        image = Image.open(image_path).convert("L")
        text_images = []

        new_size = (256, 32)

        for (index, box) in enumerate(boxes):
            brectangle = get_brectangle_from_bbox(box)
            image_crop = image.crop(brectangle)
            image_crop = resize_image_with_aspect_ratio(image_crop, new_size)
            text_images.append(image_crop)

        return text_images, texts, texts_length, encoded_texts


def decode_ctc_output(tensor):
    """ Decode the output of CTC loss to get the predicted text.

    :param tensor: CTC output
    :return: Predicted text.
    """

    indices = torch.argmax(tensor[:, 0, :], dim=1).cpu().detach().numpy()
    decoded_text = ''
    blank = False
    for index in indices:
        if index > 0:
            char = char_list[index - 1]
            if blank or len(decoded_text) == 0:
                decoded_text += char
            elif char != decoded_text[-1]:
                decoded_text += char
            blank = False
        else:
            blank = True

    return decoded_text