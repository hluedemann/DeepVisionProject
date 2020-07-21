from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch
from training_dense_net import *


def get_brectangle_from_bbox(bbox):
    bbox = np.array(bbox)
    x_min, y_min = np.min(bbox, axis=0)
    x_max, y_max = np.max(bbox, axis=0)
    return (x_min, y_min, x_max, y_max)


def add_padding_to_image(image, new_size):
    size = image.size
    x = int((new_size[0] - size[0]) / 2)
    y = int((new_size[1] - size[1]) / 2)
    image_np = np.array(image).swapaxes(1, 0)
    new_image_np = np.ones(new_size) * 255
    new_image_np[x: x + size[0], y: y + size[1]] = image_np
    new_image = Image.fromarray(new_image_np.swapaxes(1, 0))
    return new_image


def resize_image_with_aspect_ratio(image, new_size):
    size = image.size
    if size[0] > new_size[0] or size[1] > new_size[1]:
        x_scale = 128 / size[0]
        y_scale = 32 / size[1]
        if x_scale < y_scale:
            y_scale = x_scale
        else:
            x_scale = y_scale
        new_x = max(int(size[0] * x_scale), 1)
        new_y = max(int(size[1] * y_scale), 1)
        image = image.resize((new_x, new_y))
        image = add_padding_to_image(image, (128, 32))
        return image
    else:
        return image


char_list = 'abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:%-+=$ '


def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            # blank is at 0
            dig_lst.append(char_list.index(char) + 1)
        except:
            print(f"cant parse: {char}")

    return dig_lst


class ReceiptDataLoaderTask2(Dataset):

    def __init__(self, dir, transform=None, type="train"):
        super().__init__()

        self.names = os.listdir(dir)
        self.images_names = self.get_files_with_extension(dir, self.names, ".jpg")
        self.annotation_names = self.get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

        self.images = []
        self.texts = []
        self.texts_length = []
        self.texts_encoded = []

        for (image_name, annotation_name) in zip(self.images_names, self.annotation_names):
            images, texts, texts_length, texts_encoded = self.load_train_data(image_name, annotation_name)
            self.images += images
            self.texts_length += texts_length
            self.texts += texts
            self.texts_encoded += texts_encoded

        max_text_length = max(map(len, self.texts_encoded))
        print(f"max_length: {max_text_length}")
        # do not pad with zero
        print(self.texts_length[0])
        self.texts_encoded = np.array([t + [1] * (max_text_length - len(t)) for t in self.texts_encoded])
        # self.texts_length = [max_text_length for i in range(len(self.texts))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            # FIXME: add some usefull transform here
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
            image_crop = add_padding_to_image(image_crop, new_size)
            text_images.append(image_crop)

        return text_images, texts, texts_length, encoded_texts
        # return text_images, texts, encoded_texts

    def get_files_with_extension(self, dir, names, ext):
        list = []
        for name in names:
            e = os.path.splitext(name)[1]
            if e == ext:
                list.append(name)
        list.sort()
        return [os.path.join(dir, n) for n in list]
