from trdg.generators import GeneratorFromDict
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dense_net import *

from utils.data_processing import parse_annotation

generator = GeneratorFromDict(count=3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

"""
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
    if size[0] > new_size[0] or size[1] > new_size[0]:
        x_scale = new_size[0] / size[0]
        y_scale = new_size[1] / size[1]
        if x_scale < y_scale:
            y_scale = x_scale
        else:
            x_scale = y_scale
        new_x = int(size[0] * x_scale)
        new_y = int(size[1] * y_scale)
        image = image.resize((int(size[0] * x_scale), int(size[1] * y_scale)))
        image = add_padding_to_image(image, (new_size[0], new_size[1]))
        return image
    else:
        return add_padding_to_image(image, (new_size[0], new_size[1]))


char_list = "abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:,/\*!&?%()-_ "
#char_list = 'abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:%-+=$ '

def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(f"cant parse: {char}")

    return dig_lst


def load_image_task2(image_path, annotation_path):
    boxes, texts = parse_annotation(annotation_path)

    encoded_texts = []
    texts_length = []

    for text in texts:
        texts_length.append(len(texts_length))
        encoded_texts.append(encode_to_labels(text))

    image = Image.open(image_path).convert("L")
    text_images = []

    new_size = (128, 32)

    for (index, box) in enumerate(boxes):
        brectangle = get_brectangle_from_bbox(box)
        image_crop = image.crop(brectangle)
        image_crop = resize_image_with_aspect_ratio(image_crop, new_size)
        #image_crop = add_padding_to_image(image_crop, new_size)
        text_images.append(image_crop)

    return text_images, texts, texts_length, encoded_texts
"""

# char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:,/\*!&?%()-_ "
char_list = "abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:,/\*!&?%()-_ "
# char_list = 'abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:%-+=$ '

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

        for (image_name, annotation_name) in tqdm(zip(self.images_names, self.annotation_names)):
            images, texts, texts_length, texts_encoded = self.load_train_data(image_name, annotation_name)
            self.images += images
            self.texts_length += texts_length
            self.texts += texts
            self.texts_encoded += texts_encoded
        print(f"before: {len(self.texts)}")
        # generate additional train data
        """
        generator = GeneratorFromDict(count=len(self.images), width=256)

        for image, text in generator:
            self.texts.append(text)
            text = ''.join(c for c in text if c in char_list)
            self.texts_length.append(len(text))
            self.texts_encoded.append(encode_to_labels(text))

            image = image.convert("L")
            #image = add_padding_to_image(image, new_size)
            self.images.append(image)

        print(f"after: {len(self.texts)}")
        """

        max_text_length = max(map(len, self.texts_encoded))
        print(f"max_length: {max_text_length}")
        # do not pad with zero
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
            # image_crop = add_padding_to_image(image_crop, new_size)
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


def encode_ctc(tensor):
    indices = torch.argmax(tensor[:, 0, :], dim=1).cpu().detach().numpy()
    encoded_text = ''
    blank = False
    for index in indices:
        if index > 0:
            char = char_list[index - 1]
            if blank or len(encoded_text) == 0:
                encoded_text += char
            elif char != encoded_text[-1]:
                encoded_text += char
            blank = False
        else:
            blank = True

    return encoded_text


def train_task2(data, model_name="DenseNet", model_path=None):
    criterion = nn.CTCLoss(reduction="sum", zero_infinity=True)

    if model_name == "DenseNet":
        model = DenseNet()
    elif model_name == "RCNN":
        model = RCNN()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    start = 186
    for epoch in tqdm(range(start, 250), desc="Epoch"):

        model.train()

        epoch_loss = 0

        for step, (images, texts_lengths, texts_encoded, texts) in enumerate(tqdm(data_loader, desc=None)):
            images = images.to(device)

            # reshape to (T, N, C)
            log_pred = model(images).permute(1, 0, 2)
            # create input_lengths with T
            batch_size = log_pred.shape[1]
            if model_name == "DenseNet":
                input_lengths = torch.full(size=(batch_size,), fill_value=64, dtype=torch.int)
            elif model_name == "RCNN":
                input_lengths = torch.full(size=(batch_size,), fill_value=63, dtype=torch.int)
                # input_lengths = torch.full(size=(batch_size,), fill_value=31, dtype=torch.int)
            # print(f"texts: {texts}")
            # print(f"output: {encode_ctc(log_pred)}")
            # print(f"prediction: {torch.argmax(log_pred[:,0,:], dim=1)}")
            # print(f"texts_encoded: {texts_encoded}")

            loss = criterion(log_pred, texts_encoded, input_lengths, texts_lengths) / batch_size
            epoch_loss += loss / len(data) * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("### Epoch Loss: ", epoch_loss)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "check_points_for_loss/rcnn_64/model_rcnn_64_{}.ckpt".format(epoch))

        with open("check_points_for_loss/rcnn_64/loss_new.txt", "a+") as f:
            f.write(f"{epoch}, {epoch_loss}\n")


if __name__ == "__main__":
    example_image = "data/train_data/X00016469612.jpg"
    example_annotation = "data/train_data/X00016469612.txt"

    transform = transforms.Compose([transforms.ToTensor()])

    data = ReceiptDataLoaderTask2("data/train_data", transform)
    batch_size = 32
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    train_task2(data, "RCNN", "check_points_for_loss/rcnn_64/model_rcnn_64_185.ckpt")
