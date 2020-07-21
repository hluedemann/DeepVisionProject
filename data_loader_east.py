
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os

from score_functions_east import *
from data_processing import *

from tqdm import tqdm


class ReceiptDataLoaderRam(Dataset):

    def __init__(self, dir, transform=None, type="train"):
        super().__init__()

        self.names = os.listdir(dir)
        self.images_names = self.get_files_with_extension(dir, self.names, ".jpg")
        self.annotaion_names = self.get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

        self.new_size = (512, 512)

        self.images = np.zeros((len(self.images_names), *self.new_size, 3))
        self.scores = np.zeros((len(self.images_names), 128, 128))
        self.geos = np.zeros((len(self.images_names), 128, 128, 8))
        self.shortest_edges = np.zeros((len(self.images_names), 128, 128))

        print("Loading train data:")
        for idx in tqdm(range(len(self.images_names))):

            image, score, geo, boxes = self.load_train_data(idx)

            mask = np.argwhere(score != 0.0)
            deltas = geo[mask[:, 0], mask[:, 1], :]

            self.images[idx] = image
            self.scores[idx] = score
            self.geos[idx] = geo
            self.shortest_edges[idx, mask[:, 0], mask[:, 1]] = self.get_shortest_edge(deltas)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.images[idx]))
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1)

        return tensor.type(torch.FloatTensor),\
               torch.tensor(self.scores[idx]).type(torch.FloatTensor),\
               torch.tensor(self.geos[idx]).permute(2, 0, 1).type(torch.FloatTensor),\
               torch.tensor(self.shortest_edges[idx]).type(torch.FloatTensor)


    def get_shortest_edge(self, deltas):

        n1 = np.sqrt((deltas[:, 0] - deltas[:, 2]) ** 2 + (deltas[:, 1] - deltas[:, 3]) ** 2)
        n2 = np.sqrt((deltas[:, 2] - deltas[:, 4]) ** 2 + (deltas[:, 3] - deltas[:, 5]) ** 2)
        n3 = np.sqrt((deltas[:, 4] - deltas[:, 6]) ** 2 + (deltas[:, 5] - deltas[:, 7]) ** 2)
        n4 = np.sqrt((deltas[:, 6] - deltas[:, 0]) ** 2 + (deltas[:, 7] - deltas[:, 1]) ** 2)

        return np.min([n1, n2, n3, n4])

    def load_train_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotaion_names[idx])

        image_resized, boxes, scale = resize_image_and_boxes(image, boxes, (512, 512))
        score_map, geo_map = get_score_values(image_resized, boxes, scale=0.25)

        return image_resized, score_map, geo_map, boxes


    def get_files_with_extension(self, dir, names, ext):
        list = []
        for name in names:
            e = os.path.splitext(name)[1]
            if e == ext:
                list.append(name)
        list.sort()
        return [os.path.join(dir, n) for n in list]

class ReceiptDataLoader(Dataset):

    def __init__(self, dir, transform=None, type="train"):
        super().__init__()

        self.names = os.listdir(dir)
        self.type = type
        self.images_names = self.get_files_with_extension(dir, self.names, ".jpg")
        self.annotaion_names = self.get_files_with_extension(dir, self.names, ".txt")
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        if self.type == "train":
            return self.get_train_data(idx)
        else:
            return self.get_test_data(idx)

    def get_train_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotaion_names[idx])

        image_resized, boxes, scale = resize_image_and_boxes(image, boxes, (512, 512))

        if self.transform is not None:
            tensor = self.transform(image_resized)
        else:
            tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)

        score_map, geo_map = get_score_values(image_resized, boxes, scale=0.25)

        return tensor, torch.tensor(score_map), torch.tensor(geo_map).permute(2, 0, 1)

    def get_test_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotaion_names[idx])
        image_resized = image.resize((512, 512))

        scale = np.zeros(2)
        scale[0] = image_resized.size[0] / image.size[0]
        scale[1] = image_resized.size[1] / image.size[1]

        if self.transform is not None:
            tensor = self.transform(image_resized)
        else:
            tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)
        b = np.array(boxes).reshape((-1, 8))
        return tensor, scale, b

    def get_files_with_extension(self, dir, names, ext):

        list = []
        for name in names:
            e = os.path.splitext(name)[1]
            if e == ext:
                list.append(name)
        list.sort()
        return [os.path.join(dir, n) for n in list]



if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = ReceiptDataLoaderRam("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=16, shuffle=False)

    image, score, geo, edge = next(iter(data_loader))
