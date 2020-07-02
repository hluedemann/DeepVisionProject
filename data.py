
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os

from score_functions import *
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

        self.images = np.empty((len(self.images_names), *self.new_size, 3))
        self.scores = np.empty((len(self.images_names), 128, 128))
        self.geos = np.empty((len(self.images_names), 128, 128, 8))
        self.shortest_edges = np.empty((len(self.images_names), 128, 128))

        print("Loading train data:")
        for idx in tqdm(range(len(self.images_names))):

            image, score, geo = self.load_train_data(idx)



            for i in range(128):
                for j in range(128):

                    if score[i, j] == 0.0:
                        continue
                    deltas = geo[i, j, :]
                    self.shortest_edges[i, j] = self.get_shortest_edge(deltas)


            self.images[idx] = image
            self.scores[idx] = score
            self.geos[idx] = geo

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        if self.transform is not None:
            tensor = self.transform(self.images[idx])
        else:
            tensor = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1)

        return tensor.type(torch.FloatTensor),\
               torch.tensor(self.scores[idx]).type(torch.FloatTensor),\
               torch.tensor(self.geos[idx]).permute(2, 0, 1).type(torch.FloatTensor),\
               torch.tensor(self.shortest_edges[idx]).type(torch.FloatTensor)

    def get_shortest_edge(self, deltas):

        n1 = dist(*deltas[[0, 1, 2, 3]].reshape(2, 2))
        n2 = dist(*deltas[[2, 3, 4, 5]].reshape(2, 2))
        n3 = dist(*deltas[[4, 5, 6, 7]].reshape(2, 2))
        n4 = dist(*deltas[[6, 7, 0, 1]].reshape(2, 2))

        return np.min([n1, n2, n3, n4])

    def load_train_data(self, idx):

        image = Image.open(self.images_names[idx]).convert("RGB")
        boxes, texts = parse_annotation(self.annotaion_names[idx])

        image_resized, boxes, scale = resize_image_and_boxes(image, boxes, (512, 512))

        score_map, geo_map = get_score_values(image_resized, boxes, scale=0.25)

        return image_resized, score_map, geo_map


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

        if self.type == "train":
            self.annotaion_names = self.get_files_with_extension(dir, self.names, ".txt")

        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        if self.type == "train":
            return self.get_train_data(idx)
        else:
            t, s = self.get_test_data(idx)
            return t, s

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

        image_resized = image.resize((512, 512))

        scale = np.zeros(2)
        scale[0] = image_resized.size[0] / image.size[0]
        scale[1] = image_resized.size[1] / image.size[1]

        if self.transform is not None:
            tensor = self.transform(image_resized)
        else:
            tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)

        return tensor, scale

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

    print(image.shape)
    print(score.shape)
    print(geo.shape)
    print(edge.shape)

    print(edge[5])
