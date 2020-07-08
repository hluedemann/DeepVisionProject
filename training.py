from loss import *
from east_net import *

from tqdm import tqdm
from loss import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)


def train(dir):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data = ReceiptDataLoaderRam("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=4, shuffle=True)

    criterion = CustomLoss()

    model = EAST()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    for epoch in tqdm(range(100), desc="Epoch"):

        model.train()

        epoch_loss = 0

        for step, (images, score, geo, edge) in enumerate(tqdm(data_loader, desc='Batch')):

            images, score, geo, edge = images.to(device), score.to(device), geo.to(device), edge.to(device)

            score_pred, geo_pred = model(images)

            loss = criterion(score, score_pred, geo, geo_pred, edge)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("### Epcoh Loss: ", epoch_loss)
        torch.save(model.state_dict(), 'check_points/model_geo_loss_{}.ckpt'.format(epoch))

if __name__ == "__main__":

    dir = "data/train_data"

    train(dir)