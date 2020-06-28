from loss import *
from east_net import *

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)


def train(dir):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = ReceiptDataLoader("data/train_data", transform)
    data_loader = DataLoader(data, batch_size=2, shuffle=True)

    criterion = CustomLoss()

    model = EAST()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    for epoch in tqdm(range(5), desc="Epoch"):

        model.train()

        epoch_loss = 0

        for step, (images, score, geo) in enumerate(tqdm(data_loader, desc='Batch')):

            images, score, geo = images.to(device), score.to(device), geo.to(device)
            score_pred, geo_pred = model(images)

            loss = criterion(score, score_pred, geo, geo_pred)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("### Epcoh Loss: ", epoch_loss)

if __name__ == "__main__":

    dir = "data/train_data"

    train(dir)