import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


from east_net import EAST
from data_loader_east import ReceiptDataLoader
from data_processing import *
from score_functions_east import *
from cv2 import fillPoly

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

def calculate_IoU(gt_bboxs, pred_bboxs, image_size):

    gt_mask = np.zeros((int(image_size[1]), int(image_size[0])))
    pred_mask = np.zeros((int(image_size[1]), int(image_size[0])))

    gt_bboxs = gt_bboxs.reshape(-1, 4, 2).astype(np.int32)
    pred_bboxs = pred_bboxs.reshape(-1, 4, 2).astype(np.int32)

    for box in gt_bboxs:
        fillPoly(gt_mask, [box], 1)
    for box in pred_bboxs:
        fillPoly(pred_mask, [box], 1)

    intersection = np.argwhere(gt_mask + pred_mask == 2).shape[0]
    union = np.sum(gt_mask + pred_mask - gt_mask * pred_mask)

    return intersection / union

def evaluate_east(model_path, data_loader):

    net = EAST()
    net.load_state_dict(torch.load(model_path))

    data_iter = iter(data_loader)

    i = 0
    IoUs = np.zeros(len(data))

    net.to(device)
    with torch.no_grad():
        for img, scale, boxes in tqdm(data_iter):

            boxes = boxes.numpy()[0]
            scale = scale.numpy()[0]

            img = img.to(device)
            score, geo = net(img)

            score = score.cpu()
            geo = geo.cpu()

            restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(),
                                                             geo[0].permute(1, 2, 0).detach().numpy())
            restored_bboxes_scaled = scale_bounding_box(restored_bboxes[:, :8], 4.0 / scale).reshape(-1, 8)

            image_size_resized = [img.shape[2], img.shape[3]]
            IoU = calculate_IoU(boxes, restored_bboxes_scaled, image_size_resized / scale)
            IoUs[i] = IoU

            # if IoU < 0.4:
            #     print("Imag: ",data.images_names[i])
            #     print("Box: ",data.annotaion_names[i])
            #     image = Image.open(data.images_names[i])
            #     add_bounding_box(image, restored_bboxes_scaled[:, :8], "red")
            #     add_bounding_box(image, boxes, "blue")
            #     image.show()
            i += 1

    return IoUs


if __name__ == "__main__":

    test_data = "data/test_data"

    checkpoints = [25, 50, 75, 100, 125, 150, 200, 225, 250, 275, 300, 325, 350, 375]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ReceiptDataLoader(test_data, transform=transform, type="test")
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    scores = []

    print("Evaluating EAST model")

    for c in checkpoints:
        model_path = f"check_points/model_east_{c}.ckpt"

        IoUs = evaluate_east(model_path, data_loader)

        scores.append(np.average(IoUs))
        print("\nMin IoU score: ", np.min(IoUs))
        print("Max IoU score: ", np.max(IoUs))
        print("Average IoU score: ", np.mean(IoUs))



print("Final result")
print(checkpoints)
print(scores)