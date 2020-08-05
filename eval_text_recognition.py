import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pytesseract
from tqdm import tqdm
from PIL import Image
import textdistance


from models.east import EAST, load_east_model
from utils.data_loader_east import ReceiptDataLoaderEval
from utils.data_processing import get_files_with_extension, scale_bounding_box, get_brectangle_from_bbox, parse_annotation
from utils.score_functions_east import get_bounding_boxes_from_output
from utils.data_loader_text_recognition import resize_image_with_aspect_ratio, decode_ctc_output
from models.text_recognition_net import DenseNet, CRNN, load_text_recognition_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)


def get_test_bboxes(model_path, data_loader):
    """ Predict bounding boxes on test data with trained EAST net.

    :param model_path: Trained EAST model.
    :param data_loader: Data loader to load test data.
    :return: List of all bounding boxes for test images.
    """

    net = load_east_model(weight=model_path)
    net.to(device)

    data_iter = iter(data_loader)

    bboxes = []

    print("Predicting bounding boxes on test data ...")
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

            bboxes.append(restored_bboxes_scaled)

    return bboxes


def crop_bbox_images(bboxes, image_path):
    """ Crop the text images given the bounding boxes.

    :param bboxes: Bounding boxes.
    :param image_path: Image to crop from.
    :return: Croped images.
    """

    image = Image.open(image_path).convert("L")

    croped_images = []

    for box in bboxes:
        rect = get_brectangle_from_bbox(box.reshape(4, 2))

        croped_images.append(image.crop(rect))

    return croped_images


def text_recognition(model, image_path, annotation_path):
    """ Predict the text in one image given the ground truth bounding boxes.

    :param model: Path to model weights.
    :param image_path: Path to image.
    :param annotation_path: Path to annotations of image.
    :return: Return true and predicted text.
    """

    transform = transforms.Compose([transforms.ToTensor()])
    boxes, texts = parse_annotation(annotation_path)
    image = Image.open(image_path).convert("L")
    new_size = (256, 32)

    pred_text = []

    for (index, box) in enumerate(boxes):
        brectangle = get_brectangle_from_bbox(box)
        image_crop = image.crop(brectangle)
        image_crop = resize_image_with_aspect_ratio(image_crop, new_size)

        image_tensor = transform(image_crop)
        image_tensor = image_tensor.unsqueeze(0)
        log_preds = model.forward(image_tensor).permute(1, 0, 2)
        prediction = decode_ctc_output(log_preds)
        pred_text.append(prediction)

    return texts, pred_text


def get_prediction_metrics(all_true_text, all_pred_text):
    """ Calculate the word and character metric for all predicted texts.

    :param all_true_text: All ground truth texts.
    :param all_pred_text: All predicted texts.
    :return: Word accuracy and character accuracy (Levenshtein score).
    """

    levenstein = np.empty(len(all_true_text))
    score = np.empty(len(all_true_text))

    # Loop over all text images
    for j in range(len(all_true_text)):

        true_text = all_true_text[j]
        pred_text = all_pred_text[j]

        if len(true_text) != len(pred_text):
            print("Note the same number of true and predicted texts")

        s = 0
        lev = 0
        count = 0

        # Loop over all bounding boxes
        for i in range(len(true_text)):
            if pred_text[i] != "":
                if true_text[i] == pred_text[i]:
                    s += 1
                count += 1

                lev += textdistance.levenshtein.normalized_similarity(true_text[i], pred_text[i])

        if count == 0:
            score[j] = 0
            levenstein[j] = 0
        else:
            score[j] = s / count
            levenstein[j] = lev / count

    return score, levenstein


def eval_tesseract(data):
    """ Evaluate the performance of tesseract on the test data.

    :param data: Receipt data loader for evaluation.
    :return: All true texts and all predicted texts.
    """
    num_data = len(data.images_names)

    new_size = (256, 32)

    all_pred_text = []
    all_true_text = []

    print("Prdicting text for test data with tesseract ...")
    for i in tqdm(range(num_data)):
        image_name = data.images_names[i]
        annotation_name = data.annotation_names[i]
        image = Image.open(image_name).convert("L")
        boxes, texts = parse_annotation(annotation_name)

        pred_text = []

        for (index, box) in enumerate(boxes):
            brectangle = get_brectangle_from_bbox(box)
            image_crop = image.crop(brectangle)
            image_crop = resize_image_with_aspect_ratio(image_crop, new_size)
            prediction = pytesseract.image_to_string(image_crop)
            pred_text.append(prediction.upper())

        all_pred_text.append(pred_text)
        all_true_text.append(texts)

    return all_true_text, all_pred_text


def eval_rocognition_model(model, data):
    """ Evaluate the performance of the text recognition model on the test data. For this the ground truth
    bounding boxes are used.

    :param model: Path to trained model.
    :param data: Receipt data loader for evaluation.
    :return: All true texts and all predicted texts.
    """
    num_data = len(data.images_names)

    model = model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    new_size = (256, 32)

    all_pred_text = []
    all_true_text = []

    print("Predicting text of test data")
    with torch.no_grad():
        for i in tqdm(range(num_data)):
            image_name = data.images_names[i]
            annotation_name = data.annotation_names[i]
            image = Image.open(image_name).convert("L")
            boxes, texts = parse_annotation(annotation_name)

            pred_text = []

            for (index, box) in enumerate(boxes):
                brectangle = get_brectangle_from_bbox(box)
                image_crop = image.crop(brectangle)
                image_crop = resize_image_with_aspect_ratio(image_crop, new_size)

                image_tensor = transform(image_crop)
                image_tensor = image_tensor.unsqueeze(0)

                image_tensor = image_tensor.to(device)
                log_preds = model(image_tensor).permute(1, 0, 2)
                prediction = decode_ctc_output(log_preds)
                pred_text.append(prediction)

            all_pred_text.append(pred_text)
            all_true_text.append(texts)

        return all_true_text, all_pred_text


if __name__ == "__main__":
    model_path_east = "check_points_east/model_east_200.ckpt"
    path = "data/test_data/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ReceiptDataLoaderEval(path, transform=transform, type="test")
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    # tes_true, tes_pred = eval_tesseract(data)
    # score, levenstein = get_prediction_metrics(tes_true, tes_pred)
    # print("Score: ", np.mean(score[score != 0]))
    # print("Levestein similarity: ", np.mean(levenstein[levenstein != 0]))
    # print("Number of zeros: ", score[score == 0].shape)



    model_path = "check_points_final/model_rcnn_64_190.ckpt"
    model = load_text_recognition_model(model_name="CRNN", model_weights=model_path, out_put_size=64)
    all_true_text, all_pred_text = eval_rocognition_model(model, data)

    score, levenstein = get_prediction_metrics(all_true_text, all_pred_text)

    print("Score: ", np.mean(score))
    print("Levestein similarity: ", np.mean(levenstein))

