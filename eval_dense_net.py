from utils.score_functions_east import *
from training_dense_net import *
from models.east import EAST
from utils.data_loader_east import ReceiptDataLoaderEval

import pytesseract


def get_files_with_extension(dir, names, ext):
    list = []
    for name in names:
        e = os.path.splitext(name)[1]
        if e == ext:
            list.append(name)
    list.sort()
    return [os.path.join(dir, n) for n in list]


def get_test_bboxes(model_path, data_loader):
    net = EAST()
    net.load_state_dict(torch.load(model_path))

    data_iter = iter(data_loader)

    bboxes = []

    net.to(device)
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


def get_brectangle_from_bbox(bbox):
    bbox = np.array(bbox)
    x_min, y_min = np.min(bbox, axis=0)
    x_max, y_max = np.max(bbox, axis=0)
    return (x_min, y_min, x_max, y_max)


def crop_bbox_images(bboxes, image_path):
    image = Image.open(image_path).convert("L")

    croped_images = []

    for box in bboxes:
        rect = get_brectangle_from_bbox(box.reshape(4, 2))

        croped_images.append(image.crop(rect))

    return croped_images


def text_recognition(model, image_path, annotation_path):
    transform = transforms.Compose([transforms.ToTensor()])

    boxes, texts = parse_annotation(annotation_path)

    image = Image.open(image_path).convert("L")

    new_size = (256, 32)

    pred_text = []

    for (index, box) in enumerate(boxes):
        brectangle = get_brectangle_from_bbox(box)
        image_crop = image.crop(brectangle)
        image_crop = resize_image_with_aspect_ratio(image_crop, new_size)
        # image_crop = add_padding_to_image(image_crop, new_size)

        image_tensor = transform(image_crop)
        image_tensor = image_tensor.unsqueeze(0)
        log_preds = model.forward(image_tensor).permute(1, 0, 2)
        prediction = encode_ctc(log_preds)
        pred_text.append(prediction)

        print(f"prediction: {prediction}")
        print(f"text: {texts[index]}")

    return texts, pred_text


import textdistance


def get_number_correct_words(all_true_text, all_pred_text):
    levenstein = np.empty(len(all_true_text))
    score = np.empty(len(all_true_text))

    for j in range(len(all_true_text)):

        true_text = all_true_text[j]
        pred_text = all_pred_text[j]

        if len(true_text) != len(pred_text):
            print("Note the same number of true and predicted texts")

        s = 0
        lev = 0
        count = 0

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
                prediction = encode_ctc(log_preds)
                pred_text.append(prediction)

                # print(f"prediction: {prediction}")
                # print(f"text: {texts[index]}")

            all_pred_text.append(pred_text)
            all_true_text.append(texts)

        return all_true_text, all_pred_text


def load_model_task2(model_name="RCNN", model_path=None):
    if model_name == "DenseNet":
        model = DenseNet()
    else:# model_name == "RCNN":
        model = RCNN()
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    model_path_east = "check_points_east/model_east_200.ckpt"
    path = "data/test_data/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ReceiptDataLoaderEval(path, transform=transform, type="test")
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    # check_points = [40, 45 , 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125,
    check_points = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125,130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190]#, 195, 200, 205, 210, 215, 220, 225, 230, 245,
     # 250, 255, 260, 265, 270]
    # check_points = [245]

    tes_true, tes_pred = eval_tesseract(data)
    score, levenstein = get_number_correct_words(tes_true, tes_pred)


    print("Score: ", np.mean(score[score != 0]))
    print("Levestein similarity: ", np.mean(levenstein[levenstein != 0]))
    print("Number of zeros: ", score[score == 0].shape)

    with open("check_points_final/eval_tesseract.txt", "a+") as f:
        f.write(f"{np.mean(score)}, {np.mean(levenstein)}")

    """"
    for c in tqdm(check_points):
        model = RCNN()
        model_path_dense = f"check_points_for_loss/rcnn_64/model_rcnn_64_{c}.ckpt"
        # model = load_model_task2(model_name="DenseNet ", model_path=model_path_dense)
        model.load_state_dict(torch.load(model_path_dense))
        all_true_text, all_pred_text = eval_rocognition_model(model, data)

        score, levenstein = get_number_correct_words(all_true_text, all_pred_text)

        print("Score: ", np.mean(score))
        print("Levestein similarity: ", np.mean(levenstein))

        with open("check_points_for_loss/rcnn_64/eval.txt", "a+") as f:
            f.write(f"{check_points}, {np.mean(score)}, {np.mean(levenstein)}\n")
"""