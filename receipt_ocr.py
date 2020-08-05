from utils.score_functions_east import *
from training_dense_net import *
from text_detection import load_east_model, text_detection


def get_predicted_bboxes(model, image_path):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    image_resized = image.resize((512, 512))
    image_output = image.resize((128, 128))

    scale = np.zeros(2)
    scale[0] = image.size[0] / image_output.size[0]
    scale[1] = image.size[1] / image_output.size[1]

    image_tensor = transform(image_resized)
    image_tensor = image_tensor.unsqueeze(0)
    score, geo = model.forward(image_tensor)
    restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(),
                                                     geo[0].permute(1, 2, 0).detach().numpy())

    restored_bboxes_scaled = scale_bounding_box(restored_bboxes[:, :8], scale).reshape(-1, 8)

    return restored_bboxes_scaled


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

    i = 0

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

        plt.imshow(image_crop)
        plt.title(prediction, fontsize=15)
        plt.axis("off")
        plt.savefig(f"output/pred_text/{i}.png")
        plt.show()

        print(f"prediction: {prediction}")
        print(f"text: {texts[index]}")
        i += 1
    return texts, pred_text


def get_number_correct_words(true_text, pred_text):

    if len(true_text) != len(pred_text):
        print("Note the same number of true and predicted texts")

    score = 0

    for i in range(len(true_text)):
        if true_text[i] == pred_text[i]:
            score += 1

    return score / len(true_text)

def load_model_task2(model_name="RCNN", model_path=None):
    if model_name == "DenseNet":
        model = DenseNet()
    elif model_name == "RCNN":
        model = RCNN()
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    model_path_east = "check_points_final/model_east_395.ckpt"
    model_path_dense = "check_points_final/task2_RCNN_397.ckpt"
    model_path_crnn = "check_points_final/model_rcnn_64_190.ckpt"
    # image_path = "data/test_data/X51005663307.jpg"
    # annotation_path = "data/test_data/X51005663307.txt"
    image_path = "data/test_data/X51005605287.jpg"
    annotation_path = "data/test_data/X51005605287.txt"
    east = load_east_model(model_path_east)

    pred_bboxes = text_detection(east, image_path)

    # pred_bboxes = get_predicted_bboxes(model, image_path)     ## Get bounding boxes without plot
    # croped_images = crop_bbox_images(pred_bboxes, image_path)

    model = load_model_task2(model_name="RCNN", model_path=model_path_crnn)
    true_text, pred_text = text_recognition(model, image_path, annotation_path)

    score = get_number_correct_words(true_text, pred_text)

    print("Score: ", score)

    print(true_text)
    print(pred_text)
