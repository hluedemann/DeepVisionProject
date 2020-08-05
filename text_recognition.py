import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from utils.data_processing import get_brectangle_from_bbox, parse_annotation
from utils.data_loader_text_recognition import resize_image_with_aspect_ratio, decode_ctc_output
from models.text_recognition_net import load_text_recognition_model
from eval_text_recognition import get_prediction_metrics


def text_recognition_gt_bboxes(model, image_path, annotation_path):
    """ Recognize the text from the ground truth bounding boxes.

    This function also creates images for every bounding box with the true text as caption
    and saves them to "output/pred_texts".

    :param model: Trained model.
    :param image_path: Path to the image.
    :param annotation_path: Path to the annotation.
    :return: True and predicted texts.
    """

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

        image_tensor = transform(image_crop)
        image_tensor = image_tensor.unsqueeze(0)
        log_preds = model.forward(image_tensor).permute(1, 0, 2)
        prediction = decode_ctc_output(log_preds)
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


if __name__ == "__main__":
    model_path_east = "check_points_final/model_east_395.ckpt"
    model_path_recognition = "check_points_final/model_rcnn_64_190.ckpt"

    image_path = "data/test_data/X51005605287.jpg"
    annotation_path = "data/test_data/X51005605287.txt"

    model = load_text_recognition_model(model_name="CRNN", model_weights=model_path_recognition, out_put_size=64)
    true_text, pred_text = text_recognition_gt_bboxes(model, image_path, annotation_path)

    acc, lev = get_prediction_metrics([true_text], [pred_text])

    print("\nWord Accuracy: ", acc)
    print("Character Score: ", lev)
    print("\n")

    print(true_text)
    print(pred_text)
