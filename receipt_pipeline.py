from PIL import Image
from torchvision import transforms
import argparse

from text_detection import text_detection
from utils.data_loader_text_recognition import get_brectangle_from_bbox,\
    resize_image_with_aspect_ratio, decode_ctc_output
from utils.data_processing import convert_vert_list_to_tuple
from models.east import load_east_model
from models.text_recognition_net import load_text_recognition_model


def text_recognition(model, image_path, bboxes):
    """ Predict the text from image with given bounding boxes.

    :param model: Trained model for recognition.
    :param image_path: Path to image.
    :param bboxes: Bounding boxes.
    :return: List of predicted text.
    """

    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("L")

    new_size = (256, 32)
    pred_text = []
    i = 0

    bboxes = convert_vert_list_to_tuple(bboxes)

    for (index, box) in enumerate(bboxes):
        brectangle = get_brectangle_from_bbox(box)
        image_crop = image.crop(brectangle)
        image_crop = resize_image_with_aspect_ratio(image_crop, new_size)

        image_tensor = transform(image_crop)
        image_tensor = image_tensor.unsqueeze(0)
        log_preds = model.forward(image_tensor).permute(1, 0, 2)
        prediction = decode_ctc_output(log_preds)
        pred_text.append(prediction)

        i += 1
    return pred_text


def receipt_pipeline(model_east, model_recognition, image_path):
    """ Extract all the text from a receipt.
    This function combines the text detection and text recognition.

    :param model_east: Trained EAST model for text detection
    :param model_recognition: Trained model for text recognition.
    :param image_path: Path to image.
    :return: List of predicted text.
    """

    pred_bboxes = text_detection(model_east, image_path)
    pred_texts = text_recognition(model_recognition, image_path, pred_bboxes)

    return pred_texts


parser = argparse.ArgumentParser(description="Text extraction form receipts.")
parser.add_argument("image_path", help="Path to image.")
parser.add_argument("-model_east",
                    default="trained_models/model_east.ckpt",
                    help="Trained EAST model for text detection.")
parser.add_argument("-model_recognition",
                    default="trained_models/model_crnn_64.ckpt",
                    help="Trained model for recognition")
parser.add_argument("-type",
                    default="CRNN",
                    help="Type of recognition model. One of DenseNetLinear, DenseNetRNN, CRNN")
parser.add_argument("-output_size",
                    default=64,
                    help="Max length of prediction. Either 64 or 32", type=int)

if __name__ == "__main__":

    args = parser.parse_args()

    image_path = args.image_path
    model_path_east = args.model_east
    model_path_recognition = args.model_recognition
    recognition_type = args.type

    east = load_east_model(weight=model_path_east)
    model = load_text_recognition_model(recognition_type, model_path_recognition, args.output_size)

    text = receipt_pipeline(east, model, image_path)

    # Print the output
    for t in text:
        print(t)
