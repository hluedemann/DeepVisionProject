import numpy as np
from PIL import Image
from torchvision import transforms
import pytesseract

from predict_east import load_model
from score_functions_east import *


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

    image = Image.open(image_path)

    croped_images = []

    for box in bboxes:
        rect = get_brectangle_from_bbox(box.reshape(4, 2))

        croped_images.append(image.crop(rect))

    return croped_images

def text_recognition(images):

    texts = []
    for image in images:
        text = pytesseract.image_to_string(image)
        plt.imshow(image)
        plt.title(text)
        plt.show()
        texts.append(text)
    return texts

if __name__ == "__main__":

    model_path = "check_points/model_east_250.ckpt"
    image_path = "testImages/brown.jpg"
    model = load_model(model_path)


    pred_bboxes = get_predicted_bboxes(model, image_path)
    croped_images = crop_bbox_images(pred_bboxes, image_path)

    #texts = text_recognition(croped_images)

