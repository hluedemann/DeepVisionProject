import numpy as np
from east_net import *
import torch
import matplotlib.pyplot as plt


def plot_predicted_score(score):
    plt.imshow(score)
    plt.show()

def load_model(weight_path):
  east = EAST()
  east.load_state_dict(torch.load(weight_path))
  return east

def text_detection(model, image_path):
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
  score, geo = east.forward(image_tensor)
  restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(), geo[0].permute(1, 2, 0).detach().numpy())
  add_bounding_box(image_output, restored_bboxes[:,:8], "red")
  plot_image(image_output, "prediction")

  restored_bboxes_scaled = scale_bounding_box(restored_bboxes[:,:8], scale)
  restored_bboxes_scaled = restored_bboxes_scaled.reshape(-1, 8)
  add_bounding_box(image, restored_bboxes_scaled[:,:8], "red")
  plot_image(image, "prediction_original_size")


if __name__ == "__main__":

    east = load_model("check_points_colab/model_geo_loss_99.ckpt")
    image_path = "data/task1_2_test/X00016469670.jpg"
    text_detection(east, image_path)