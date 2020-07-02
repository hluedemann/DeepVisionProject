import numpy as np
from east_net import *
import torch
import matplotlib.pyplot as plt

def plot_predicted_score(score):

    plt.imshow(score)
    plt.show()




if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data = ReceiptDataLoader("data/task1_2_test", transform, "test")
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    east = EAST()
    east.load_state_dict(torch.load("check_points/model_geo_loss_10.ckpt"))

    example = Image.open(data.images_names[0])
    example = example.resize((512, 512))
    new_x = np.around(int(0.25 * example.size[0]))
    new_y = np.around(int(0.25 * example.size[1]))
    example = example.resize((new_x, new_y))

    for step, data in enumerate(data_loader):


        image, scale = data

        score, geo = east.forward(image)

        print(score.shape)
        print(geo.shape)
        restored_bboxes = get_bounding_boxes_from_output(score[0][0].detach().numpy(), geo[0].permute(1, 2, 0).detach().numpy())

        #restored_boxes_scales = scale_bounding_box(restored_bboxes[:, :8], 4 / scale.detach().numpy())
        #restored_boxes_scales = restored_boxes_scales.reshape(-1, 8)
        add_bounding_box(example, restored_bboxes[:, :8], "red")
        plot_image(example, "resized_boxes")
        example.show()

        plot_predicted_score(score[0][0].detach().numpy())

        if step == 0:
            break