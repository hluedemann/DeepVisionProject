
from training_dense_net import *
from dense_net import *


def load_model_task2(model_path):
  densenet = DenseNet()
  densenet.load_state_dict(torch.load(model_path))
  return densenet

def text_recognition(model, image_path, annotation_path):
  transform = transforms.Compose([transforms.ToTensor()])

  boxes, texts = parse_annotation(annotation_path)

  image = Image.open(image_path).convert("L")

  new_size = (256, 32)

  for (index, box) in enumerate(boxes):
    brectangle = get_brectangle_from_bbox(box)
    image_crop = image.crop(brectangle)
    image_crop = resize_image_with_aspect_ratio(image_crop, new_size)
    image_crop = add_padding_to_image(image_crop, new_size)

    image_tensor = transform(image_crop)
    image_tensor = image_tensor.unsqueeze(0)
    log_preds = model.forward(image_tensor).permute(1, 0, 2)
    print(f"shape: {log_preds.shape}")
    print(f"indices: {torch.argmax(log_preds[:,0,:], dim=1)}")
    prediction = encode_ctc(log_preds)
    print(f"prediction: {prediction}")
    print(f"text: {texts[index]}")

densenet = load_model_task2("model_task2_17.ckpt")
image_path = "data/train_data/X00016469612.jpg"
annotation_path = "data/train_data/X00016469612.txt"
text_recognition(densenet, image_path, annotation_path)