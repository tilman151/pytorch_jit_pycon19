import torch
import torch.jit as jit
import numpy as np
import PIL.Image
import json

from IPython.display import Image, display


with open('./imagenet_classes.json', mode='rt') as f:
    CLASS_DICT = json.load(f)

IMAGENET_MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
IMAGENET_STD = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)


def predict_imagenet(net, image_file):
    # Load image and resize
    image = PIL.Image.open(image_file).resize((244, 244))
    # Convert to numpy and normalize
    image = np.array(image, dtype=np.float32) / 255.
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    # Convert to PyTorch and make channel first
    image = torch.as_tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    # Predict top class
    logits = net(image)
    class_idx = logits.squeeze(0).argmax().item()
    # Output predictions
    print('It is a %s.' % CLASS_DICT[str(class_idx)])
    display(Image(filename=image_file))


net = jit.load('./model.pth')
predict_imagenet(net, './whats_this.jpg')
print('https://i.ytimg.com/vi/2fb-g_V-UT4/hqdefault.jpg')