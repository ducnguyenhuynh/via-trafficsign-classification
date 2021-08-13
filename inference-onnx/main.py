import cv2
import torch
from PIL import Image
from torchvision import transforms

image = Image.open("dataset/test/0/00059.jpg")
image = transforms.Resize(128)(image)
image = transforms.ToTensor()(image)
image = transforms.Normalize(mean= [0.4151, 0.3771, 0.4568], std = [0.2011, 0.2108, 0.1896])(image)


print(image.shape)
print(image[0][0][:20])