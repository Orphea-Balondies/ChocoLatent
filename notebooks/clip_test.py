from PIL import Image
import requests

from transformers import CLIPImageProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("/root/chocolatent/model/clip-vit-base-patch32")
processor = CLIPImageProcessor.from_pretrained("/root/chocolatent/model/clip-vit-base-patch32")

image_path = "/root/chocolatent/init_images/lego-minifigure-faces/0001.jpg"
image = Image.open(image_path)

inputs = processor(images=image, return_tensors="pt", padding=True)

outputs = model(inputs["pixel_values"])
logits_per_image = outputs.pooler_output  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
