import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn.functional as F
import requests
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torchvision.transforms as T
import json, os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import lpips
from torchvision import transforms
from utils import PreprocessTransform

adv_image_root = '/home/zhangjianwei/photoguard/output/'
adv_image_dirname = 'lego-minifigure-faces/maxDis20Eps0.08tp'
adv_image_dir = os.path.join(adv_image_root, adv_image_dirname)
out_image_root = '/home/zhangjianwei/photoguard/decoded'
out_image_dir = os.path.join(out_image_root, adv_image_dirname)
os.makedirs(out_image_dir, exist_ok=True)
model_id_or_path = "/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(
            model_id_or_path, subfolder="vae"
        )

def load_image(path):
    img = Image.open(path)
    img = img.resize((512, 512)).convert('RGB')
    img = np.array(img).astype(np.float32)/ 255.0
    img = torch.from_numpy(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0))
    return img 

for image_name in os.listdir(adv_image_dir):
    if not image_name.endswith('.png'):
        continue
    adv_image_path = os.path.join(adv_image_dir, image_name)
    image_index = image_name.split('.')[0]
    input_image = Image.open(adv_image_path)
    input_image = input_image.resize((512, 512)).convert('RGB')
    input_image = PreprocessTransform()(input_image)
    latent_representation = vae.encode(input_image).latent_dist.mean
    reconstructed_image = vae.decode(latent_representation).sample
    reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)
    #reconstructed_image = (reconstructed_image.clamp(0, 1) * 255).byte() 
    reconstructed_image = T.ToPILImage()(reconstructed_image[0]).convert("RGB")
    out_image_path = os.path.join(out_image_dir, image_name)
    reconstructed_image.save(out_image_path)
    print(f'{image_name} decode complete')

'''''
#input_image = load_image("/home/zhangjianwei/photoguard/init_images/mix-27/2.png")
input_image = Image.open("/home/zhangjianwei/photoguard/init_images/mix-27/2.png")
input_image = input_image.resize((512, 512)).convert('RGB')
input_image = PreprocessTransform()(input_image)


latent_representation = vae.encode(input_image).latent_dist.mean

reconstructed_image = vae.decode(latent_representation).sample
reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)
#reconstructed_image = (reconstructed_image.clamp(0, 1) * 255).byte() 
reconstructed_image = T.ToPILImage()(reconstructed_image[0]).convert("RGB")

reconstructed_image.save("/home/zhangjianwei/photoguard/eliminated/test/2.png")
'''''