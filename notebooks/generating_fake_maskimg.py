
import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

import torchvision.transforms as T

from utils import preprocess, prepare_mask_and_masked_image, recover_image

to_pil = T.ToPILImage()

# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../stable-diffusion-v1-5/",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")

init_image = Image.open(f'../init_images/stable-diffusion-v1-5/10ScimP01aprNoisent01E/mix-27/19_adv-photoguard-10ScimP01aprNoisent01E.png').convert('RGB').resize((512,512))

# prompt = "man riding a motorcycle at night"
# prompt = "two men in a wedding"
# prompt = "two men in a restaurant hugging"
# prompt = "two men in a classroom"
# prompt = "two men in a library"
prompt = "one girl sittingl in a black forest"


# A good seed
SEED = -1

# Uncomment the below to generated other images
# SEED = np.random.randint(low=0, high=100000)

torch.manual_seed(SEED)
print(SEED)

strength = 0.4
guidance_scale = 7
num_inference_steps = 100

image_nat = pipe_inpaint(prompt=prompt, 
                     image=init_image, 
                     eta=1,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale,
                     strength=strength,
                    ).images[0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

ax[0].imshow(init_image)
ax[1].imshow(image_nat)

ax[0].set_title('Source Image', fontsize=16)
ax[1].set_title('Fake Image.', fontsize=16)

for i in range(2):
    ax[i].grid(False)
    ax[i].axis('off')
    
fig.suptitle(f"Prompt: {prompt} | Seed: {SEED}", fontsize=20)
fig.tight_layout()
plt.show()