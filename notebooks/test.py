from diffusers import AutoPipelineForImage2Image,StableDiffusionXLImg2ImgPipeline,DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import torchvision.transforms as T
import torch
from utils import preprocess
import os
# use from_pipe to avoid consuming additional memory when loading a checkpoint
model_id_or_path="stable-diffusion-xl-base-1.0"

pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id_or_path,
            use_safetensors=True,
        )
pipe = pipe.to('cuda')
'''''
vae = AutoencoderKL.from_pretrained(
            os.path.join(model_id_or_path,"vae"),
            )
vae=vae.to('cuda')
'''''
image_path = "init_images/typeA-10/9.png"
init_image = Image.open(image_path).convert("RGB")
resize = T.transforms.Resize(512)
center_crop = T.transforms.CenterCrop(512)
init_image = center_crop(resize(init_image))
X = preprocess(init_image).to('cuda')
test_X = torch.rand(*X.shape).to('cuda')

'''''
prompt = "A majestic lion jumping from a big stone at night"
image = pipe(
    prompt=prompt,
    output_type="latent",
    image=test_X,
).images
'''''
print(pipe.vae.encode)
print(pipe.vae)
image = pipe.vae.encode(test_X)
image = image.latent_dist.mean
print(image)