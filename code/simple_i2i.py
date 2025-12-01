from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torchvision.transforms as T
import torch
import os
model_id_or_path = 'model/stable-diffusion-v1-5'
pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
    model_id_or_path, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

init_image = load_image("exp/demo/04lr400e-5fc-01id04il15md01ml/lego-0001.jpg")
i2i_image_dir = "exp/i2itest/"
os.makedirs(i2i_image_dir,exist_ok=True)
i2i_image_name = "lego-0001_adv-g.jpg"
i2i_image_path = os.path.join(i2i_image_dir,i2i_image_name)

prompt = "a black face toy man "
generator = torch.Generator(device="cpu").manual_seed(1000)
image = pipe_img2img(prompt, image=init_image, generator=generator,strength=0.7,guidance_scale=6).images[0]
#i2i_image = T.ToPILImage()(image).convert("RGB")
image.save(i2i_image_path)
