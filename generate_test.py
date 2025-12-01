import os
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torchvision.transforms as T
import random
to_pil = T.ToPILImage()
# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
output_root = "generated/"
Lora_root = "/home/zhangjianwei/stable-diffusion-webui/models/Lora"
model_id_or_path = "stable-diffusion-v1-5"
prompts = {
   'lego':['a white face man','a black face man','a man in front of brown background'],
   'old-photos':['a woman, monochrome, realistic,  blurry photo', 'a girl, monochrome, realistic,  blurry photo', 'a standing man, monochrome, realistic,  blurry photo'],
   'van-gogh':['a field','trees','flowers'],
}

   
pipe = DiffusionPipeline.from_pretrained(
    model_id_or_path,
)
pipe = pipe.to("cuda")

for name, param in pipe.unet.named_parameters():
    print(f"{name}: {param.shape}")
'''''
for lora_name in ['lego','old-photos','van-gogh']:
    for prompt in prompts[lora_name]:
        random_numbers = random.sample(range(0,10000000000), 20)
        for seed in random_numbers:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            for lora_version in ['original','min20eps0.12Black', 'Imin20appro0.08epsEcDcAWAY']:
                output_dir = os.path.join(output_root,f"{lora_name}-{lora_version}",prompt)
                os.makedirs(output_dir,exist_ok=True)
                pipe.load_lora_weights(Lora_root, weight_name=f"{lora_name}-{lora_version}.safetensors", adapter_name=f"{lora_name}-{lora_version}")
                image = pipe(prompt, height=512, width=512, generator=generator).images[0]
                image.save(os.path.join(output_dir,f"{seed}.png"))
                print(f"{output_dir}/{seed} has been saved")
                pipe.disable_lora()
'''''