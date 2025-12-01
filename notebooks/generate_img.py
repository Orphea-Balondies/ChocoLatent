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
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torchvision.transforms as T
from skimage.metrics import structural_similarity
from utils import preprocess, recover_image
to_pil = T.ToPILImage()
# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
image_dir = "../init_images/test/"
image_list = [
{"image_name": "0019.jpg","SEED":782732,"prompt":"a white face man with brown hair","strength":0.2,"guidance_scale":7.5},
{"image_name": "0019_adv-maxDis20Eps0.08tl.png","SEED":782732,"prompt":"a white face man with brown hair","strength":0.2,"guidance_scale":7.5},
{"image_name": "0008_adv-photoguard-min20eps0.12Black.png","SEED":782732558,"prompt":"a white face man with hat","strength":0.2,"guidance_scale":7.5}]

process_item=[0,1]
version = "05ScimNoisenTuL"
eps=0.03
#image_adv_path = None
image_adv_path = f"{image_dir}dog-new-2_mist_4_100_512_2_0_1_0_1.png"

def normalization(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return 2*normalized_tensor-1

def S_CON_tr_simularity(image1, image2):
    mean1 = image1.mean(dim=[0, 2, 3], keepdim=True)
    mean2 = image2.mean(dim=[0, 2, 3], keepdim=True)

    variance1 = ((image1 - mean1) ** 2).mean(dim=[0, 2, 3]).norm()
    variance2 = ((image2 - mean2) ** 2).mean(dim=[0, 2, 3]).norm()
    
    covariance = ((image1 - mean1) * (image2 - mean2)).mean(dim=[0, 2, 3]).norm()
    
    SCIM = (covariance + 2*0.01)/(variance1 + variance2 + 2*0.01)
    return SCIM
    
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    X_adv.requires_grad_(True)
    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)
    pbar = tqdm(range(iters))
    print(f'latentmax:{model(X).latent_dist.mean.max()} latentmin:{model(X).latent_dist.mean.min()}')
    #sys.exit(0)
    
    similarity = S_CON_tr_simularity(X, X_adv)
    
    epst = (model(X).latent_dist.mean.max() - model(X).latent_dist.mean.min())/5
    target_tensor = (torch.rand_like(model(X).latent_dist.mean) * epst - epst / 2) +model(X).latent_dist.mean.max()+model(X).latent_dist.mean.min()-model(X).latent_dist.mean
    gap2black = normalization(model(X_adv).latent_dist.mean).norm()
    gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
    gap = gap2noise
    appro = (X-X_adv).norm()
    ratio =(0.5*gap*similarity).detach().item()
    ratio2  = (0.1*gap/appro).detach().item()
    
    print(f"ratio:{ratio}")
    print(f'similarity:{similarity}')
    print(f"gap:{gap}")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        similarity = S_CON_tr_simularity(X, X_adv)
        gap2black = normalization(model(X_adv).latent_dist.mean).norm()
        gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
        gap = gap2black
        appro = (X-X_adv).norm()
        loss = ratio/similarity+gap
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        grad, = torch.autograd.grad(loss, [X_adv])
        '''
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        '''
        grad_flattened = grad.view(grad.size(0), grad.size(1), -1)
        normalized_grad = F.normalize(grad_flattened, p=2, dim=-1)
        normalized_grad = normalized_grad.view_as(grad)
        #print(f"normalized_grad:{normalized_grad}")
        X_adv = X_adv - normalized_grad.detach() * actual_step_size
        
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        if mask is not None:
            X_adv.data *= mask
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return X_adv
    
# You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
def adv(image_adv_path,init_image, prompt, strength, guidance_scale, SEED=None):
    """""
    if not os.path.exists(image_adv_path):
      with torch.autocast('cuda'):
          X = preprocess(init_image).half().cuda()
    
          adv_X = pgd(X, 
                      model=pipe.vae.encode, 
                      clamp_min=-1, 
                      clamp_max=1,
                      eps=eps, # The higher, the less imperceptible the attack is 
                      step_size=0.01, # Set smaller than eps
                      iters=1000, # The higher, the stronger your attack will be
                     )
      
          # convert pixels back to [0,1] range
          adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
          
      adv_image = to_pil(adv_X[0]).convert("RGB")
      adv_image.save(image_adv_path)
      print('create photoguard adv image to generate')
    else:
      adv_image = Image.open(image_adv_path).convert("RGB")
      print('using existing photoguard adv image to generate')
    """""
    negative_prompt = "NSFW"
    # a good seed (uncomment the line below to generate new images)
    if not SEED:
      SEED = np.random.randint(low=0, high=10000)
    print(f"seed:{SEED}")
    # Play with these for improving generated image quality
    strength = 0.5
    guidance_scale = 7.5
    num_inference_steps = 50
      
    with torch.autocast('cuda'):

      torch.manual_seed(SEED)  
      image_nat = pipe(prompt=prompt, 
                        image=init_image, 
                        #mask_image=mask_image, 
                        #eta=1,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        ).images[0]
      torch.manual_seed(SEED)
      image_nat.save(f"{image_dir}{image_name[:image_name.rfind('.')]}_gened.{image_name.split('.')[-1]}")
      """""
      image_adv = pipe(prompt=prompt, 
                         image=adv_image, 
                         #mask_image=mask_image, 
                         #eta=1,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         strength=strength,
                        ).images[0]
    image_adv.save(f"{image_adv_path[:image_adv_path.rfind('.')]}_gened.{image_name.split('.')[-1]}")
    """""
    return image_nat
'''
with torch.autocast('cuda'):
    torch.manual_seed(SEED)
    image_nat = pipe_img2img(prompt=prompt, init_image=init_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
    torch.manual_seed(SEED)
    image_adv = pipe_img2img(prompt=prompt, init_image=adv_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
'''   
# adv_image = recover_image(adv_image, init_image, mask_image, background=True)
for i in process_item:
    image_dict = image_list[i]
    image_name = image_dict["image_name"]
    if not image_adv_path:
      image_adv_path = f"{image_dir}{image_name[:image_name.rfind('.')]}_adv-photoguard-{version}-{eps}.png"
    mask_image = "../../mist/test/transparent_mask.png"
    #model_id_or_path = "../stable-diffusion-inpainting"
    model_id_or_path = "../stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
    if model_id_or_path.split('-')[-1]=='inpainting':
      pipe = StableDiffusionInpaintPipeline.from_pretrained(
          model_id_or_path,
          revision="fp16",
          torch_dtype=torch.float16,
      )
      print('modeltype:inpainting')
      pipe = pipe.to("cuda")
    else:
      pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16", 
        torch_dtype=torch.float16,
      )
      pipe = pipe.to("cuda")
      print('modeltype:i2i')
    init_image = Image.open(image_dir+image_name).convert("RGB")
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_image = center_crop(resize(init_image))
    #mask_image = Image.open(f'{mask_image}').convert('RGB')
    #mask_image = ImageOps.invert(mask_image).resize((512,512))
    image_nat = adv(image_adv_path,init_image, image_dict["prompt"], image_dict["strength"], image_dict["guidance_scale"], image_dict["SEED"])


leng = len(image_list)
fig, ax = plt.subplots(nrows=leng, ncols=4, figsize=(20,6))
for i in range(leng):
    ax[i][0].imshow(init_image)
    ax[i][1].imshow(adv_image)
    ax[i][2].imshow(image_nat)
    ax[i][3].imshow(image_adv)
    
    ax[i][0].set_title('Source Image', fontsize=16)
    ax[i][1].set_title('Adv Image', fontsize=16)
    ax[i][2].set_title('Gen. Image Nat.', fontsize=16)
    ax[i][3].set_title('Gen. Image Adv.', fontsize=16)
    for j in range(4):
        ax[i][j].grid(False)
        ax[i][j].axis('off')
    

fig.tight_layout()
plt.show()