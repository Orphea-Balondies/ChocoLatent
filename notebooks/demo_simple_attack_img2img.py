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
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLImg2ImgPipeline
import torchvision.transforms as T
from skimage.metrics import structural_similarity
from utils import preprocess, recover_image
import argparse
import json

import torch.distributed as dist
import torch.multiprocessing as mp


to_pil = T.ToPILImage()
# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../demo/images/', help='The main directory path of images')
    parser.add_argument('--image_list', nargs='+', help='Images\' name, could be a string or list. If infer')
    parser.add_argument('--SCIMvalues', nargs='*', type=int, default=[10])
    parser.add_argument('--eps', nargs='*', type=float, default=[0.1])
    parser.add_argument('--model_path', type=str, default="/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5")
    parser.add_argument('--out_dir', type=str, default=None, help='select the output directory')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'protect', 'infer'])
    parser.add_argument('--protecttype', type=str, default='SCIM', choices=['SCIM', 'original'])
    parser.add_argument('--device', default='cuda')    
    parser.add_argument('--rewrite',type=bool, default=True)    

    
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    image_dir = args.dir
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(args.dir,'protected')
    
    
    print(f'image_list:{args.image_list[0]}')
    try:
        # 尝试将参数作为 JSON 列表解析    
        image_list = json.loads(args.image_list[0]) 
    except Exception as e:
        # 如果解析失败，则认为参数是多个字符串
        image_list = [{"image_name":image} for image in args.image_list]
        raise e
    
    
    model_id_or_path = args.model_path      
    #model_id_or_path = "../stable-diffusion-inpainting"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
    '''
    image_list = [
    {"image_name": "dog-new-2.jpg","SEED":9955,"prompt":"dog under heavy rain and muddy ground real","strength":0.5,"guidance_scale":7.5}]
    ,
    {"image_name": "two-cats-lying.png","prompt":"only an orange cat lying on steps real,no brown cat,remove brown cat","SEED":2828,"strength":0.5,"guidance_scale":9.5},
    {"image_name": "mountains-in-distance.jpg","prompt":"only one tower on the mountain top in a distance,tower at the middle of the picture,tower at the moutain top,tower at the high space of the picture","SEED":6251,"strength":0.55,"guidance_scale":9.5}]
    process_item=[1,2]
    '''
    print(f'args.device:{args.device}')
    global device
    device= torch.device(args.device)
    mode=args.mode
    ptype=args.protecttype
    rewrite=args.rewrite
    
    if model_id_or_path.split('-')[-1]=='inpainting':
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        print('modeltype:inpainting')
        pipe = pipe.to(device)
    elif "xl" in model_id_or_path.split('-'):
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id_or_path,
        )
        pipe = pipe.to(device)

        print('modeltype:XL')
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path,
            revision="fp16", 
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(device)
        print('modeltype:V1-5')


    for image_dict in image_list:
        try:
            image_name = image_dict["image_name"]
        except Exception as e:
            print(f'image_dict:{image_dict}')
            raise e
        print(f'begin to protect {image_name}')
        image_path = os.path.join(image_dir, image_name)
        init_image = Image.open(image_path).convert("RGB")
        resize = T.transforms.Resize(512)
        center_crop = T.transforms.CenterCrop(512)
        init_image = center_crop(resize(init_image))
        
        if mode in ['all','protect']:
            if args.protecttype == 'SCIM':
                print(f'using SCIM protection, SCIMvalues:{args.SCIMvalues}')
                for SCIMvalue in args.SCIMvalues:
                    
                    print(f'use strength of {SCIMvalue} SCIM')
                    
                    eps = args.eps[0]
                    version = f"{SCIMvalue}ScimP01aprNoisent{str(eps).replace('.','')}E"
                    
                    singl_adv_process(version,output_dir,image_name,mode,pipe,init_image,ptype,rewrite,SCIMvalue,eps,image_dict)
                    print(f"{SCIMvalue} SCIM of {image_name} protect finished")    
            if args.protecttype == 'original':
                print(f'using original protection, epsvalues:{args.eps}')
                for eps in args.eps:
                    
                    print(f'use limitation of {eps} eps')
                    
                    SCIMvalue= args.SCIMvalues[0]
                    version = f"original{str(eps).replace('.','')}E"
                    
                    singl_adv_process(version,output_dir,image_name,mode,pipe,init_image,ptype,rewrite,SCIMvalue,eps,image_dict)         
                    print(f"{eps} eps of {image_name} original protect finished") 
                    
                    
        if args.mode in ['infer','all']:    
            prompt = image_dict.get("prompt")
            strength = image_dict.get("strength")
            guidance_scale = image_dict.get("guidance_scale")
            SEED = image_dict.get("SEED")
            image_gened_path = f"{image_path[:image_path.rfind('.')]}_gened.{image_path.split('.')[-1]}"
            image_nat = infer(pipe,adv_image, prompt, strength, guidance_scale, SEED)
        
def singl_adv_process(version,output_dir,image_name,mode,pipe,init_image,ptype,rewrite,SCIMvalue,eps,image_dict):
    image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-photoguard-{version}.png"
    image_adv_path=os.path.join(output_dir, version, image_adv_name)   
    image_adv_dir = os.path.dirname(image_adv_path)
    if not os.path.exists(image_adv_dir):
      os.makedirs(image_adv_dir)
    
    if mode=='protect':    
      if not os.path.exists(image_adv_path) or rewrite:
          adv_image = adv(pipe,image_adv_path, init_image,ptype, SCIMvalue,eps)
    
    elif mode=='all':
      if not os.path.exists(image_adv_path) or rewrite:
          adv_image = adv(pipe,image_adv_path, init_image,ptype, SCIMvalue,eps)
          print('create photoguard adv image to generate')
      else:
          adv_image = Image.open(image_adv_path).convert("RGB")
          print('using existing photoguard adv image to generate')
      prompt = image_dict.get("prompt")
      strength = image_dict.get("strength")
      guidance_scale = image_dict.get("guidance_scale")
      SEED = image_dict.get("SEED")
      image_adv_gened_path = f"{image_adv_path[:image_adv_path.rfind('.')]}_gened.{image_adv_path.split('.')[-1]}"
      image_adv_gened = infer(pipe,adv_image, image_adv_gened_path, prompt, strength, guidance_scale, SEED)
    return
  


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
    
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None, SCIMvalue=10):
    print(f'X.shape:{X.shape}')
    global device

    #if type(model).__name__ == "StableDiffusionXLImg2ImgPipeline":
    #X=X.to(torch.float32)
    #model.to(torch.float32)
    model=model.encode
    X_adv = X.clone().detach().to(device) + (torch.rand(*X.shape)*2*eps-eps).to(device)
    X_adv.requires_grad_(True)
    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)
    pbar = tqdm(range(iters))
    
    print(f'latentmax:{model(X).latent_dist.mean.max()} latentmin:{model(X).latent_dist.mean.min()}')
    
    similarity = S_CON_tr_simularity(X, X_adv)
    epst = (model(X).latent_dist.mean.max() - model(X).latent_dist.mean.min())/30
    target_tensor = (torch.rand_like(model(X).latent_dist.mean) * epst - epst / 2) +model(X).latent_dist.mean.max()+model(X).latent_dist.mean.min()-model(X).latent_dist.mean
    
    gap2black = (model(X_adv).latent_dist.mean).norm()
    gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
    gap = gap2black
    appro = (X-X_adv).norm()
    #ratio =(10*gap/similarity).detach().item()
    ratio =(SCIMvalue*gap*similarity).detach().item()
    ratio2  = (0.1*gap/appro).detach().item()
    
    print(f"ratio:{ratio}")
    print(f'similarity:{similarity}')
    print(f"gap:{gap}")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        similarity = S_CON_tr_simularity(X, X_adv)
        gap2black = (model(X_adv).latent_dist.mean).norm()
        gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
        gap = gap2black
        appro = (X-X_adv).norm()
        loss = gap+ratio/similarity+ratio2*appro
        #loss = gap+ratio/similarity
        
        if i % (len(pbar)//4) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}", flush=True)
            pbar.update(len(pbar)//4)

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
        #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        if mask is not None:
            X_adv.data *= mask
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return X_adv

def pgd_original(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    global device
    X_adv = X.clone().detach().to(device) + (torch.rand(*X.shape)*2*eps-eps).to(device)
    pbar = tqdm(range(iters))
    epst = (model(X).latent_dist.mean.max() - model(X).latent_dist.mean.min())/30
    target_tensor = (torch.rand_like(model(X).latent_dist.mean) * epst - epst / 2) +model(X).latent_dist.mean.max()+model(X).latent_dist.mean.min()-model(X).latent_dist.mean
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean-target_tensor).norm()

        if i % (len(pbar)//4) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}", flush=True)
            pbar.update(len(pbar)//4)

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv    
    
    
# You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
def adv(pipe,image_adv_path,init_image,ptype,SCIMvalue,eps):
    global device
    
    X = preprocess(init_image).to(device)
    if type(pipe).__name__ != "StableDiffusionXLImg2ImgPipeline":
        X = X.half()
    if ptype=='SCIM':
        adv_X = pgd(X, 
                    model=pipe.vae, 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=eps, # The higher, the less imperceptible the attack is 
                    step_size=0.1, # Set smaller than eps
                    iters=1000, # The higher, the stronger your attack will be
                    SCIMvalue=SCIMvalue,
                    )
    elif ptype=='original':
        adv_X = pgd_original(X, 
                    model=pipe.vae.encode, 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=eps, # The higher, the less imperceptible the attack is 
                    step_size=eps/2, # Set smaller than eps
                    iters=1000, # The higher, the stronger your attack will be
                    )
    # convert pixels back to [0,1] range
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
          
    adv_image = to_pil(adv_X[0]).convert("RGB")
    adv_image.save(image_adv_path)
    
    
    
    
    return adv_image
      
    
def infer(pipe,image, image_path, prompt, strength, guidance_scale, SEED=None):    
    global device 
    negative_prompt = "NSFW"
    # a good seed (uncomment the line below to generate new images)
    if not SEED:
      SEED = np.random.randint(low=0, high=10000)
    print(f"seed:{SEED}")
    # Play with these for improving generated image quality

    num_inference_steps = 50
      
    with torch.autocast('cuda'):
      torch.manual_seed(SEED)  
      image_nat = pipe(prompt=prompt, 
                         image=image, 
                         #mask_image=mask_image, 
                         #eta=1,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         strength=strength,
                        ).images[0]
      
    image_nat.save(image_path)
    return image_nat

if __name__ == '__main__':
    main()