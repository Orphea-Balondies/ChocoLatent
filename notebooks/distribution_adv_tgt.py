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
from skimage.metrics import structural_similarity
from utils import preprocess, recover_image
from dataset import CLDataset
import argparse
import json, os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import lpips as LPIPS
from torchvision import transforms
from utils import PreprocessTransform
from decimal import Decimal
to_pil = T.ToPILImage()
init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])

def setup(rank, world_size, nproc_per_gpu):
    # 初始化分布式环境
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    gpu_id = rank // nproc_per_gpu
    torch.cuda.set_device(gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../demo/images/', help='The main directory path of images')
    parser.add_argument('--image_list', nargs='+', help='Images\' name, could be a string or list.')
    parser.add_argument('--maxDis', nargs='*', type=int, default=[50])
    parser.add_argument('--eps', nargs='*', type=float, default=[0.1])
    parser.add_argument('--model_path', type=str, default="/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5")
    parser.add_argument('--out_dir', type=str, default=None, help='select the output directory')
    parser.add_argument('--adv_mode', type=str, default='tgt', choices=['tgt', 'ntgt'])
    parser.add_argument('--tgt_path', type=str, default="latent_output/MIST.png", help='path of target image')
    parser.add_argument('--tgttype', type=str, default='latent', choices=['latent', 'pixiel'])
    parser.add_argument('--rewrite',type=bool, default=True)
    parser.add_argument('--nproc_per_gpu',type=int, default=1)    
    parser.add_argument('--img2img', default='none', choices=['none', 'all', 'only_o'], help='whether to generate img2img result, choose only_o to skip attack steps') 
    args = parser.parse_args()
    return args

def main(rank, world_size):

    args = parse_args()
    nproc_per_gpu = args.nproc_per_gpu

    gpu_id = rank // nproc_per_gpu
    setup(rank, world_size, nproc_per_gpu)
    # 设置设备
    global device
    device = torch.device(f"cuda:{gpu_id}")

    image_dir = args.image_dir
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(args.dir,'protected')
    
    print(f'image_list:{args.image_list[0]}')
    try:
        image_list = json.loads(args.image_list[0]) 
    except Exception as e:
        print(args.image_list)
        image_list = [{"image_name":image} for image in args.image_list]
        raise e
    
    model_id_or_path = args.model_path      
    img2img_option = args.img2img

    if img2img_option == 'none':
        pipe = AutoencoderKL.from_pretrained(
                model_id_or_path, subfolder="vae"
            )
        pipe = pipe.to("cuda")
    else:
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id_or_path,
                revision="fp16", 
                torch_dtype=torch.float16,
            )
        model = model.to("cuda")
        pipe = model.vae

    dataset = CLDataset(image_dir,image_list)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)  

    for init_images, image_args in dataloader :  
        image_name = image_args["image_name"][0] 
        init_image = init_images[0]
        image_arg = image_args
        if img2img_option in ['all','none']:
            adv_mode=args.adv_mode
            if adv_mode=='tgt':
                tgt_path=args.tgt_path
                tgttype=args.tgttype
                adv_ver='t'+tgttype[0]
            else: adv_ver='n'
            print(f'using original protection, epsvalues:{args.eps}')
            for eps in args.eps:
                for maxDis in args.maxDis:
                    print(f'use limitation of {eps} eps; {maxDis} maxDis')
                    version = f"maxDis{maxDis}Eps{eps}{adv_ver}"
                    image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-{version}.png"
                    image_adv_path = os.path.join(output_dir, version, image_adv_name)
                    os.makedirs(os.path.dirname(image_adv_path), exist_ok=True)
                    rewrite=args.rewrite
                    if not os.path.exists(image_adv_path) or rewrite:
                        if img2img_option=='all':
                            print('create photoguard adv image to generate')
                        adv_image = adv(pipe,image_adv_path, init_image, adv_mode, eps, maxDis, tgttype, tgt_path)
                        print(f"{eps} eps of {image_name} adv protect finished")
                    else:
                        if img2img_option=='all':
                            adv_image = None
                            print('using existing photoguard adv image to generate')
                        print(f"{eps} eps of {image_name} adv is existed, skip")
                
                    
        if img2img_option in ['only_o','all']:

            prompt = image_arg.get("prompt")
            strength = image_arg.get("strength")
            guidance_scale = image_arg.get("guidance_scale")
            SEED = image_arg.get("SEED")
            image_gened_path = f"{image_path[:image_path.rfind('.')]}_gened.{image_path.split('.')[-1]}"
            image_nat = infer(pipe,adv_image, prompt, strength, guidance_scale, SEED)
            if args.mode == 'all':
                if adv_image == None:
                    adv_image = Image.open(image_adv_path).convert("RGB")
                image_adv_gened_path = f"{image_adv_path[:image_adv_path.rfind('.')]}_gened.{image_adv_path.split('.')[-1]}"
                image_adv_gened = infer(pipe,adv_image, image_adv_gened_path, prompt, strength, guidance_scale, SEED)
        
def adv(pipe,image_adv_path,init_image, adv_mode, eps, maxDis, tgttype=None, tgt_path=None):
    X = init_image.cuda()
    adv_X = pgd(X, 
                model=pipe, 
                adv_mode = adv_mode,
                tgttype = tgttype,
                tgt_path = tgt_path,
                maxDis=maxDis,
                clamp_min=-1,
                clamp_max=1,
                eps=eps, # The higher, the less imperceptible the attack is 
                step_size=5, # Set smaller than eps
                iters=5000, # The higher, the stronger your attack will be
                )
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
          
    adv_image = to_pil(adv_X[0]).convert("RGB")
    adv_image.save(image_adv_path)
    print(f'{image_adv_path} saved')
    return adv_image
      
def normalization(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return 2*normalized_tensor-1

def list_to_scientific_notation(numbers):
    # 将数字转换为 Decimal 类型
    for i in range(len(numbers)):
      dec = Decimal(str(numbers[i])).normalize()
      # 去除小数点后不必要的零
      # 使用科学计数法保留有效数字并输出
      numbers[i] = f"{dec:.3e}"
    return numbers 
   
def pgd(X, model,  adv_mode, tgttype, tgt_path=None, maxDis=50, eps=0.1, step_size=0.2, iters=200, clamp_min=-1, clamp_max=1):
    print(f'X.shape:{X.shape},X.dtype:{X.dtype},maxDis{maxDis},step_size{step_size},iters{iters}')
    X_adv = X.clone().detach().cuda() + (torch.rand(*X.shape, device=X.device)*2*eps-eps)
    X_adv.requires_grad_(True)
    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)
    pbar = tqdm(range(iters))
    
    def loss_tl(model,Dis,X,X_adv,tgt):
        latent_X_adv = model.encode(X_adv).latent_dist.mean
        latent_Dis = (latent_X_adv-tgt).norm()
        loss = 400*Dis + latent_Dis
        return loss, latent_Dis
    def loss_tp(model,Dis,X,X_adv,tgt):
        latent_X_adv = model.encode(X_adv).latent_dist.mean
        decoded_X_adv = model.decode(latent_X_adv).sample
        decoded_Dis = (decoded_X_adv-tgt).norm()
        loss = 20*Dis + decoded_Dis
        return loss, decoded_Dis
    def loss_n(model,Dis,X,X_adv,tgt=None):
        latent_X_adv = model.encode(X_adv).latent_dist.mean
        decoded_X_adv = model.decode(latent_X_adv).sample
        decoded_Dis = (X-decoded_X_adv).norm()
        loss = 20*Dis - decoded_Dis
        return loss, decoded_Dis
    
    adv_Dis_name='decoded_Dis'
    if adv_mode=='tgt':
        tgt_image = Image.open(tgt_path).convert("RGB")  # 读取图片
        tgt =  init_transform(tgt_image).cuda() # 应用数据变换
        if tgttype=='latent':
            tgt = model.encode(tgt).latent_dist.mean
            loss_fn = loss_tl
            adv_Dis_name='latent_Dis'
        else:
            loss_fn = loss_tp
    else:
        loss_fn = loss_n
        
    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
    lpips = loss_fn_alex(X, X_adv)

    print(f'lpips:{lpips}')

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        lpips = loss_fn_alex(X, X_adv)
        Dis = torch.maximum((X-X_adv).norm(), torch.tensor(maxDis))
        loss, adv_Dis = loss_fn(model,Dis,X,X_adv,tgt)
        adv_Dis_nott, Dis_nott = list_to_scientific_notation([adv_Dis.item(), Dis.item()])
        if i % (len(pbar)//4) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4} | Dis: {Dis_nott} | {adv_Dis_name}: {adv_Dis_nott}", flush=True)
            #print(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}", flush=True)
            pbar.update(len(pbar)//4)
        grad, = torch.autograd.grad(loss, [X_adv])
        grad_flattened = grad.view(grad.size(0), grad.size(1), -1)
        normalized_grad = F.normalize(grad_flattened, p=2, dim=-1)
        normalized_grad = normalized_grad.view_as(grad)
        #print(f"normalized_grad:{normalized_grad}")
        X_adv = X_adv - normalized_grad.detach() * actual_step_size
        X_adv.data = torch.clamp(X_adv, min=X - eps, max=X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return X_adv   
    
def infer(pipe,image, image_path, prompt, strength, guidance_scale, SEED=None):    

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
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank,world_size)