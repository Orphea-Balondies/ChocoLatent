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
from dataset import MyAdvDataset
import argparse
import json, os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import lpips as LPIPS
from decimal import Decimal
from torchvision import transforms
from utils import PreprocessTransform

to_pil = T.ToPILImage()
init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])
# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details

def setup(rank, world_size, nproc_per_gpu):
    # 初始化分布式环境
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    gpu_id = rank // nproc_per_gpu
    torch.cuda.set_device(gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../demo/images/', help='The main directory path of images')
    parser.add_argument('--image_list', nargs='+', help='Images\' name, could be a string or list. If infer')
    parser.add_argument('--Lpipsvalues', nargs='*', type=float, default=[1])
    parser.add_argument('--eps', nargs='*', type=float, default=[0.1])
    parser.add_argument('--model_path', type=str, default="/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5")
    parser.add_argument('--out_dir', type=str, default=None, help='select the output directory')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'protect', 'infer'])
    parser.add_argument('--protecttype', type=str, default='Lpips', choices=['Lpips', 'original'])
    parser.add_argument('--rewrite',type=bool, default=True)
    parser.add_argument('--nproc_per_gpu',type=int, default=1)    

    
    
    args = parser.parse_args()
    return args

def main(rank, world_size):

    args = parse_args()
    nproc_per_gpu = args.nproc_per_gpu

    gpu_id = rank // nproc_per_gpu
    setup(rank, world_size, nproc_per_gpu)
    # 设置设备
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
    #model_id_or_path = "../stable-diffusion-inpainting"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
    
    mode=args.mode
    ptype=args.protecttype
    rewrite=args.rewrite
    '''''
    if model_id_or_path.split('-')[-1]=='inpainting':
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        print('modeltype:inpainting')
        pipe = pipe.to("cuda")
    elif "xl" in model_id_or_path.split('-'):
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id_or_path,
        )
        pipe = pipe.to("cuda")

        print('modeltype:XL')
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path,
        )
        pipe = pipe.to("cuda")
        print('modeltype:V1-5')
    '''''

    pipe = AutoencoderKL.from_pretrained(
            model_id_or_path, subfolder="vae"
        )
    pipe = pipe.to("cuda")

    '''''
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
    '''''

    dataset = MyAdvDataset(image_dir,image_list)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)  

    for init_images, image_args in dataloader :  
        image_name = image_args["image_name"][0] 
        init_image = init_images[0]
        image_arg = image_args
        if mode in ['all','protect']:
            if args.protecttype == 'Lpips':
                print(f'using Lpips protection, Lpipsvalues:{args.Lpipsvalues}')
                for Lpipsvalue in args.Lpipsvalues:
                    
                    print(f'use strength of {Lpipsvalue} Lpips')
                    
                    eps = args.eps[0]
                    #version = f"Constant{Lpipsvalue}Black"
                    #version = f"freetgtAppromin10eps{eps}it40000step3"
                    #version = f"Imax{Lpipsvalue}LpipsOssimEcDcAWAY"
                    version = f"Imin20appro{eps}epsEcDcAWAY"

                    image_adv_path, adv_image = adv_processes(version,output_dir,image_name,init_image,mode,pipe,ptype,rewrite,Lpipsvalue,eps)
                    print(f"{Lpipsvalue} Lpips of {image_name} protect finished")    
                                   
def adv_processes(version,output_dir,image_name,init_image,mode,pipe,ptype,rewrite,Lpipsvalue,eps):
    try:
        image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-photoguard-{version}.png"
    except Exception as e:
        print(f'image_name:{image_name}')
        raise e
    image_adv_path = os.path.join(output_dir, version, image_adv_name)   
    image_adv_dir = os.path.dirname(image_adv_path)    
    
    os.makedirs(image_adv_dir, exist_ok=True)
    
    if not os.path.exists(image_adv_path) or rewrite:
        adv_image = adv(pipe,image_adv_path, init_image,ptype, Lpipsvalue,eps)
        if mode=='all':
            print('create photoguard adv image to generate')
    elif mode=='all':
        adv_image = None
        print('using existing photoguard adv image to generate')

    return image_adv_path, adv_image
  
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

def ssim(img1, img2, C1=1e-6, C2=9e-6, window_size=11):
    # 创建高斯窗口
    def gaussian(window_size, sigma):
        x = torch.arange(window_size).float()  # 将 x 转换为张量
        gauss = torch.exp(-(x - (window_size - 1) / 2)**2 / (2 * sigma**2))
        return gauss / gauss.sum()
    # 生成窗口
    def create_window(window_size, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        return _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)

    # 确保输入张量为浮点类型
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    # 获取输入张量的形状
    (_, channel, height, width) = img1.size()
    
    # 创建高斯窗口
    window = create_window(window_size, 1.5).to(img1.device)
    window = window.expand(channel, 1, window_size, window_size)
    # 计算均值和方差
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def pgd(X, model, image_index, eps=0.1, step_size=0.2, iters=200, clamp_min=-1, clamp_max=1, mask=None, Lpipsvalue=10):
    print(f'X.shape:{X.shape},X.dtype:{X.dtype},Lpipsvalue:{Lpipsvalue},step_size{step_size},iters{iters}')
    
    X_adv = X.clone().detach().cuda() + (torch.rand(*X.shape, device=X.device)*2*eps-eps)
    X_adv.requires_grad_(True)
    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)
    pbar = tqdm(range(iters))
    
    #print(f'latentmax:{model(X).latent_dist.mean.max()} latentmin:{model(X).latent_dist.mean.min()}')

    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
    lpips = loss_fn_alex(X, X_adv)
    
    ratio  = 20
    ratio2 = 5

    print(f'lpips:{lpips}')

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        lpips = torch.maximum(loss_fn_alex(X, X_adv), torch.tensor(Lpipsvalue))
        appro = torch.maximum((X-X_adv).norm(), torch.tensor(20))
        latent_X_adv = model.encode(X_adv).latent_dist.mean
        decoded_X_adv = model.decode(latent_X_adv).sample
        decoded_appro = (X-decoded_X_adv).norm()
        #d_SSIM = ssim(X,decoded_X_adv)
        gap = (model.encode(X).latent_dist.mean-latent_X_adv).norm()

        loss = ratio*appro - decoded_appro
        #loss = ratio2*lpips + d_SSIM
        
        dappro_nott, appro_nott, gap_nott = list_to_scientific_notation([decoded_appro.item(), appro.item(), gap.item()])
        if i % (len(pbar)//4) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4} | appro: {appro_nott} | dappro_nott: {dappro_nott} | latent_gap: {gap_nott}", flush=True)
            #print(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}", flush=True)
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
        X_adv.data = torch.clamp(X_adv, min=X - eps, max=X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        if mask is not None:
            X_adv.data *= mask
    print(X_adv)
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return X_adv   
    
# You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
def adv(pipe,image_adv_path,init_image,ptype,Lpipsvalue,eps):
    
    X = init_image.cuda()
    image_index = image_adv_path.split('/')[-1].split('_')[0]
    #X = X.half()
    '''''
    if type(pipe).__name__ != "StableDiffusionXLImg2ImgPipeline":
        X = X.half()
    '''''
    if ptype=='Lpips':
        adv_X = pgd(X, 
                    model=pipe, 
                    image_index = image_index,
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=eps, # The higher, the less imperceptible the attack is 
                    step_size=5, # Set smaller than eps
                    iters=10000, # The higher, the stronger your attack will be
                    Lpipsvalue=Lpipsvalue,
                    
                    )
        
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
          
    adv_image = to_pil(adv_X[0]).convert("RGB")
    adv_image.save(image_adv_path)
    print(f'{image_adv_path} saved')
    return adv_image


if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank,world_size)