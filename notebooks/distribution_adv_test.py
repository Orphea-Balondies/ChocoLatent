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
from torchvision import transforms
from utils import PreprocessTransform
from decimal import Decimal
to_pil = T.ToPILImage()
init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])
# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
latent_root = '/home/zhangjianwei/photoguard/latent_output/KateShadow/'
latent_version = 'latentmax500Zi01approAWAYit20000st40'
latent_dir = os.path.join(latent_root, latent_version)

def setup(rank, world_size, nproc_per_gpu):
    # 初始化分布式环境
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    gpu_id = rank // nproc_per_gpu
    torch.cuda.set_device(gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../demo/images/', help='The main directory path of images')
    parser.add_argument('--image_list', nargs='+', help='Images\' name, could be a string or list. If infer')
    parser.add_argument('--Lpipsvalues', nargs='*', type=int, default=[1])
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
    
    mode=args.mode
    ptype=args.protecttype
    rewrite=args.rewrite
    
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
                    version = f"min60appro{eps}Eps{latent_version}"
                    
                    image_adv_path, adv_image = adv_processes(version,output_dir,image_name,init_image,mode,pipe,ptype,rewrite,Lpipsvalue,eps)
                    print(f"{Lpipsvalue} Lpips of {image_name} protect finished")    
            if args.protecttype == 'original':
                print(f'using original protection, epsvalues:{args.eps}')
                for eps in args.eps:
                    
                    print(f'use limitation of {eps} eps')
                    
                    Lpipsvalue= args.Lpipsvalues[0]
                    version = f"original{str(eps).replace('.','')}E"
                    
                    image_adv_path, adv_image = adv_processes(version,output_dir,image_name,init_image,mode,pipe,ptype,rewrite,Lpipsvalue,eps)
                    print(f"{eps} eps of {image_name} original protect finished") 
                    
                    
        if args.mode in ['infer','all']:

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
        
def adv_processes(version,output_dir,image_name,init_image,mode,pipe,ptype,rewrite,Lpipsvalue,eps):
    try:
        image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-photoguard-{version}.png"
    except Exception as e:
        print(f'image_name:{image_name}')
        raise e
    image_adv_path = os.path.join(output_dir, version, image_adv_name)   
    image_adv_dir = os.path.dirname(image_adv_path)    
    
    if not os.path.exists(image_adv_dir):
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
   
def pgd(X, model, image_index, eps=0.1, step_size=0.2, iters=200, clamp_min=-1, clamp_max=1, mask=None, Lpipsvalue=10):
    print(f'X.shape:{X.shape},X.dtype:{X.dtype},Lpipsvalue:{Lpipsvalue},step_size{step_size},iters{iters}')
    X_adv = X.clone().detach().cuda() + (torch.rand(*X.shape, device=X.device)*2*eps-eps)
    X_adv.requires_grad_(True)
    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)
    pbar = tqdm(range(iters))
    
    print(f'latentmax:{model(X).latent_dist.mean.max()} latentmin:{model(X).latent_dist.mean.min()}')
    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
    lpips = loss_fn_alex(X, X_adv)
    '''''
    epst = (model(X).latent_dist.mean.max() - model(X).latent_dist.mean.min())
    target_tensor = torch.rand_like(model(X).latent_dist.mean) * epst + model(X).latent_dist.mean.min()
    
    for featurefake_image_name in os.listdir("/home/zhangjianwei/photoguard/eliminated/FeatureFakeNolim15000it4step"):
        if image_index == featurefake_image_name[11:11+len(image_index)] and featurefake_image_name[11+len(image_index)]=='_':
            featurefake_image = Image.open(os.path.join("/home/zhangjianwei/photoguard/eliminated/FeatureFakeNolim15000it4step",featurefake_image_name)).convert("RGB")  # 读取图片
            # 应用数据变换
            featurefake_X =  init_transform(featurefake_image).cuda()
            target_tensor = model(featurefake_X).latent_dist.mean
    '''''  
    global device
    tag = 0                 
    for latent_name in os.listdir(latent_dir):
        if image_index == latent_name[:len(image_index)] and latent_name[len(image_index)]=='_' and latent_name.endswith('.pt'):
            assert tag != 1
            latent_path = os.path.join(latent_dir, latent_name)
            target_tensor = torch.load(latent_path, map_location=device)
            tag = 1
    assert tag == 1

    #gap2black = (model(X_adv).latent_dist.mean).norm()
    gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
    gap = gap2noise
    #appro = (X-X_adv).norm()
    #ratio =(Lpipsvalue*gap/lpips).detach().item()
    ratio = Lpipsvalue
    ratio2  = 150
    
    print(f"ratio:{ratio}")
    print(f'lpips:{lpips}')
    print(f"gap:{gap}")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        #lpips = loss_fn_alex(X, X_adv)
        #gap2black = (model(X_adv).latent_dist.mean).norm()
        gap2noise = (model(X_adv).latent_dist.mean-target_tensor).norm()
        gap = gap2noise
        #appro = (X-X_adv).norm()
        appro = torch.maximum((X-X_adv).norm(), torch.tensor(60))
        loss = gap + ratio2*appro #+ ratio*lpips
        #loss = gap+ratio/similarity
        
        appro_nott, gap_nott = list_to_scientific_notation([appro.item(), gap.item()])
        if i % (len(pbar)//8) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4} | appro: {appro_nott} | latent_gap: {gap_nott}", flush=True)
            #print(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4}", flush=True)
            pbar.update(len(pbar)//8)

        grad, = torch.autograd.grad(loss, [X_adv])
        '''
        X_adv = X_adv - grad.detach() * actual_step_size
        
        X_adv.data = torch.clamp(X_adv, X - eps, X + eps)

        '''
        grad_flattened = grad.view(grad.size(0), grad.size(1), -1)
        normalized_grad = F.normalize(grad_flattened, p=2, dim=-1)
        normalized_grad = normalized_grad.view_as(grad)
        #print(f"normalized_grad:{normalized_grad}")
        X_adv = X_adv - normalized_grad.detach() * actual_step_size
        X_adv.data = torch.clamp(X_adv, min=X - eps, max=X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        #torch.cuda.empty_cache()

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
                    model=pipe.encode, 
                    image_index = image_index,
                    clamp_min=-1,
                    clamp_max=1,
                    eps=eps, # The higher, the less imperceptible the attack is 
                    step_size=5, # Set smaller than eps
                    iters=20000, # The higher, the stronger your attack will be
                    Lpipsvalue=Lpipsvalue,
                    
                    )
    elif ptype=='original':
        adv_X = pgd_original(X, 
                    model=pipe.encode, 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=eps, # The higher, the less imperceptible the attack is 
                    step_size=eps/2, # Set smaller than eps
                    iters=800, # The higher, the stronger your attack will be
                    )
    # convert pixels back to [0,1] range
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
          
    adv_image = to_pil(adv_X[0]).convert("RGB")
    adv_image.save(image_adv_path)
    print(f'{image_adv_path} saved')
    return adv_image
      
    
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