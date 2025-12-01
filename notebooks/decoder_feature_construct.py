import torch
import numpy as np
import lpips as LPIPS
from torch.utils.data import DataLoader, DistributedSampler
from dataset import MyAdvDataset
from diffusers import AutoencoderKL
from tqdm import tqdm
import os
import torch.distributed as dist
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import json
import openpyxl
from torchvision import transforms
from utils import PreprocessTransform
from PIL import Image
from decimal import Decimal
import argparse

init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])
to_pil = T.ToPILImage()
"""""
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "8"
"""""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=15000)
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--latent_gap_mins', type=int, nargs='*', default=600)
    parser.add_argument('--imgappro_min', type=int, nargs='*', default=70)
    parser.add_argument('--image_dir', type=str, default="/home/zhangjianwei/photoguard/init_images/mix-27")
    parser.add_argument('--output_dir', type=str, default="/mnt/algorithm/user_dir/zhangjianwei/photoguard/latent_output")
    parser.add_argument('--nproc_per_gpu', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    nproc_per_gpu = args.nproc_per_gpu
    gpu_id = rank // nproc_per_gpu
    import datetime
    timeout = datetime.timedelta(hours=4)
    dist.init_process_group(backend = "gloo", init_method="env://", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(gpu_id)
    # 设置设备
    device = torch.device(f"cuda:{gpu_id}")

    iters=args.iters
    step=args.step


    model_id_or_path = "/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5"
    pipe = AutoencoderKL.from_pretrained(
            model_id_or_path, subfolder="vae"
        )
    pipe = pipe.to("cuda")
    
    image_dir = args.image_dir
    image_list = []
    for _,_,files in os.walk(image_dir):
        for f in files:
            if f.split('.')[-1].lower() not in ['jpg','png','jpeg']:
                continue
            image_dict={"image_name": f}
            image_list.append(image_dict)

    output_dir = args.output_dir

    dataset = MyAdvDataset(image_dir,image_list)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)  
    xlsx_list = []
    for latent_gap_min in args.latent_gap_mins:

        imgappro_min = args.imgappro_min

        version = f"latentmax{latent_gap_min}Zi01approAWAYit{iters}st{step}"

        image_adv_dir = os.path.join(output_dir, version) 
        os.makedirs(image_adv_dir, exist_ok=True)
        outexcel_name = f'{version}_compare.xlsx'
        for init_images, image_args in dataloader :  
            image_name = image_args["image_name"][0] 
            init_image = init_images[0]

            try:
                image_adv_name = f"{image_name[:image_name.rfind('.')]}_{version}.png"
            except Exception as e:
                print(f'image_name:{image_name}')
                raise e
            print(f'image_adv_dir:{image_adv_dir}')
            image_adv_path = os.path.join(image_adv_dir, image_adv_name)   

            rewrite = 0
            if os.path.exists(image_adv_path):
                continue
            print(f"image_adv_path:{image_adv_path}")
            lpips, gap, SSIM_nott,gap2black,mse, latent_max, latent_min = adv(pipe,image_adv_path, init_image, latent_gap_min, imgappro_min, iters, step, rewrite)

            row = {
                "image_name": image_name,
                "lpips": lpips,
                "gap": gap,
                "SSIM": SSIM_nott,
                "gap2black":gap2black,
                "mse":mse,
                "latent_max": latent_max,
                "latent_min": latent_min
            }
            xlsx_list.append(row)
        temp_file = os.path.join(image_adv_dir,f"temp_output_{rank}.json")
        try:
            with open(temp_file,'w') as f:
                json.dump(xlsx_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(xlsx_list)
        
        torch.distributed.barrier()
        if rank == 0:
            xlsx_list = []
            for r in range(world_size):
                temp_file = os.path.join(image_adv_dir,f"temp_output_{r}.json")
                try:
                    with open(temp_file, 'r') as f:
                        xlsx_list.extend(json.load(f))
                    os.remove(temp_file)
                except Exception as e:
                    pass
            df = pd.DataFrame(xlsx_list)
            # 写入 Excel 文件
            df.to_excel(os.path.join(image_adv_dir, outexcel_name), index=False, engine='openpyxl')

def list_to_scientific_notation(numbers):
    # 将数字转换为 Decimal 类型
    for i in range(len(numbers)):
      dec = Decimal(str(numbers[i])).normalize()
      # 去除小数点后不必要的零
      # 使用科学计数法保留有效数字并输出
      numbers[i] = f"{dec:.3e}"
    return numbers

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean([2, 3], keepdim=True)
    mu2 = img2.mean([2, 3], keepdim=True)

    sigma1 = img1.var([2, 3], keepdim=True)
    sigma2 = img2.var([2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean([2, 3], keepdim=True)

    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1.pow(2) + mu2.pow(2) + C1) * (sigma1 + sigma2 + C2)
    
    ssim_map = ssim_numerator / ssim_denominator
    return ssim_map.mean()

def pgd(X, model, latent_gap_min=600, imgappro_min=70, step_size=0.2, iters=200, clamp_min=-1, clamp_max=1):
    #print(f'X.shape:{X.shape},X.dtype:{X.dtype},latent_gap_min:{latent_gap_min},step_size{step_size},iters{iters}')
    latent_X = model.encode(X).latent_dist.mean
    latent_X_adv = torch.zeros(*latent_X.shape, device=X.device)
    latent_X_adv.requires_grad_(True)
    pbar = tqdm(range(iters))
    
    print(f'latentmax:{latent_X.max()} latentmin:{latent_X.min()}')
    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()

    ratio = 200
    ratio2 = 0.1

    print(f"ratio:{ratio}")

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 1500) / iters * i
        X_adv = model.decode(latent_X_adv).sample
        
        lpips = loss_fn_alex(X, X_adv)
        
        gap = torch.maximum((latent_X-latent_X_adv).norm(), torch.tensor(latent_gap_min))
        
        appro = (X-X_adv).norm()
        
        loss = -ratio2*appro + gap

        appro_nott, lpips_nott, gap_nott = list_to_scientific_notation([appro.item(), lpips.item(), gap.item()])
        if i % (len(pbar)//8) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4}")
            print(f"[Running attack]: Loss {loss.item():.2f} | step size: {actual_step_size:.4} | appro_nott : {appro_nott} | latent_gap: {gap_nott}", flush=True)
            pbar.update(len(pbar)//8)

        grad, = torch.autograd.grad(loss, [latent_X_adv])
        '''
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        '''

        grad_flattened = grad.view(grad.size(0), grad.size(1), -1)
        normalized_grad = F.normalize(grad_flattened, p=2, dim=-1)
        normalized_grad = normalized_grad.view_as(grad)
        #print(f"normalized_grad:{normalized_grad}")
        latent_X_adv = latent_X_adv - normalized_grad.detach() * actual_step_size
        #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        latent_X_adv.grad = None
    #print(latent_X_adv)
    lpips = loss_fn_alex(X, X_adv)
    SSIM = ssim(X, X_adv)
    mse = (X-X_adv).norm().item()
    SSIM_nott, lpips_nott, gap_nott = list_to_scientific_notation([SSIM.item(), lpips.item(), gap.item(), ])
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return latent_X_adv, X_adv, lpips_nott, gap_nott, SSIM_nott, mse, latent_X_adv.max().item(), latent_X_adv.min().item()

def adv(pipe,image_adv_path,init_image, latent_gap_min, imgappro_min, iters=15000, step_size=4, rewrite=1):
    X = init_image.cuda()
    latent_adv_path = image_adv_path.replace('.png', '.pt')
    if not os.path.exists(image_adv_path) or not os.path.exists(latent_adv_path) or  rewrite:
        latent_X_adv, X_adv, lpips_nott, gap_nott, SSIM_nott, mse, latent_max, latent_min = pgd(X, 
                    model=pipe, 
                    latent_gap_min=latent_gap_min,
                    imgappro_min=imgappro_min,
                    clamp_min=-1, 
                    clamp_max=1,
                    step_size=step_size, # Set smaller than eps
                    iters=iters, # The higher, the stronger your attack will be
                    )
        gap2black = latent_X_adv.norm().item()
        torch.save(latent_X_adv, latent_adv_path)
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
        adv_image = to_pil(X_adv[0]).convert("RGB")
        adv_image.save(image_adv_path)
        print(f'{image_adv_path} saved')
    else:
        adv_image = Image.open(image_adv_path).convert("RGB")  # 读取图片
        # 应用数据变换
        latent_X_adv = torch.load(latent_adv_path)
        X_adv =  init_transform(adv_image).cuda()
        loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
        lpips = loss_fn_alex(X, X_adv)
        mse = (X-X_adv).norm().item()
        SSIM = ssim(X, X_adv)
        gap = (pipe.encode(X).latent_dist.mean-latent_X_adv).norm()
        SSIM_nott, lpips_nott, gap_nott = list_to_scientific_notation([SSIM.item(), lpips.item(), gap.item()])
        gap2black = latent_X_adv.norm().item()
        latent_max = latent_X_adv.max().item()
        latent_min = latent_X_adv.min().item()
        print(f'read from {image_adv_path}')
    return lpips_nott, gap_nott, SSIM_nott, gap2black, mse, latent_max, latent_min

if __name__ == '__main__':
    main()