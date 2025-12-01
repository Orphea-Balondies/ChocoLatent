import torch
import numpy as np
import lpips as LPIPS
from torch.utils.data import DataLoader, DistributedSampler
from dataset import MyAdvDataset
from diffusers import StableDiffusionImg2ImgPipeline
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
    parser.add_argument('--Lpipsvalues', type=float, nargs='*', default=0.1)
    parser.add_argument('--image_dir', type=str, default="/home/zhangjianwei/photoguard/init_images/mix-27")
    parser.add_argument('--output_dir', type=str, default="/mnt/algorithm/user_dir/zhangjianwei/photoguard/output")
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
    timeout = datetime.timedelta(hours=2)
    dist.init_process_group(backend = "gloo", init_method="env://", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(gpu_id)
    # 设置设备
    device = torch.device(f"cuda:{gpu_id}")

    iters=args.iters
    step=args.step


    model_id_or_path = "/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
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
    for Lpipsvalue in args.Lpipsvalues:
        version = f"FeatureFakeConstant{Lpipsvalue}{iters}it{step}step"
        image_adv_dir = os.path.join(output_dir, version) 
        os.makedirs(image_adv_dir, exist_ok=True)
        outexcel_name = f'{version}_compare.xlsx'
        for init_images, image_args in dataloader :  
            image_name = image_args["image_name"][0] 
            init_image = init_images[0]

            try:
                image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-photoguard-{version}.png"
            except Exception as e:
                print(f'image_name:{image_name}')
                raise e
            print(f'image_adv_dir:{image_adv_dir}')
            image_adv_path = os.path.join(image_adv_dir, image_adv_name)   

            rewrite = 0
            print(f"image_adv_path:{image_adv_path}")
            lpips, gap = adv(pipe,image_adv_path, init_image, Lpipsvalue, iters, step, rewrite)

            row = {
                "image_name": image_name,
                "lpips": lpips,
                "gap": gap
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

def pgd(X, model, Lpipsvalue=10, step_size=0.2, iters=200, clamp_min=-1, clamp_max=1):
    #print(f'X.shape:{X.shape},X.dtype:{X.dtype},Lpipsvalue:{Lpipsvalue},step_size{step_size},iters{iters}')
    
    X_adv = torch.rand(*X.shape, device=X.device)*2-1
    X_adv.requires_grad_(True)
    pbar = tqdm(range(iters))
    
    print(f'latentmax:{model(X).latent_dist.mean.max()} latentmin:{model(X).latent_dist.mean.min()}')
    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
    lpips = loss_fn_alex(X, X_adv)
    gap = (model(X_adv).latent_dist.mean-model(X).latent_dist.mean).norm()
    ratio = 10
    
    #ratio2  = (0*gap/appro).detach().item()
    print(f"ratio:{ratio}")
    print(f'lpips:{lpips}')
    print(f"gap:{gap}")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 1500) / iters * i
        lpips = loss_fn_alex(X, X_adv)
        gap = (model(X_adv).latent_dist.mean-model(X).latent_dist.mean).norm()
        #appro = (X-X_adv).norm()
        loss = gap - ratio*lpips

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


    print(X_adv)
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)     
    return X_adv, lpips, gap

def adv(pipe,image_adv_path,init_image, Lpipsvalue, iters=15000, step_size=4, rewrite=0):
    X = init_image.cuda()
    if not os.path.exists(image_adv_path) or rewrite:
        adv_X, lpips, gap = pgd(X, 
                    model=pipe.vae.encode, 
                    Lpipsvalue=Lpipsvalue,
                    clamp_min=-1, 
                    clamp_max=1,
                    step_size=step_size, # Set smaller than eps
                    iters=iters, # The higher, the stronger your attack will be
                    )
        
        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
            
        adv_image = to_pil(adv_X[0]).convert("RGB")
        adv_image.save(image_adv_path)
        print(f'{image_adv_path} saved')
    else:
        adv_image = Image.open(image_adv_path).convert("RGB")  # 读取图片
        # 应用数据变换
        adv_X =  init_transform(adv_image).cuda()
        loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()
        lpips = loss_fn_alex(X, adv_X)
        gap = (pipe.vae.encode(adv_X).latent_dist.mean-pipe.vae.encode(X).latent_dist.mean).norm()
        print(f'read from {image_adv_path}')
    return lpips.detach().item(), gap.detach().item()
if __name__ == '__main__':
    main()