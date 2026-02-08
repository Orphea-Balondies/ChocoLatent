import os
from PIL import Image, ImageOps
import requests
import torch
import numpy as np
import sys
import torch
import torch.nn.functional as F
import requests
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModel
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from dataset import CLDataset
import argparse
import json, os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms
from utils import PreprocessTransform,preprocess, recover_image,StepLossCollector
from pgd import pgd
from datetime import datetime
to_pil = T.ToPILImage()
init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])

def setup(rank, world_size, nproc_per_gpu):
    # 初始化分布式环境
    #dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    dist.init_process_group("gloo")
    gpu_id = rank // nproc_per_gpu
    torch.cuda.set_device(gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default='init_images/', help='The main directory path of images')
    parser.add_argument('--image_dirname', type=str, default='lego-minifigure-faces', help='The main directory path of images')
    parser.add_argument('--model_path', type=str, default='model/stable-diffusion-v1-5')
    parser.add_argument('--adv_output_root', type=str, default='exp', help='select the output directory')
    parser.add_argument('--rewrite',type=bool, default=True)
    parser.add_argument('--nproc_per_gpu',type=int, default=1)
    parser.add_argument('--batch_size',type=int, default=4)   

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

    image_dir = os.path.join(args.image_root,args.image_dirname)
    output_root = args.adv_output_root
    out_dirname = args.image_dirname

    model_id_or_path = args.model_path
    model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    model = model.to("cuda")
    pipe = model.vae

    clip_model = CLIPVisionModel.from_pretrained("model/clip-vit-base-patch32").to("cuda")

    dataset = CLDataset(image_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)


    lr = 4
    e = 600
    fc = 8
    id = 0.1
    il = 0.2
    md = 20
    ml = 0.15
    dcc = 8
    dl = 0.1
    #exp_version = f"{lr}lr{e}e-{fc}fc_meanpatch-{id}id{il}il{md}md{ml}ml".replace('.', '')
    exp_version = f"{lr}lr{e}e-{dl}decoded_lpips-{id}id{il}il{md}md{ml}ml".replace('.', '')
    step_collector = StepLossCollector()
    for init_images, image_args in dataloader :
        image_name = image_args["image_name"][0]
        init_image = init_images.cuda()
        image_arg = image_args


        image_adv_name = f"{image_name[:image_name.rfind('.')]}_adv-{exp_version}.png"
        image_adv_path = os.path.join(output_root, exp_version, 'adv_images', out_dirname, image_adv_name)
        os.makedirs(os.path.dirname(image_adv_path), exist_ok=True)

        rewrite=args.rewrite
        if not os.path.exists(image_adv_path) or rewrite:
            print('create photoguard adv image to generate')
            X_adv = pgd(init_image,pipe,iters=e,initial_lr=lr,step_collector=step_collector,clip_model=clip_model,\
                        r_f_c=fc,r_i_d=id,r_i_l=il,r_d_l=dl,max_img_Dis=md,max_img_lpips=ml,r_d_cc=dcc)
            X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
            adv_image = to_pil(X_adv[0]).convert("RGB")
            adv_image.save(image_adv_path)
            print(f'{image_adv_path} saved')
    print(f"rank {rank} adv finished")
    dist.barrier()
    #if rank == 0:
    print(f"START TO GATHER")
    all_step_data = gather_step_data(step_collector, world_size, rank)
    if rank == 0:
        current_time = datetime.now().strftime("%m-%d-%H-%M")
        log_path = os.path.join(output_root, exp_version, "log", out_dirname, current_time)
        os.makedirs(log_path,exist_ok=True)
        tb_writer = SummaryWriter(log_path)
        print("Recording to TensorBoard...")
        for step, step_data in all_step_data.items():
            process_step_to_tensorboard(step_data, tb_writer, step)
    dist.destroy_process_group()

def gather_step_data(step_collector, world_size, rank):
    """收集所有进程的step数据"""
    all_step_data = {}
    device="cuda:{}".format(rank)
    # 确定最大step数
    max_step = max(step_collector.step_metrics.keys()) + 1 if step_collector.step_metrics else 0

    for step in range(max_step):

        # 收集指标
        gathered_metrics = {}
        for metric_name in step_collector.step_metrics[step].keys():
            local_metrics = step_collector.step_metrics[step][metric_name]

            metric_tensor = torch.tensor(local_metrics, dtype=torch.float32, device=device)
            gathered_metric = [torch.zeros_like(metric_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_metric, metric_tensor)
            gathered_metrics[metric_name] = gathered_metric

        all_step_data[step] = gathered_metrics
        #if rank == 0:
        #    print(f"step{step} all gathered")
    return all_step_data

def process_step_to_tensorboard(step_data, tb_writer, step):
    """处理step数据并记录到TensorBoard"""
    # 处理其他指标
    for metric_name, metric_tensors in step_data.items():
        all_metrics = []
        for tensor in metric_tensors:
            if tensor.numel() > 0:
                all_metrics.extend(tensor.tolist())

        if all_metrics:
            mean_metric = np.mean(all_metrics)
            tb_writer.add_scalar(f'{metric_name}', mean_metric, step)
            tb_writer.add_histogram(f'{metric_name}_dist', np.array(all_metrics), step)

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank,world_size)
                                                    