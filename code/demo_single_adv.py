from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from pgd import pgd
from torchvision import transforms
import torchvision.transforms as T
from utils import PreprocessTransform,StepLossCollector
import os
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
image_path = "init_images/lego-minifigure-faces/0001.jpg"
expname = "04lr400e-5fc-01id04il15md01ml"
image_adv_root = "/home/zjw/ChocoLatent/exp/demo/"
image_adv_path = os.path.join(image_adv_root,expname,'lego-0001.jpg')
image = Image.open(image_path).convert("RGB")  # 读取图片
init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])
init_image = init_transform(image)[None].cuda()
model_id_or_path = '/root/chocolatent/model/stable-diffusion-v1-5'
model = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
#model = model.to("cuda")
pipe = model.vae
pipe = pipe.to("cuda")
step_collector = StepLossCollector()
X_adv = pgd(init_image,pipe,iters=400,initial_lr=0.4,step_collector=step_collector,r_f_c=5,r_i_d=0.1,r_i_l=0.4,r_d_d=0.05)

X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
adv_image = T.ToPILImage()(X_adv[0]).convert("RGB")
os.makedirs(os.path.dirname(image_adv_path),exist_ok=True)
adv_image.save(image_adv_path)
print(f'{image_adv_path} saved')
tb_path = os.path.join(image_adv_root,expname)
os.makedirs(tb_path,exist_ok=True)
tb_writer = SummaryWriter(tb_path)
for step, step_data in step_collector.step_metrics.items():
    for metric_name, metric_tensors in step_data.items():
        if metric_name != 'decoded_X_adv':
            tb_writer.add_scalar(f'{metric_name}', metric_tensors[0], step)
        elif metric_name == 'decoded_X_adv':
            tb_writer.add_images(f'{metric_name}', metric_tensors[0], step)


print(f"eps of {image_path} adv protect finished")
