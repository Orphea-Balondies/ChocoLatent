from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage import io, color
from scipy.stats import entropy
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import os
import pandas as pd
import openpyxl
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLImg2ImgPipeline
import lpips
from decimal import Decimal

init_image_dir = '../init_images/old-photos'
adv_image_dir = '../output/old-photos/Imin35appro0.08epsEcDcAWAY'
target_image_dir = '/home/zhangjianwei/photoguard/latent_output/KateShadow/latentmax500Zi01approAWAYit20000st40'
output_dir = adv_image_dir
output_name = f'{adv_image_dir.split("/")[-2]}_compare.xlsx'
model_id_or_path = '../stable-diffusion-v1-5'
'''
image_nameinit = '16.png'
image_nameadv = '16_adv-photoguard-5ScimP0aprBlack01E.png'
image_namecompare = '16_adv-photoguard-0ScimP0aprBlack01E.png'


image_nameinit = 'two-cats-lying.png'
image_nameadv = 'two-cats-lying_adv-photoguard-05noisePnt-0.1.png'
image_namecompare = 'two-cats-lying_adv-photoguard.png'

image_nameinit = 'mountains-in-distance.jpg'
image_nameadv = 'mountains-in-distance_adv-photoguard-05noisePnt-0.1.png'
image_namecompare = 'mountains-in-distance_adv-photoguard-0.02.png'

gened_suffix = image_nameinit.split('.')[-1]

image_list=dict(
  image1_nameinit = image_nameinit,
  image1_nameinit_gened = image_nameinit[:image_nameinit.rfind('.')] + '_gened.' + gened_suffix,
  image1_nameadv = image_nameadv,
  image1_nameadv_gened = image_nameadv[:image_nameadv.rfind('.')] + '_gened.' + gened_suffix,
  image1_namecompare = image_namecompare,
  image1_namecompare_gened = image_namecompare[:image_namecompare.rfind('.')] + '_gened.' + gened_suffix
)

cal_list={
('a2iP', 'a2iS', 'a2iR'):['image1_nameadv', 'image1_nameinit'],
('c2iP', 'c2iS', 'c2iR'):['image1_namecompare', 'image1_nameinit'],
('ag2iP', 'ag2iS', 'ag2iR'):['image1_nameadv_gened', 'image1_nameinit'],
('cg2iP', 'cg2iS', 'cg2iR'):['image1_namecompare_gened', 'image1_nameinit'],
('ag2igP', 'ag2igS', 'ag2igR'):['image1_nameadv_gened', 'image1_nameinit_gened'],
('cg2igP', 'cg2igS', 'cg2igR'):['image1_namecompare_gened', 'image1_nameinit_gened']
}
'''

def image_metric(image):
  if len(image.shape) == 3 and image.shape[2] == 3:
      image = color.rgb2gray(image) 
    #print(f"image:{image}")  
  hist, _ = np.histogram(image.flatten(), bins=256, range=[0,1])
  #print(f"Histogram: {hist}")
  hist = hist / hist.sum()

  ent = entropy(hist + 1e-10)  
  
  return image.mean(), image.var(), ent

def torch_image(image_path):
  image_np = np.transpose(load_and_preprocess_image(image_path), (2, 0, 1))
  image = torch.from_numpy(np.expand_dims(image_np, axis=0))
  return image

def image_metric_row2(image_dir, image_name, type, image_index):
  image_path = os.path.join(image_dir, image_name)
  image = torch_image(image_path)
  zeros_torch = torch.zeros_like(image)
  print(image.shape)
  print(zeros_torch.shape)
  lpips = loss_fn_alex(image, zeros_torch).item()
  gap2black = (model(image).latent_dist.mean).norm().detach().item()
  target_tensor = None
  for target_image_name in os.listdir(target_image_dir):
    if image_index == target_image_name[:len(image_index)] and target_image_name[len(image_index)]=='_':
      if target_image_name.endswith('.png'):
        target_image_path = os.path.join(target_image_dir, target_image_name)
        target_image = torch_image(target_image_path)
      if target_image_name.endswith('.pt'):
        target_tensor_path = os.path.join(target_image_dir, target_image_name)
        target_tensor = torch.load(target_tensor_path, map_location='cpu')
  if target_tensor is not None:
      target2black =  target_tensor.norm().detach().item()
      lpips2target = loss_fn_alex(image, target_image).item()
      gap2target = (model(image).latent_dist.mean-target_tensor).norm().detach().item()
  else:
      target2black, lpips2target, gap2target = 0,0,0 
  lpips, gap2black, lpips2target, gap2target = list_to_scientific_notation([lpips, gap2black, lpips2target, gap2target])

  row_name = {
    "original":["Olpips2zero","Ogap2black","original_name", "Olpips2target", "Ogap2target"],
    "protected":["Plpips2zero","Pgap2black","protected_name", "Plpips2target", "Pgap2target"],
    "generated":["Glpips2zero","Ggap2black","generated_name", "Glpips2target", "Ggap2target"]
  }
  row = {
    row_name[type][0]: lpips,
    row_name[type][1]: gap2black,
    row_name[type][2]: image_name,
    row_name[type][3]: lpips2target,
    row_name[type][4]: gap2target,
  }
  return row, target2black

def image_metric_row(image_dir, image_name):
  image_path = os.path.join(image_dir, image_name)
  image = load_and_preprocess_image(image_path)
  mean, var, ent = image_metric(image)

  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)

  latent_image = model(image).latent_dist.mean.detach()
  latent_image = latent_image.numpy()
  latent_mean, latent_var, latent_ent = image_metric(latent_image)
  ''''''
  row = {
    "image_name": image_name,
    #"image_mean": mean,
    #"image_var": var,
    #"image_ent": ent,
    #"latent_mean": latent_mean,
    #"latent_var": latent_var,
    #"latent_ent": latent_ent,
  }
  return row

def load_and_preprocess_image(image_path, target_size=None):
    image = Image.open(image_path).convert('RGB')  # 转换为RGB模式
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    image = center_crop(resize(image))
    image = np.array(image).astype(np.float32) / 255.0

    return 2*image - 1.0


def list_to_scientific_notation(numbers):
    # 将数字转换为 Decimal 类型
    for i in range(len(numbers)):
      dec = Decimal(str(numbers[i])).normalize()
      # 去除小数点后不必要的零
      # 使用科学计数法保留有效数字并输出
      numbers[i] = f"{dec:.3e}"
    return numbers


def calculate_psnr_ssim(image_path1, image_path2):
    '''
    # image1 和 image2 的形状应该是 (height, width, channels)
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    target_size = (min(image1.size[0], image2.size[0]), min(image1.size[1], image2.size[1]))
    '''
    image1 = load_and_preprocess_image(image_path1)
    image2 = load_and_preprocess_image(image_path2)
    
    psnr_value = peak_signal_noise_ratio(image1, image2, data_range=1)

    win_size = min(image1.shape[0], image1.shape[1])
    if win_size % 2 == 0:  # win_size 必须是奇数
        win_size -= 1
    try:
      ssim_value = structural_similarity(image1, image2, multichannel=True, channel_axis=2, win_size=win_size,  data_range=1.0)
    except Exception as e:
      print(f'image1:{image1.shape},image2:{image2.shape}')
      raise e

    rmse = np.sqrt(mean_squared_error(image1, image2))

    image1_tensor = torch.from_numpy(np.expand_dims(np.transpose(image1, (2, 0, 1)), axis=0))
    image2_tensor = torch.from_numpy(np.expand_dims(np.transpose(image2, (2, 0, 1)), axis=0))
    print(image1_tensor)
    print(image2_tensor)
    crosslpips  = loss_fn_alex(image1_tensor, image2_tensor).item()
    mse = (image1_tensor-image2_tensor).norm().item()

    psnr_value, ssim_value, crosslpips, mse = list_to_scientific_notation([psnr_value, ssim_value, crosslpips, mse])

    latent_gap=(model(image1_tensor).latent_dist.mean-model(image2_tensor).latent_dist.mean).norm().detach().item()


    return psnr_value, ssim_value, crosslpips, latent_gap, mse
    

if model_id_or_path.split('-')[-1]=='inpainting':
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        print('modeltype:inpainting')     
elif "xl" in model_id_or_path.split('-'):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id_or_path,
    )

    print('modeltype:XL')
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
    )
    print('modeltype:V1-5')

model = pipe.vae.encode
loss_fn_alex = lpips.LPIPS(net='alex')

xlsx_list=[]
for image_name in os.listdir(init_image_dir):
  if image_name.split('.')[-1].lower() not in ['jpg','png','jpeg']:
     continue
  init_image_path = os.path.join(init_image_dir, image_name)
  image_index = image_name.split('.')[0]
  init_row,_ = image_metric_row2(init_image_dir, image_name, "original", image_index)
  #xlsx_list.append(init_row)
  for adv_image_name in os.listdir(adv_image_dir):
    if image_index == adv_image_name[:len(image_index)] and adv_image_name[len(image_index)]=='_' and adv_image_name.endswith('.png'):
      adv_image_path = os.path.join(adv_image_dir, adv_image_name)
      adv_row, target2black = image_metric_row2(adv_image_dir, adv_image_name, "protected", image_index)
      adv_row["target2black"] = target2black
      adv_row["psnr"], adv_row["ssim"], adv_row["crosslpips"], adv_row["latent_gap"], adv_row["mse"] = calculate_psnr_ssim(init_image_path, adv_image_path)
      init_row.update(adv_row)
      print(init_row)
      xlsx_list.append(init_row)

df = pd.DataFrame(xlsx_list)
# 写入 Excel 文件
df.to_excel(os.path.join(output_dir, output_name), index=False, engine='openpyxl')
    

'''''
for cal,value in cal_list.items():
  value.append(calculate_psnr_ssim(image_path + image_list[value[0]], image_path + image_list[value[1]]))
  print(f"{cal[0]}:{value[2][0]}  {cal[1]}:{value[2][1]}  {cal[2]}:{value[2][2]}")
for image_name in image_list.values():
  ent = calculate_entropy(os.path.join(image_path,image_name))
  print(f"ENT-{image_name}:{ent}")

'''''