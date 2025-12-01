from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage import io, color
from scipy.stats import entropy
import numpy as np
import torch
import lpips
import os
from decimal import Decimal
import pandas as pd
import torchvision.transforms as T
import copy
img_root = "/home/zhangjianwei/photoguard/generated"
loss_fn_alex = lpips.LPIPS(net='alex')

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

def p_d(A, B):
    # 计算平均值
    # 计算百分比差距
    diff = abs(B - A) / B * 100
    return diff

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
      #print(f'image1:{image1.shape},image2:{image2.shape}')
      raise e

    rmse = np.sqrt(mean_squared_error(image1, image2))

    image1_tensor = torch.from_numpy(np.expand_dims(np.transpose(image1, (2, 0, 1)), axis=0))
    image2_tensor = torch.from_numpy(np.expand_dims(np.transpose(image2, (2, 0, 1)), axis=0))
    #print(image1_tensor)
    #print(image2_tensor)
    crosslpips  = loss_fn_alex(image1_tensor, image2_tensor).item()
    mse = (image1_tensor-image2_tensor).norm().item()

    

    return psnr_value, ssim_value, crosslpips, mse


for lora_name in ['lego','old-photos','van-gogh']:
    original_root = os.path.join(img_root,f'{lora_name}-original')
    out_xlsx_path = os.path.join(original_root, f'{lora_name}_evaluation.xlsx')
    xlsx_list = []
    every_row = {}
    #mean_dict = {'min20eps0.08Black':{}, 'Imin20appro0.08epsEcDcAWAY':{}}
    for prompt_name in os.listdir(original_root):
        original_dir = os.path.join(original_root,prompt_name)
        if not os.path.isdir(original_dir):
            continue
        for img_name in  os.listdir(original_dir):
            original_path = os.path.join(original_dir, img_name)
            every_row["id"] = f"{lora_name}/{prompt_name}/{img_name}"
            for adv_version in ['min20eps0.08Black', 'Imin20appro0.08epsEcDcAWAY']:
                #mean_dict[adv_version][prompt_name] = []
                adv_path = os.path.join(img_root, f'{lora_name}-{adv_version}', prompt_name, img_name)
                metrics_valu = calculate_psnr_ssim(original_path, adv_path)
                for i,metric in enumerate(['psnr','ssim','lpips','mse']):
                    every_row[f'{adv_version}_{metric}'] = metrics_valu[i]
            xlsx_list.append(copy.deepcopy(every_row))
        print (f"{lora_name}/{prompt_name} evaluation complete")

    df = pd.DataFrame(xlsx_list)
    # 写入 Excel 文件
    average_row = {}
    difference_row = {}
    for column_name in df.columns:
        if column_name != 'id':
            average_column = df[column_name].mean()
            average_row[column_name] = average_column
        average_row['id'] = 'average'
    for i,metric in enumerate(['psnr','ssim','lpips','mse']):
        difference =p_d(average_row[f'Imin20appro0.08epsEcDcAWAY_{metric}'], average_row[f'min20eps0.08Black_{metric}'])
        difference_row[f'Imin20appro0.08epsEcDcAWAY_{metric}'] = difference
    difference_row['id'] = 'difference'
    df.loc[len(df)] = average_row
    df.loc[len(df)] = difference_row

    df_without_id = df.drop(columns=['id'])

    # 使用 applymap 将数值列转换为科学计数法
    df_sci = df_without_id.applymap(lambda x: f"{x:.3e}")

    # 将转换后的结果合并回原 DataFrame
    df.update(df_sci)
    df.to_excel(out_xlsx_path, index=False, engine='openpyxl')
    print (f"{lora_name} evaluation all complete!!")

