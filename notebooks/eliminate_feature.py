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
import os
import torchvision.transforms as T
init_image_dir = '../init_images/mix-27'
featurefake_image_dir = '/home/zhangjianwei/photoguard/output/FeatureFakeNolim15000it4step'
eliminated_output_root = '../eliminated/test/'
eliminated_output_dir = os.path.join(eliminated_output_root,os.path.basename(featurefake_image_dir))
os.makedirs(eliminated_output_dir, exist_ok=True)
import numpy as np
from PIL import Image

def main():
    for image_name in os.listdir(init_image_dir):
        init_image_path = os.path.join(init_image_dir, image_name)
        init_X = load_image(init_image_path)
        #xlsx_list.append(init_row)
        image_index = image_name.split('.')[0]
        for featurefake_image_name in os.listdir(featurefake_image_dir):
            if image_index == featurefake_image_name[:len(image_index)] and featurefake_image_name[len(image_index)]=='_':
                eliminated_output_name = f"eliminated_{featurefake_image_name}"
                eliminated_output_name2 = f"2_eliminated_{featurefake_image_name}"
                featurefake_image_path = os.path.join(featurefake_image_dir, featurefake_image_name)
                featurefake_X = load_image(featurefake_image_path)
                eliminated_X = init_X - featurefake_X + 255.0/2.0
                
                eliminated_X1 = np.round(eliminated_X)
                # 确保数据范围在 [0, 255]
                eliminated_X1 = np.clip(eliminated_X1, 0, 255)
                
                eliminated_output_path = os.path.join(eliminated_output_dir,eliminated_output_name)
                eliminated_image1 = T.ToPILImage()(eliminated_X1)
                eliminated_image1 = eliminated_image1.convert("RGB")
                eliminated_image1.save(eliminated_output_path)

                
                eliminated_output_path2 = os.path.join(eliminated_output_dir,eliminated_output_name2)
                eliminated_image2 = T.ToPILImage()(eliminated_X)
                eliminated_image2 = eliminated_image2.convert("RGB")
                eliminated_image2.save(eliminated_output_path2)
                
        
# 加载图片并转换为 NumPy 数组
def load_image(path):
    img = Image.open(path)
    img = img.resize((512, 512)).convert('RGB')
    return np.array(img).astype(np.float32)/2

# 定义目标函数
def optimize_scale_and_offset(A, B):
    # 展平图片像素
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # 构建矩阵 [B, 1]
    X = np.vstack([B_flat, np.ones_like(B_flat)]).T
    
    # 通过最小二乘法求解 s 和 o
    params, _, _, _ = np.linalg.lstsq(X, A_flat, rcond=None)
    s, o = params
    return s, o
"""""
# 加载图片A和B
A = load_image('imageA.png')
B = load_image('imageB.png')

# 计算最佳的比例缩放和偏移
s, o = optimize_scale_and_offset(A, B)
print(f"最佳缩放因子: {s}, 最佳偏移量: {o}")

# 应用变换 f(B) = s * B + o
B_transformed = s * B + o

# 保存结果图片
B_transformed_img = Image.fromarray(np.clip(B_transformed, 0, 255).astype(np.uint8))
B_transformed_img.save('B_transformed.png')
"""""
if __name__ == "__main__":
    main()