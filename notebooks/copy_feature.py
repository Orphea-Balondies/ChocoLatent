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
import lpips as LPIPS
from decimal import Decimal
import os
import torchvision.transforms as T
init_image_dir = '../init_images/mix-27'
featurefake_image_dir = '../output/FeatureFakeNolim15000it4step'
eliminated_image_dir = '../eliminated/FeatureFakeNolim15000it4step'
randomnoise_output_root = '../eliminated/randomnoise/'
randomnoise_output_dir = os.path.join(randomnoise_output_root,os.path.basename(eliminated_image_dir))
outexcel_name = f'{os.path.basename(eliminated_image_dir)}_compare.xlsx'
os.makedirs(randomnoise_output_dir, exist_ok=True)
import numpy as np
from PIL import Image

def main():
    model_id_or_path = "/mnt/algorithm/user_dir/zhangjianwei/photoguard/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float32,
    )
    model = pipe.vae.encode
    loss_fn_alex = LPIPS.LPIPS(net='alex' )
    xlsx_list = []
    for image_name in os.listdir(init_image_dir):
        init_image_path = os.path.join(init_image_dir, image_name)
        init_X = load_image(init_image_path)
        #xlsx_list.append(init_row)
        image_index = image_name.split('.')[0]
        for eliminated_image_name in os.listdir(eliminated_image_dir):
            if image_index == eliminated_image_name[11:len(image_index)+11] and eliminated_image_name[len(image_index)+11]=='_':
                randomnoise_output_name = f"randomnoise_{eliminated_image_name}"
                eliminated_image_path = os.path.join(eliminated_image_dir, eliminated_image_name)
                eliminated_X = load_image(eliminated_image_path)

                epst = (eliminated_X.max() - eliminated_X.min())/4
                randomnoise_X = (np.random.rand(*init_X.shape).astype(np.float32) * epst - epst / 2) 
                randomnoise_X = (randomnoise_X - randomnoise_X.min()) / (randomnoise_X.max() - randomnoise_X.min())

                randomnoise_output_path = os.path.join(randomnoise_output_dir,randomnoise_output_name)
                randomnoise_image = T.ToPILImage()(randomnoise_X)
                randomnoise_image = randomnoise_image.convert("RGB")
                randomnoise_image.save(randomnoise_output_path)
                init_torch = torch.from_numpy(init_X[None].transpose(0, 3, 1, 2))
                randomnoise_torch = torch.from_numpy(randomnoise_X[None].transpose(0, 3, 1, 2))
                eliminated_torch = torch.from_numpy(eliminated_X[None].transpose(0, 3, 1, 2))

                #lpips = loss_fn_alex(init_torch, randomnoise_torch).item()
                gap2noise = (model(init_torch).latent_dist.mean-model(randomnoise_torch).latent_dist.mean).norm().item()  
                gap2black = model(init_torch).latent_dist.mean.norm().item()
                gap2tgt = (model(init_torch).latent_dist.mean-model(eliminated_torch).latent_dist.mean).norm().item()  

                row = {
                    "image_name": image_name,
                    #"lpips": lpips,
                    "gap2noise": gap2noise,
                    "gap2black": gap2black,
                    "gap2tgt": gap2tgt
                }
                xlsx_list.append(row)
                df = pd.DataFrame(xlsx_list)
                # 写入 Excel 文件
                df.to_excel(os.path.join(randomnoise_output_dir, outexcel_name), index=False, engine='openpyxl')
        
# 加载图片并转换为 NumPy 数组
def load_image(path):
    img = Image.open(path)
    img = img.resize((512, 512)).convert('RGB')
    return np.array(img).astype(np.float32)

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