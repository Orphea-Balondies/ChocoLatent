from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from dataclasses import dataclass
from typing import Dict

totensor = T.ToTensor()
topil = T.ToPILImage()


@dataclass
class PGDStepResult:
    loss: float
    grad_norm: float
    step_size: float
    iteration: int
    data_idx: int

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image)
    return  2*image - 1.0

class PreprocessTransform:
    def __call__(self, image):
        image_preprocessed = preprocess(image)
        return image_preprocessed

class StepLossCollector:
    """Step级别的Loss收集器"""
    def __init__(self):
        self.step_metrics = {}     # {step: {metric_name: [values]}}

    def record_step(self, step: int, metrics: Dict = None):
        """记录每个step的loss和指标"""
        # 记录其他指标
        if metrics:
            if step not in self.step_metrics:
                self.step_metrics[step] = {}

            for metric_name, value in metrics.items():
                if metric_name not in self.step_metrics[step]:
                    self.step_metrics[step][metric_name] = []
                self.step_metrics[step][metric_name].append(value)

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

                                                                                                            
