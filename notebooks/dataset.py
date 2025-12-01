import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from utils import PreprocessTransform

init_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    PreprocessTransform()
])

class CLDataset(Dataset):
    def __init__(self, image_dir, image_list, transform=init_transform):
        """
        初始化数据集，加载所有图片的路径和标签（假设图片的文件名是标签的一部分）。

        :param img_dir: 图片所在的文件夹路径
        :param transform: 数据预处理和增强的变换（比如 ToTensor, Normalize 等）
        """
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        # 返回数据集的总长度
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        根据索引返回一个数据样本和标签。

        :param idx: 数据索引
        :return: 图片和对应的标签
        """
        image_name = self.image_list[idx]["image_name"]
        image_path = os.path.join(self.image_dir,image_name)
        image = Image.open(image_path).convert("RGB")  # 读取图片
        # 应用数据变换
        Xs = self.transform(image)
        image_args = {"image_name":image_name}
        for arg in ["SEED","prompt","strength","guidance_scale"]:
            if arg in self.image_list[idx].keys():
                image_args[arg] = self.image_list[idx][arg]

        return Xs, image_args

