import os
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor
import timm
from timm.data import create_transform, resolve_data_config
from huggingface_hub import hf_hub_download
from typing import Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub.utils import HfHubHTTPError
import pandas as pd

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data

def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels

class TaggerInferencer:
    """
    加载一次模型，重复用来对任意文件夹图片进行推理。
    """

    def __init__(self, model_path, torch_device=None):
        """
        model_path: 模型路径
        torch_device: torch.device 类型（None 时自动检测）
        """
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch_device

        # ---- 模型加载（只执行一次） ----
        print(f"[Init] Loading model from {model_path} ...")
        self.model = timm.create_model("hf-hub:SmilingWolf/wd-vit-tagger-v3", checkpoint_path = "/root/chocolatent/model/wd-vit-tagger-v3/model.safetensors").eval()

        if self.device.type != "cpu":
            self.model = self.model.to(self.device)

        final_transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        # 预处理可提前准备（也只做一次）
        self.transform = transforms.Compose([
            pil_ensure_rgb,
            pil_pad_square,
            final_transform
        ])
        self.labels = load_labels_hf(repo_id="SmilingWolf/wd-vit-tagger-v3")
        self.gen_threshold: float = 0.35
        self.char_threshold: float = 0.75

        print("[Init] Model loaded and ready.\n")

    # -------------------------------------------------------
    # 这里是可以被外部代码反复调用的推理函数
    # -------------------------------------------------------
    def infer_folder(self, folder_path:Path, batch_size=4):
        """
        对 folder_path 下的所有图片做推理
        """

        # 获取要处理的图片
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if not image_files:
            print(f"[Warn] No images found in {folder_path}")
            return

        image_files.sort()
        metadata_file = os.path.join(folder_path, "metadata.jsonl")

        print(f"[Infer] Processing folder: {folder_path}")
        print(f"[Infer] Total images: {len(image_files)}\n")

        special_label = str(folder_path.stem)

        with open(metadata_file, "w", encoding="utf-8") as f_out:
            with torch.inference_mode():

                for i in tqdm(range(0, len(image_files), batch_size)):
                    batch_files = image_files[i:i + batch_size]

                    # 加载 batch
                    imgs = []
                    for name in batch_files:
                        img = Image.open(os.path.join(folder_path, name)).convert("RGB")
                        imgs.append(self.transform(img))

                    inputs = torch.stack(imgs).to(self.device)

                    # 推理
                    outputs = self.model.forward(inputs)
                    outputs = F.sigmoid(outputs).to("cpu")

                    # 写 jsonl
                    for name, out in zip(batch_files, outputs):
                        caption, taglist, ratings, character, general = get_tags(
                            probs=out,
                            labels=self.labels,
                            gen_threshold=self.gen_threshold,
                            char_threshold=self.char_threshold,
                        )
                        caption = special_label + ", " + caption
                        f_out.write(json.dumps({
                            "file_name": name,
                            "caption": caption
                        }, ensure_ascii=False) + "\n")

        print(f"\n[Infer] metadata.jsonl saved to: {metadata_file}\n")



# ================================================
# 可选：添加命令行接口
# ================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="model/wd-vit-tagger-v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_root", type=str, default="init_images")
    parser.add_argument("--image_dirs", type=str, nargs="+", required=True)

    args = parser.parse_args()

    infer = TaggerInferencer(
        model_path=args.model_path
    )
    for image_dir in args.image_dirs:
        folder_path = Path(args.image_root).joinpath(image_dir)
        infer.infer_folder(folder_path)