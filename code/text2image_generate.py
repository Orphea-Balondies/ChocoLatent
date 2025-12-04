import os
import torch
from diffusers import AutoPipelineForText2Image
from typing import List
import random

class DiffusionT2I:
    """
    单次加载基础模型，支持多次调用。
    对于每个 seed,每个 LoRA 都会生成 generate_num 张图片。
    """

    def __init__(
        self,
        base_model_path: str,
        torch_device: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch_device

        print(f"[Init] Loading base model from {base_model_path} ...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )
        self.pipe.to(torch_device)
        self.pipe.set_progress_bar_config(disable=True)

        print("[Init] Model loaded and ready.\n")

    # =====================================================
    # 生成接口：可被外部反复调用
    # =====================================================
    def generate(
        self,
        prompt: str,
        lora_paths: List[str],
        generate_num: int,
        output_root: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ):
        """
        prompt: 文本提示词
        lora_paths: 多个LoRA路径
        generate_num: 每个LoRA生成多少张
        seed: 用于保证不同LoRA对同一个种子可复现
        output_root: 输出根目录
        """

        seeds = [random.randint(0, 2**32 - 1) for _ in range(generate_num)]
        print("Generated seeds:", seeds)

        # ---- 3. 遍历 LoRA ----
        for lora_path in lora_paths:

            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            save_dir = os.path.join(output_root, lora_name)
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n>>> Loading LoRA: {lora_name}")
            self.pipe.load_lora_weights(lora_path)

            # ---- 4. 对每个种子生成 1 张图 ----
            for seed in seeds:
                generator = torch.Generator(device=self.device).manual_seed(seed)

                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    generator=generator,
                ).images[0]

                save_path = os.path.join(save_dir, f"seed_{seed}.png")
                image.save(save_path)
                print(f"Saved: {save_path}")
                
        print("\n[DONE] All images generated.\n")

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="model/stable-diffusion-v1-5")
    parser.add_argument("--generate_num", type=int, default=10)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--loras", type=str, nargs="+", required=True)
    parser.add_argument("--prompt", default=None)

    args = parser.parse_args()

    prompt = args.prompt if args.prompt else args.loras[0].split("/")[-2]
    output_root = args.output_root if args.output_root else os.path.dirname(args.loras[-1]).replace('lora','generated_images')
    os.makedirs(output_root,exist_ok=True)

    t2i = DiffusionT2I(base_model_path=args.model_path)
    t2i.generate(
        prompt=prompt,
        lora_paths=args.loras,
        generate_num=args.generate_num,
        output_root=output_root,
    )