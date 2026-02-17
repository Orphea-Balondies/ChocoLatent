# ChocoLatent 项目代码说明（基于潜空间对抗防护协议 v0.2）

本文档用于把 `exp_plan/experiment_protocol_latent_protection.md` 的实验协议映射到当前代码实现，帮助快速理解项目结构、运行流程和关键扩展点。

## 1. 项目定位

本项目实现的是“防护者在发布前对训练图像加扰动，攻击者再进行 LoRA 微调”的统一实验框架。核心目标是对不同保护方法在统一预算下进行可复现实验与对比。

协议入口文档：

- `exp_plan/experiment_protocol_latent_protection.md`

当前代码重点已覆盖：

- 保护图生成主流程（阶段 A 可直接跑）
- 多方法统一优化接口（`chocolatent / glaze / photoguard / robust-ldm / mist`）
- 统一输出结构（`manifest.csv + metrics.json + per_image.csv + config.yaml + logs`）

## 2. 目录与职责

### 2.1 核心代码目录

- `code/distribution_adv_tgt.py`
  - 保护图批处理主入口（可视为 `run_protect` 的核心实现）。
  - 负责参数解析、预算网格遍历、调用 PGD、保存图片与指标。
- `code/pgd.py`
  - 对抗优化核心模块。
  - 负责预算约束（L2/LPIPS）、方法目标函数、优化循环与最终指标计算。
- `code/dataset.py`
  - 读取输入图像目录并做基础预处理（resize/crop/归一化到 `[-1,1]`）。
- `code/utils.py`
  - 预处理函数与 step 级指标收集器（TensorBoard 用）。

### 2.2 实验协议目录

- `experiments/scripts/run_protect.py`
  - 协议风格入口，内部调用 `code/distribution_adv_tgt.py`。
- `experiments/scripts/run_lora_train.py`
  - 封装 LoRA 训练脚本调用。
- `experiments/scripts/run_generate_eval.py`
  - 封装生成评测脚本调用。
- `experiments/scripts/run_metrics.py`
  - 计算配对生成差距指标 `PGG_lpips` 与 `PGG_clip`。
- `experiments/scripts/run_conclusion_metrics.py`
  - 计算协议结论指标（`PGG/MLS/PG/CLIP-T/QRR`，可选 `FID`）。
- `experiments/scripts/run_robustness.py`
  - 生成鲁棒性变换数据（JPEG/resize/crop/noise）。
- `experiments/scripts/run_analysis.py`
  - 聚合各 run 的 `metrics.json` 为汇总表。
- `experiments/scripts/run_full_pipeline.py`
  - 一键全流程总控：保护图 -> clean/adv LoRA 微调 -> 多 prompt+seed 生成 -> 指标计算与汇总。
- `experiments/configs/`
  - 方法、数据、阶段 A sweep 的示例配置模板。

### 2.3 兼容脚本目录

- `scripts/adv_protect.sh`
  - 命令行一键保护图入口，默认按阶段 A 网格运行。

## 3. 保护流程（对应协议第 5 节）

### 3.1 入口与参数层

`code/distribution_adv_tgt.py` 的 `parse_args()` 定义了协议需要的主要参数：

- 实验元信息：`method/stage/dataset_type/task_type/src_model/tgt_model/...`
- 预算：`budget_l2`、`budget_lpips` 及网格 `budget_l2_grid`、`budget_lpips_grid`
- 优化参数：`iters/initial_lr/eps/strict_lpips_projection/...`

关键位置：

- `code/distribution_adv_tgt.py:75`

### 3.2 目录与文件产出层

`prepare_protocol_dirs()` 会创建统一输出目录：

- `protected_images/`
- `ft_clean/`
- `ft_adv/`
- `generated_clean/`
- `generated_adv/`
- `edited_clean/`
- `edited_adv/`
- `metrics/`
- `logs/`

关键位置：

- `code/distribution_adv_tgt.py:222`

### 3.3 核心运行层

`main()` 的逻辑是：

1. 解析预算网格并构造 `exp_id`
2. 加载数据集与 VAE
3. 对每个 `(budget_l2, budget_lpips)` 组合生成 `run_id`
4. 调用 `pgd(...)` 生成 `x_adv`
5. 保存 `protected_images`
6. 汇总并写出每个 run 的 `per_image.csv / metrics.json / config.yaml`
7. 最后写全局 `manifest.csv`

关键位置：

- `code/distribution_adv_tgt.py:341`
- `code/distribution_adv_tgt.py:530`
- `code/distribution_adv_tgt.py:587`

## 4. 对抗优化模块（对应协议第 2/6/7 节）

### 4.1 统一预算约束

`pgd()` 中统一支持：

- `L2` 预算（按 `||delta||2 / sqrt(d)` 归一化口径）
- `LPIPS` 预算（可启用严格投影）
- 可选 `L_inf` 限制（`eps`）

关键位置：

- `code/pgd.py:200`
- `code/pgd.py:41`
- `code/pgd.py:65`

### 4.2 方法目标函数

`_attack_objective()` 把 5 种方法统一到一个接口：

- `chocolatent`：最大化 `LPIPS(x, decode(encode(x_adv)))`
- `robust-ldm`：最大化潜变量距离
- `photoguard`：把解码结果拉向灰图目标
- `mist`：把解码结果拉向指定目标图
- `glaze`：先生成 `style_target`（默认 ONNX `mosaic-9`，也支持 AdaIN），再让解码结果对齐 `style_target`

关键位置：

- `code/pgd.py:116`

### 4.3 最终指标回传

`pgd(..., return_info=True)` 会返回每张图指标，供主流程直接落盘：

- 输入侧：`input_l2/input_l2_normed/input_linf/input_lpips/input_psnr/input_ssim`
- 解码侧：`decoded_lpips/decoded_l2/decoded_psnr/decoded_ssim`
- 潜空间：`latent_l2`
- 目标分数：`attack_score`

关键位置：

- `code/pgd.py:145`

## 5. 指标与结果文件

### 5.1 每个 run 的输出

在 `experiments/outputs/<exp_id>/metrics/<run_id>/` 下：

- `per_image.csv`：逐图指标
- `metrics.json`：均值、标准差、95% CI 等汇总
- `config.yaml`：该 run 参数快照（目前用 JSON 语法写入，兼容 YAML 读取）

### 5.2 全局输出

在 `experiments/outputs/<exp_id>/` 下：

- `config.yaml`：实验总配置
- `manifest.csv`：协议定义的核心元数据字段

字段定义位置：

- `code/distribution_adv_tgt.py:28`
- `code/distribution_adv_tgt.py:52`

## 6. 常用运行方式

### 6.1 阶段 A 保护图（推荐）

```bash
python experiments/scripts/run_protect.py \
  --model_path model/stable-diffusion-v1-5 \
  --image_root init_images \
  --image_dirname lego-minifigure-faces \
  --method chocolatent \
  --budget_l2_grid 4/255,8/255,12/255 \
  --budget_lpips_grid 0.1,0.2,0.5 \
  --output_root experiments/outputs \
  --exp_id stageA-demo
```

### 6.2 Shell 一键入口

```bash
bash scripts/adv_protect.sh
```

可通过环境变量覆盖关键参数，例如：

- `METHOD=glaze`
- `GLAZE_STYLE_BACKEND=onnx_mosaic`（默认）
- `GLAZE_STYLE_ONNX_PATH=model/style_transfer/mosaic-9.onnx`
- `GLAZE_STYLE_BACKEND=adain` 时才需要 `GLAZE_STYLE_IMAGE_PATH=/abs/path/to/style_ref.png`
- `GLAZE_STYLE_ALPHA=1.0`（仅 adain 使用）
- `BUDGET_L2_GRID=8/255`
- `BUDGET_LPIPS_GRID=0.2`
- `EXP_ID=my-exp`

### 6.3 全流程一键入口

```bash
python experiments/scripts/run_full_pipeline.py \
  --exp_id full-demo \
  --image_root init_images \
  --image_dirname lego-minifigure-faces \
  --glaze_style_backend onnx_mosaic \
  --glaze_style_onnx_path model/style_transfer/mosaic-9.onnx \
  --prompt_file exp_plan/prompts_style.txt \
  --seed_file exp_plan/seeds_eval.txt
```

## 7. 扩展建议（下一步研发）

1. 在 `experiments/scripts/run_metrics.py` 增补 `MLS/PG/QRR/TR/SR_T/AUC`。
2. 增加阶段 B/C 任务编排脚本，实现跨模型迁移与鲁棒性曲线全自动评估。
3. 增加统计检验模块（paired t-test/Wilcoxon、Holm-Bonferroni、效应量）。
4. 将 `config.yaml` 写入方式切换为原生 YAML（当前为 JSON 兼容格式）。
5. 打通 `generated_clean/generated_adv` 与下游论文图表脚本，形成一键复现实验链路。
