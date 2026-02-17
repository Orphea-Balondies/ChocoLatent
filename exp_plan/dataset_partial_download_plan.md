# 部分下载计划（WikiArt / CustomConcept101 / Person）

更新时间：2026-02-09  
目标：在可控存储下构建三类子集，优先高频/高使用组，满足每组约 25-40 张。

## 1. 目标配额

- `WikiArt`：15 组（画风 style），每组 40 张（偏大，满足“同画风数据量更大”）。
- `Concept`（CustomConcept101）：15 组，每组 30 张。
- `Person`：15 组，每组 30 张。
- 全局控制：每个子任务设置 `max-output-gb`，超限自动报错并停止。

## 2. 数据源可得性与策略

### 2.1 WikiArt

- 使用 HuggingFace `huggan/wikiart` 的本地 parquet 缓存分片进行抽样，不下载全量 30GB+。
- 分组优先级：按已加载分片中的 `style` 样本数降序（高频优先）。
- 落地方式：解码 parquet 中 `image.bytes`，统一保存为 JPEG（可缩放长边，默认 768）。

### 2.2 CustomConcept101（Concept）

- 官方来源：Custom Diffusion 仓库给出的 `benchmark_dataset.zip`（Google Drive，约 3.39GB）。
- 原始数据每概念常见 3-15 张，无法直接满足 25-40。
- 处理策略：按 `class_prompt` 聚合多个 instance 子目录，同类合并后再按组抽样到 30 张。
- 分组优先级：先按“可用实例数”再按“zip 内可用图片总数”排序，优先高频概念类。

### 2.3 Person

- 论文描述（CopyrightMeter）采用：`10 celebrities x 15 images`，来自 LAION。
- 当前公开仓库中未找到可直接一键下载的官方 person bundle 链接。
- 因此脚本支持两种模式：
  - `official_laion`：输入 `person,url` CSV 清单后自动下载（用于你后续补全官方 URL 时）。
  - `hf_rows_lfw`（fallback，默认）：使用 HuggingFace datasets-server 的 LFW 行接口，直接按身份频次抽样并下载图片 URL。
  - `lfw_parquet`：使用 HuggingFace LFW parquet 本地文件模式。
  - `lfw`：原始 LFW tgz 方式（官网链路在某些网络下可能不可达）。

## 3. 实现文件

- 脚本：`scripts/prepare_partial_datasets.py`
- 输出：
  - `manifest.csv`：逐图来源与输出路径
  - `group_summary.csv`：每组配额与实际样本数
  - `summary.json`：任务汇总、大小、参数

## 4. 推荐执行命令

在仓库根目录执行：

```bash
# 1) WikiArt（style 组更大）
python scripts/prepare_partial_datasets.py wikiart \
  --target-groups 15 \
  --images-per-group 40 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 4 \
  --out-root datasets/partial/WikiArt

# 2) Concept（官方 zip，不全量展开，仅抽取目标组）
python scripts/prepare_partial_datasets.py concept \
  --download-if-missing \
  --target-groups 15 \
  --images-per-group 30 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 6 \
  --out-root datasets/partial/Concept

# 3) Person（推荐 fallback：hf_rows_lfw）
python scripts/prepare_partial_datasets.py person \
  --source hf_rows_lfw \
  --target-groups 15 \
  --images-per-group 30 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 4 \
  --out-root datasets/partial/Person
```

如果你希望 `WikiArt` 改成“单一高热度组，仅 40 张（适合小样本 diffusion 训练）”，用：

```bash
python scripts/prepare_partial_datasets.py wikiart \
  --group-by artist \
  --target-groups 1 \
  --images-per-group 40 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 1 \
  --out-root datasets/partial/WikiArt_hot40_top1_artist
```

如需指定固定组（例如 van gogh）：

```bash
python scripts/prepare_partial_datasets.py wikiart \
  --group-by artist \
  --focus-group vincent-van-gogh \
  --target-groups 1 \
  --images-per-group 40 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 1 \
  --out-root datasets/partial/WikiArt_hot40_vangogh
```

如果你后续拿到 LAION person URL 清单：

```bash
python scripts/prepare_partial_datasets.py person \
  --source official_laion \
  --laion-manifest /path/to/person_urls.csv \
  --target-groups 15 \
  --images-per-group 30 \
  --images-min 25 \
  --images-max 40 \
  --max-output-gb 4 \
  --out-root datasets/partial/Person
```

## 5. 与你给定论文的对齐说明

- `CopyrightMeter (2024)`：明确提到评测使用 `WikiArt / CustomConcept101 / Person`，其中 Person 为 LAION 中 10 名人、每人 15 张。
- `Custom Diffusion (CVPR 2023)`：提供 CustomConcept101 的官方下载入口与评测 prompt。
- `IMPRESS (NeurIPS 2024)`：作为抗净化攻击评测可直接接入现有子集，不影响本下载流程。
- `Robustness of Latent Diffusion (2023)` 与 `ImperceptibleProtectionStyle (2025)`：可在本子集上按你现有实验协议复用评测脚本。

## 6. 存储估算（经验值）

- WikiArt 子集（15x40，JPEG 768）：约 1.2-2.5GB
- Concept 子集（15x30，JPEG 768）：约 0.8-2.0GB
- Person 子集（15x30，JPEG 768）：约 0.6-1.5GB
- 原始下载额外：
  - CustomConcept101 官方 zip：约 3.39GB
  - LFW tgz：约 180MB（量级）

## 7. 本机当前实测（2026-02-09）

- 已完成：
  - `WikiArt`：15 组，565 张（`datasets/partial/WikiArt`）
  - `WikiArt_hot40_top1_artist`：1 组，40 张，组名 `vincent-van-gogh`（`datasets/partial/WikiArt_hot40_top1_artist`）
  - `Concept`：12 组，358 张（`datasets/partial/Concept`）
- 已下载：
  - `CustomConcept101` 官方 zip：`datasets/raw/customconcept101/benchmark_dataset.zip`
- 待补：
  - `Person`：在当前网络环境下访问 `datasets-server.huggingface.co` 存在间歇性 SSL/502，需重试命令或切换网络后执行。
