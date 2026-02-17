# 潜空间对抗防护统一实验协议（v0.2）

更新时间：2026-02-10  
用途：作为后续实验代码设计与实现的唯一上游文档（先定义协议，再写代码）。

## 1. 目标与研究问题

本实验围绕 5 类潜空间扰动方案进行统一评测：`Glaze`、`PhotoGuard`、`On the Robustness of Latent Diffusion Models`（下文简称 `Robust-LDM`）、`Mist`、`ChocoLatent`。  
对应目标设定：

1. `Glaze`：先生成“新画风目标图”，再最小化受扰动图与该目标图的潜空间距离（targeted latent alignment to stylized target）。
2. `PhotoGuard`：使用“纯灰目标图”，最小化受扰动图与灰目标图的潜空间距离（targeted latent alignment to gray target）。
3. `Robust-LDM`：最大化受扰动图到原图片的潜空间距离（targeted latent alignment to robust target）。
4. `Mist`：使用“指定特殊图片”作为目标图，最小化受扰动图与目标图的潜空间距离（targeted latent alignment to specified target image）。
5. `ChocoLatent`：直接优化解码后像素空间的破坏目标（pixel-space degradation objective after decoding）。

说明：除 `ChocoLatent` 外，`Glaze/PhotoGuard/Robust-LDM/Mist` 的优化核心均为“拉近潜空间距离到方法对应目标图”。

核心问题：

1. 施加保护后，攻击者用受保护图微调 LoRA / DreamBooth，是否显著降低模仿/再生成能力？
2. 受保护图微调出来的模型，其生成质量下降多少？
3. 保护扰动本身对原图破坏程度有多大？
4. 不同数据类型下，各方法表现是否稳定？
5. 不同预算组合（`L2` 预算 × `LPIPS` 预算）下，效果-失真权衡如何变化？
6. 扰动在 SD v1.5 与 SDXL 间是否具备鲁棒性与迁移性？
7. 同一扰动噪声（同一保护样本）在不同 diffusion 模型上是否仍有效？
8. 对“恶意编辑”（而非仅风格模仿）的防护是否同样有效？

## 2. 统一威胁模型与对照设置

### 2.1 攻防设定

1. 防护者可在发布前修改训练图像，得到受保护图 `x_adv`。
2. 攻击者获取公开图像后，仅做常规 LoRA / DreamBooth 微调，不改训练代码。
3. 评测时对比两条平行链路：
   - 干净链路：`原图 -> LoRA_clean -> 生成图 G_clean`
   - 防护链路：`受保护图 -> LoRA_adv -> 生成图 G_adv`
4. 两条链路使用相同 base model、相同微调配置、相同 prompts、相同 seeds。

### 2.2 统一优化模板（用于代码接口）

统一约束：

$$
\|\delta\|_2\le B_2,\ \mathrm{LPIPS}(x,x+\delta)\le B_p
$$

其中 `B_2` 为 `L2` 扰动预算，`B_p` 为 `LPIPS` 感知预算。

对方法分两类定义目标：

1. 潜空间目标对齐类（`Glaze/PhotoGuard/Robust-LDM/Mist`）：

$$
\delta_m^*=\arg\min_{\delta}\ \left\|E(x+\delta)-E(t_m)\right\|_2^2
$$

其中 `E` 为潜空间编码器，`t_m` 为方法 `m` 的目标图（如风格目标图、灰目标图、鲁棒目标图、指定目标图）。

2. 像素空间破坏类（`ChocoLatent`）：

$$
\delta_{\text{choco}}^*=\arg\max_{\delta}\ \mathcal{L}_{pixel}\!\left(D(E(x+\delta)),x\right)
$$

其中 `D` 为解码器，`\mathcal{L}_{pixel}` 为像素/感知破坏目标。

## 3. 实验因素与矩阵

### 3.1 主因素

1. 方法 `M`：`{Glaze, PhotoGuard, Robust-LDM, Mist, ChocoLatent}`。
2. `L2` 预算 `B2`：`{4/255, 8/255, 12/255}`（按归一化 `\|\delta\|_2/\sqrt{d}` 口径）。
3. `LPIPS` 预算 `Bp`：`{0.1, 0.2, 0.5}`。
4. 数据类型 `D`：见第 4 节。
5. 任务类型 `Task`：`{风格模仿, 恶意编辑}`。
6. 攻击者微调方式 `FT`：`{LoRA, DreamBooth（暂时不做）}`。
7. 模型版本：
   - 保护生成模型 `S_src ∈ {SD1.5, SDXL}`
   - 攻击微调/评测模型 `S_tgt ∈ {SD1.5, SDXL}`
8. 随机重复 `seed`：建议 `3~5` 次。（暂时不做）

### 3.2 三阶段执行（避免组合爆炸）

1. 阶段 A（筛选）：固定 `D` 的小样本，只跑 `B2 x Bp`，选每个方法前 2 组配置。
2. 阶段 B（主实验）：在全部数据类型、全部模型组合上跑“方法主对比 + 最优配置”。
3. 阶段 C（鲁棒性）：对阶段 B 最优配置做跨模型迁移与噪声扰动存活性评测。

## 4. 数据设计与分析分层（暂时不去详细研究）

建议至少 4 类数据，每类单独统计，再做整体汇总：

1. `D1 主体一致型`：同一主体多视角（人脸/角色/物体），用于主体再生成评测。
2. `D2 风格一致型`：不同内容同风格，适合评测风格模仿防护（Glaze/Mist）。
3. `D3 纹理高频型`：细节多、纹理密（油画笔触/织物/复杂背景），用于检验扰动存活性。
4. `D4 自然场景型`：普通摄影内容，检验方法泛化与视觉可用性。

每类数据建议统一为同规模子集（例如每类 `K` 个概念，每概念 `N_train/N_eval` 固定），避免数据量本身成为混杂因素。

数据结论分析方式：

1. 先做类内比较：同一 `D` 下比较 5 种方法。
2. 再做类间比较：同一方法在 `D1~D4` 的性能波动。
3. 做交互分析：`方法 × 数据类型` 是否显著交互。
4. 任务分层：`Task=风格模仿` 与 `Task=恶意编辑` 必须分开统计与报告。

## 5. 统一实验流程（可直接映射脚本）

对每个实验单元 `(M, B2, Bp, D, Task, FT, S_src, S_tgt, seed)`：

1. 生成受保护图：`X_adv = protect(M, X_clean, config)`。
2. 训练基线微调模型：`FT_clean <- X_clean`（`FT ∈ {LoRA, DreamBooth}`）。
3. 训练防护微调模型：`FT_adv <- X_adv`。
4. 统一生成评测图：
   - 风格模仿：同 prompts 集 `P_eval_style`，同随机种子集 `Z_eval`，得到 `G_clean` 与 `G_adv`
   - 恶意编辑：同编辑提示词与 mask 集 `P_eval_edit/M_eval`，得到 `E_clean` 与 `E_adv`
5. 计算指标并落盘：
   - 防护有效性指标
   - 扰动失真指标
   - 鲁棒性指标
6. 输出单元结果：`metrics.json + per_image.csv + config.yaml + logs`。

## 6. 指标体系

### 6.1 防护效果（核心）

#### A. 双模型生成差距（你提出的核心口径）

同 prompt、同 seed 下配对比较：

$$
\mathrm{PGG}_{lpips}=\frac{1}{N}\sum_i \mathrm{LPIPS}(g_i^{clean}, g_i^{adv})
$$

$$
\mathrm{PGG}_{clip}= \frac{1}{N}\sum_i \left(1-\cos(\phi(g_i^{clean}),\phi(g_i^{adv}))\right)
$$

`PGG` 越大，说明“原图微调模型”和“受保护图微调模型”的行为差异越大。

#### B. 模仿泄漏分数（建议主指标）

$$
\mathrm{MLS}(G,R)=\frac{1}{|G|}\sum_{g\in G}\max_{r\in R}\cos(\psi(g),\psi(r))
$$

其中 `R` 是受保护对象参考图集。定义：

$$
\mathrm{PG}=\mathrm{MLS}(G_{clean},R)-\mathrm{MLS}(G_{adv},R)
$$

`PG` 越大，防护越强（微调后更难模仿原对象/风格）。

与常见文献命名对齐（精简口径）：

1. `CLIP-I`：图像-图像相似度。在本协议中可由 `MLS` 直接实现；当 `\psi` 取 CLIP image encoder 时，`MLS` 即 `CLIP-I` 检索式泄漏分数。主文优先报告 `PG`，避免与 `MLS/CLIP-I` 重复。
2. `CLIP-T`：文本-图像对齐度，用于检查防护是否过度破坏语义跟随能力（见 C，`CLIP-T` 即 `CLIPScore`）。

针对 `Task=风格模仿`，在 `PG` 之外补充以下指标（与 B 同属“模仿防护效果”，不再单独拆分）：

1. `Diffusion-CLS-Acc@Top3`：基于 diffusion classifier 的风格分类准确率（越低越好）。  
2. `FID`（风格任务可记作 `FID_style`）：模仿生成分布相对参考风格分布的偏离；建议同时报告 `\Delta FID = FID_{adv} - FID_{clean}`，便于直接比较防护增益。

说明：上述指标与 `PG` 共同构成“模仿防护效果”口径，可避免仅靠通用 CLIP/LPIPS 导致“风格是否被学到”判断不充分。

#### C 受保护图微调模型的生成质量

在 `G_adv` 上报告：

1. 文本对齐：`CLIP-T`（即 `CLIPScore(prompt, image)`）。
2. 感知质量：`MUSIQ` 或 `NIQE`（二选一，优先选计算简单的实现）。

相对质量保持率：

$$
\mathrm{QRR}=\frac{Q_{adv}}{Q_{clean}}
$$

其中 `Q` 为“越大越好”指标（如 CLIPScore、MUSIQ）。  
若用 NIQE（越小越好），则改为 `QRR = NIQE_clean / NIQE_adv`。

对 `Task=恶意编辑`，额外报告：

1. `Edit-SSIM(E_adv,E_clean)`（越高越接近无保护编辑结果）。
2. `Edit-PSNR(E_adv,E_clean)`（越高越好）。
3. `Edit-VIFp(E_adv,E_clean)`（越高越好）。

### 6.2 扰动对原图破坏程度（输入侧损失）

在 `X_clean` 与 `X_adv` 间计算：

1. `L2`, `L_inf` 扰动强度。
2. `PSNR`, `SSIM` 像素一致性。
3. `LPIPS` 感知差异。
4. 可选：`DISTS`（若算力允许）。

建议将 `SSIM/LPIPS` 作为论文主文口径，`L2/L_inf/PSNR` 放附录支撑。

### 6.3 鲁棒性指标

#### A. 跨模型迁移鲁棒性（SD1.5/SDXL）

四个方向：`1.5->1.5`、`1.5->XL`、`XL->1.5`、`XL->XL`。  
定义迁移保持率：

$$
\mathrm{TR}=\frac{PG_{cross}}{PG_{intra}}
$$

`TR` 越接近 1 越好。

#### B. 扰动噪声/传播鲁棒性（同一段噪声跨模型）

同一受保护样本在下列变换后再被攻击者使用：

1. JPEG：`q=95/75/50`
2. Resize：`1.0/0.75/0.5`
3. Crop：`0%/5%/10%`
4. Gaussian noise：`\sigma=1/255,2/255,4/255`

定义扰动存活率：

$$
\mathrm{SR}_T=\frac{PG_{after\,T}}{PG_{clean\_pipeline}}
$$

`SR_T` 越高表示扰动越抗现实噪声。

建议将每种变换从“离散点”扩展为“强度曲线”（severity curve），例如：

1. JPEG：`q ∈ {95,85,75,65,55,45}`。
2. Blur：`σ ∈ {0.005,0.01,0.02,0.03}`。
3. Noise：`σ ∈ {1,2,4,6}/255`。
4. Resize：`ratio ∈ {1.0,0.875,0.75,0.625,0.5}`。

每条曲线均报告 AUC（面积）作为单值汇总，减少“只看单点”的偏差。

## 7. 扰动预算组合与基线消融

每个方法都跑 `B2 x Bp` 网格，并画三类曲线：

1. 防护-可感知性帕累托图：`PG` 对 `LPIPS(X_clean,X_adv)`。
2. 防护-生成质量帕累托图：`PG` 对 `QRR`。
3. 鲁棒性强度曲线：`severity` 对 `SR_T`（并报告 AUC）。

新增强基线（必须）：

1. `RandomNoise@MatchedLPIPS`：随机噪声调到与方法相同 LPIPS。
2. `RandomNoise@MatchedPSNR`：随机噪声调到与方法相同 PSNR。

目的：验证“防护增益”来自方法目标优化，而不是“任何同强度噪声”。

推荐报告策略：

1. 每方法给出一个“平衡点配置”（兼顾高 PG 与低输入失真）。
2. 每方法给出一个“高强度配置”（追求最大防护，允许更大代价）。

## 8. 统计分析、预算公平与显著性

### 8.1 预算公平协议（必须执行）

为保证不同方法公平比较，采用“双预算对齐”：

1. 扰动预算对齐：`L2` 预算上限一致，同时将 `LPIPS` 预算控制在同一目标区间（如 `±0.005`）。
2. 计算预算对齐：每方法优化迭代步数与总反向次数对齐，记录 `GPU-hour` 与 wall-clock。

当方法机制差异过大时，采用“分层公平”：

1. `Perceptual-Fair`：按等 LPIPS/PSNR 比较防护效果。
2. `Compute-Fair`：按等 `GPU-hour` 比较防护效果。

两种口径都报告，避免单一口径偏置结论。

### 8.2 统计显著性与效应量

1. 每项配置至少 `3` 次随机重复，报告 `mean ± std` 与 `95% CI`。
2. 两模型配对比较使用 `paired t-test`（非正态时改 `Wilcoxon`）。
3. 多重比较用 `Holm-Bonferroni` 校正。
4. 报告效应量（`Cohen's d` 或 `Cliff's delta`）。
5. 关键主结论只基于显著且效应量足够的结果。

## 9. 代码实现接口草案

建议目录：

```text
experiments/
  configs/
    methods/*.yaml
    datasets/*.yaml
    sweeps/*.yaml
  scripts/
    run_protect.py
    run_lora_train.py
    run_generate_eval.py
    run_metrics.py
    run_robustness.py
    run_analysis.py
  outputs/
    <exp_id>/
      protected_images/
      ft_clean/
      ft_adv/
      generated_clean/
      generated_adv/
      edited_clean/
      edited_adv/
      metrics/
      logs/
```

核心元数据字段（`manifest.csv`）：

`exp_id, method, budget_l2, budget_lpips, dataset_type, dataset_name, task_type, finetune_method, src_model, tgt_model, lora_rank, lora_steps, dreambooth_steps, perturb_budget_l2, perturb_budget_lpips_target, optimize_steps, gpu_hours, seed, prompt_set_id, edit_prompt_set_id, metric_version`

## 10. 论文图表输出模板（第 5/6 章可直接复用）

1. 主结果表：`方法 × 任务类型 × 数据类型 × PG/PGG/FID/QRR`。
2. 鲁棒性表：`方法 × (1.5->XL, XL->1.5) × TR`。
3. 扰动存活表：`方法 × 噪声变换 × SR_T`。
4. 强度曲线图：`severity -> SR_T`（含 AUC）。
5. 消融图：`B2/Bp -> PG, QRR, 输入LPIPS` 三联图。
6. 双目标图：`Utility vs Protection` 帕累托散点图。
7. 随机噪声基线图：`方法 vs Matched Random Noise` 对照柱状图。
8. 可视化案例：`x_clean, x_adv, g_clean, g_adv, e_clean, e_adv` 对齐图。

## 11. 待补充信息清单（你后续提供）

1. 每个数据类型的具体数据集名称与授权范围。
2. 每类数据的概念数 `K`、每概念训练图数 `N_train`、评测图数 `N_eval`。
3. LoRA 与 DreamBooth 训练超参数（rank、lr、steps、batch、文本模板）。
4. Prompt 集大小与构成（风格模仿提示、恶意编辑提示、通用质量提示）。
5. 恶意编辑任务的 mask 生成规则与编辑类型清单（背景替换、局部重绘等）。
6. 指标模型选型细节（风格分类器、CLIP 版本、是否使用 diffusion classifier）。
7. 可用 GPU 资源与可接受总训练时长（用于预算公平中的 Compute-Fair 配置）。

## 12. 执行顺序建议（落地优先级）

1. 先完成阶段 A（小样本 + 参数筛选）。
2. 再完成阶段 B（主结果矩阵）。
3. 最后完成阶段 C（鲁棒性与跨模型迁移）。

这样可以保证先拿到可用结论，再逐步扩展到完整论文实验版图。
