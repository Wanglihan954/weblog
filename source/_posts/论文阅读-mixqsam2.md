---
title: >-
  论文阅读｜Mix-QSAM2: Mixed-Precision Quantization for High Fidelity Segmentation in
  Resource Constrained Scenarios
categories:
  - 文献阅读
  - Tracking
tags:
  - 文献笔记
  - AI论文
  - SAM2
  - 视频目标分割
  - Tracking
description: >-
  SAM2 在精度上树立了新标杆，但计算与显存需求阻碍其在资源受限设备上的部署。本文提出一个统一的"重要性驱动"优化框架：1)
  Importance-driven Mixed-Precision Quantization——用 Weight-Activation Importance
  Score 分析每层对量化的敏感度，按重要性分配位宽，关键层保留高精度；…
readmore: true
mathjax: true
abbrlink: f09a6a41
date: 2026-08-15 20:30:00
updated: 2026-08-15 23:00:00
---
> 本文基于论文、补充材料与公开代码整理。文中的“我的理解”和“批判性思考”属于个人分析；
> 论文插图均来自原论文或补充材料，仅用于学习与讨论。

## 论文信息

**Title:** Mix-QSAM2: Mixed-Precision Quantization for High Fidelity Segmentation in Resource Constrained Scenarios  
**Authors:** Yuzhe Duan, Xuanxuan Ren, Guizhe Dong, Xu Yang, Yanhua Yang  
**Venue:** AAAI 2026  
**DOI:** 10.1609/aaai.v40i5.37374  
**GitHub:** 无官方代码  
**Project Page:** 无  
**IF / CCF:** CCF-A

### 摘要

SAM2 在精度上树立了新标杆，但计算与显存需求阻碍其在资源受限设备上的部署。本文提出一个统一的"重要性驱动"优化框架：1) Importance-driven Mixed-Precision Quantization——用 Weight-Activation Importance Score 分析每层对量化的敏感度，按重要性分配位宽，关键层保留高精度；2) Selective Importance-driven Synthesis (SIS)——当 memory bank 超容量时，用内容相似度识别最冗余的历史帧并把它们合成为单个代表特征，保留信息多样性同时增强时序上下文理解。在 COCO 与 SAV benchmark 上持续超越 SOTA 量化方法，为量化与动态记忆管理的协同设计提供了一条实践路径。

<!-- more -->

---

## 论文资源

- **PDF:** 已导入 Zotero
- **Paper:** DOI 10.1609/aaai.v40i5.37374
- **GitHub:** 无官方代码

---

## 1. 研究动机

### 要解决什么问题？

> SAM2 精度高但计算与显存开销大，无法部署在资源受限设备（边缘端）上；同时其视频推理的 memory bank 随时间累积大量与当前上下文高度相似的冗余特征，造成显存浪费。两个瓶颈需要协同解决。

### 现有方法的问题

- 量化方面：现有 PTQ 方法（PTQ4SAM 等）多用**均匀位宽**（全部 6-bit 或 4-bit）或静态混合精度，忽略不同层对量化误差敏感度差异极大的事实——低敏感层用高位宽是浪费，高敏感层用低位宽则精度崩塌
- 记忆管理方面：现有 memory bank 策略（固定帧缓存、时间邻近启发式）是**内容无关**的——无法区分"清晰的高质量记忆"与"被遮挡/模糊的损坏记忆"，简单丢弃或均匀压缩会丢失关键上下文
- 量化与记忆优化被**独立处理**，没有统一的重要性度量框架，错过协同优化机会

### 作者的核心思路

> 用"重要性"这一统一原则贯穿两端：参数层面用 Weight-Activation Importance Score 指导混合精度位宽分配（重要层多给位宽），内容层面用上下文相似度指导记忆压缩（冗余帧合成为代表特征），两者都在不改变架构的前提下协同提升 SAM2 的部署效率与保真度。

---

## 2. 主要贡献

1. **Contribution 1：** 提出 Importance-driven Mixed-Precision Quantization——基于 Weight-Activation Importance Score（权重-激活联合重要度）的逐层混合精度 PTQ，含与 OBS 二阶方法理论关联的推导与贪心位宽分配算法
2. **Contribution 2：** 提出 Selective Importance-driven Synthesis (SIS)——内容感知的记忆压缩机制，只合成 top-k 最冗余的历史帧，保留多样性并增强长时序理解
3. **Contribution 3：** 系统性验证"量化 + 记忆压缩"的协同设计：4-bit 混合精度下 SA-V J&F 50.3（Model-B），超过所有对比方法在 6-bit 均匀量化下的水平；不修改任何架构

#### 我认为真正的新意

> 单个组件都有前身（HAWQ 做 Hessian 感知混合精度、类 OBS 重要性度量在剪枝文献里常见、记忆压缩有 Mamba/RMem 等），真正的新意在于**把"重要性"做成一个贯穿参数与内容的统一设计语言**，并且给出了一个工程上非常廉价的重要性度量：S_l = mean(|W| ⊙ A) 只需要一次校准集前向的通道能量统计 + 权重绝对值，就近似了 OBS 的 Hessian 加权剪枝准则（式 5 的对角近似推导）。SIS 的 top-k 冗余合成（softmax 加权平均）与"只合成最冗余的、保留最独特的"这一不对称处理，比全局均匀压缩更符合直觉且便宜。但坦白说，SIS 用 t-1 帧做 reference 的 cosine 相似度仍然是"时间邻近"的变体，内容感知程度有限——这是我认为它的弱点。

---

## 3. 方法

> **阅读说明**
> 无官方代码，按论文 Method 整理。框架不修改 SAM2 架构，属于推理侧优化（PTQ + 记忆压缩）。

### 3.1 整体框架

![Figure 1: A comparison of our proposed efficient framework (right) against the original SAM2 (left). Our method com- bines Importance-dri...](https://20020730.xyz/images/tracking/mixqsam2/fig1.webp)
![Figure 2: Our proposed optimization framework. The two core modules are: 1) Importance-driven Mixed-Precision Quantiza- tion (bottom-left...](https://20020730.xyz/images/tracking/mixqsam2/fig2.webp)


**核心架构图**

> 论文 Figure 1（右）：对比原始 SAM2（左，高延迟 + 高显存）与本文框架（右）：Importance-driven Mixed-Precision Quantization（压缩模型）+ Selective Importance-driven Synthesis（压缩 memory bank）。Figure 3：SAM-2B 各 transformer block 的混合精度位宽分配（候选 3/4/5 bit）。

```text
SAM2 标准流水线（streaming, 逐帧）:
Frame t → Image Encoder → 特征（与 memory bank cross-attention）→ Mask Decoder → 掩码
                        ↑                    ↓
             记忆特征写入 memory bank    memory bank（随时间冗余累积）
                          ↓ 本文改造 ↓
1) 量化侧: 每层重要性 S_l（校准集统计 |W|·A）
   → 贪心位宽分配（候选 3/4/5 bit，目标平均位宽 B_avg）
   → 逐层 PTQ 重建（≥20000 iterations/模块）
2) 记忆侧: memory bank 超容量时
   → 以 t-1 帧特征为 reference，cosine 相似度量化各帧重要性
   → 排序，选 top-k (k=3) 最冗余帧
   → softmax 加权融合为 F_synth
   → 重建 bank = F_synth + 未选帧（保持独特信息）+ reference 帧
```

#### 整体流程

1. **离线校准与量化**：用小校准集（32 张图 + 8 个视频）统计每个 linear layer 的输入通道能量 A，与权重绝对值构造重要性矩阵 H，得层重要性 S_l；据此在候选位宽集合（3/4/5）上贪心分配，满足目标平均位宽约束；随后逐模块 PTQ 重建（每模块 ≥20000 次迭代）
2. **在线推理**：量化后的 SAM2 逐帧处理；memory bank 达到容量上限时触发 SIS——以最近帧（t-1）特征为 reference，对更早的历史帧算 cosine 相似度并排序，把 top-k 最相似（最冗余）的帧做 softmax 加权融合成单个代表特征 F_synth，替换掉这 k 帧
3. 新 bank = {F_synth} ∪ {未选中的独特帧} ∪ {reference 帧}，保证信息多样性不丢失

---

### 3.2 Core Module 1 — `Importance-driven Mixed-Precision Quantization`

#### 为什么需要？

均匀位宽 PTQ 把每层都量化到同一位宽，但各层对量化误差的敏感度差异巨大（关键层量化 4-bit 崩精度，冗余层 6-bit 纯浪费）；静态混合精度（HAWQ 式）需要二阶 Hessian 计算，成本高。需要一个便宜的、可解析的层重要性度量来指导位宽分配。

#### 核心做法

- 对每个 linear layer l：校准集上计算输入通道能量向量 A（每通道 L2 范数），与权重绝对值逐元素相乘得重要性矩阵 H，其均值即层重要性 S_l
- 位宽分配建模为带平均位宽约束的加权误差最小化问题，用贪心迭代算法近似求解（高重要层先给高位宽，再按约束逐步下调/上调）
- 与 OBS 的关联：在 λ→0 和对角近似下，OBS 的 Hessian 加权重要性退化为 (|W_ij|·||X_j||_2)^2——即本文度量（差一个平方，单调等价），给出了理论根基

#### 关键公式

仿射量化（b 为位宽，S 为 scale，Z 为 zero-point）：

$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{S} + Z\right),\; 0,\; 2^b - 1\right)$$

输入通道重要性（校准集平均能量）、重要性矩阵与层重要性（与 OBS 的 λ→0 对角近似等价）：

$$A_j^{(l)} = \left(\sum_{n=1}^{N} \left(X_{n,j}^{(l)}\right)^2\right)^{\frac{1}{2}}, \qquad H_{i,j}^{(l)} = |W_{i,j}^{(l)}| \cdot A_j^{(l)}, \qquad S_l = \frac{1}{C_{\text{out}}\cdot C_{\text{in}}} \sum_{i,j} H_{i,j}^{(l)} \;\propto\; \left(|W_{ij}| \cdot \|X_j\|_2\right)^2$$

位宽分配（约束优化 + 贪心求解，p_l 为层参数量，E(b_l) ∝ 2^{-b_l} 单调误差模型）：

$$\min_{b_l \in \mathcal{B}} \sum_{l \in \mathcal{L}} S_l \cdot E(b_l) \quad \text{s.t.} \quad \frac{\sum_{l} b_l \cdot p_l}{\sum_{l} p_l} \leq B_{\text{avg}}$$

#### 代码对应

```text
File: 无官方代码
Class: 无
Function: 贪心迭代位宽调整（按重要性排名初始化 → 逐层降/升位宽至满足约束）
```

#### 我的理解

这个重要性度量的工程价值在于它的"零训练成本"：A_j 只需校准集一次前向的通道统计，H 是逐元素乘法，S_l 是均值——没有任何可学习参数，却在 4-bit 下带来巨大增益（Model-B 上 50.3 vs PTQ4SAM 40.8，+9.5 J&F）。它的理论故事（OBS 对角近似）也讲得通：OBS 的 Fisher/Hessian 加权本质上就是"权重越大、输入激活能量越大的位置越重要"，本文度量恰好是它的第一阶近似。局限也很明显：只对 linear layer 定义（作者自认 Limitation），SAM2 里的卷积层、LayerNorm、位置编码都没有覆盖；且贪心算法是启发式，没有最优性保证。

---

### 3.3 Core Module 2 — `Selective Importance-driven Synthesis (SIS)`

#### 核心做法

- 触发条件：memory bank 超出容量上限时（而非固定时间间隔）
- 重要性量化：以最近记忆帧（t-1）的特征为 reference context，对每个更早的历史帧算 cosine 相似度——相似度高 = 冗余高 = "重要"（值得被合成）
- 排序 → 选 top-k（k=3）最冗余帧 → softmax 归一化加权平均合成 F_synth
- 非对称重建：bank = {F_synth} ∪ {未选中的低相似度帧（独特信息）} ∪ {reference 帧}——只压缩冗余，不动独特帧

#### 关键公式

SIS 合成（F_i、s_i 为第 i 个最冗余帧的特征与相似度分数，w 为 softmax 权重）：

$$F_{\text{synth}} = \sum_{i=1}^{k} w_i \cdot F_i, \qquad w = \text{softmax}(\{s_1, \dots, s_k\}),\quad s_i = \text{cosine}(F_i, F_{t-1})$$

#### 代码对应

```text
File: 无官方代码
Class: 无
Function: 相似度排序 → top-k 选择 → softmax 加权平均 → bank 重建
```

#### 我的理解

SIS 的核心主张是"内容感知"胜过"时间启发式"：时间邻近不代表语义冗余（例如视角突变后的相邻帧信息量很大），而相似度高的帧才是真正冗余。用 t-1 帧做 reference 是有意为之——它代表"当前上下文"，与它相似的过去帧对理解现在帮助最小。Softmax 加权保证合成特征被最可靠的信息主导。但我认为有两个隐患：(1) cosine 相似度对**外观相似但语义不同**的帧（如两个相似的干扰物交替出现）可能误判为冗余并融合，这正是视频分割最怕的错误；(2) 对无人机跨视角场景，视角变化剧烈时 t-1 帧与早期帧的相似度普遍偏低，SIS 可能长时间不触发压缩，记忆膨胀问题仍在——它更像"锦上添花的压缩"而非"有界记忆的保证"。

---


**论文机制图**

![Figure 3: Mixed-precision bit allocation for the SAM-2B model. This figure illustrates the final bit width assigned to each layer. Candid...](https://20020730.xyz/images/tracking/mixqsam2/fig3.webp)
![Figure 5: Result of image instance segmentation](https://20020730.xyz/images/tracking/mixqsam2/fig5.webp)

### 3.4 论文与代码对照

|Paper Module|Code File|Class / Function|作用|
|---|---|---|---|
|Importance-driven Mixed-Precision Quantization|—|—|无官方代码，无法对照|
|Selective Importance-driven Synthesis|—|—|同上|
|SAM2 基座（冻结）|—|—|同上|

#### 论文和代码不一致的地方

无官方代码，Paper ↔ Code 对照暂缺。最接近的开源参照：PTQ4SAM（CVPR 2024，均匀位宽 PTQ）与 Q-MiniSAM2（IJCAI 2025，同组 Ren Xuanxuan 参与的量化评测工作），可用作实现混合精度扩展的基座。

---

### 3.5 训练与推理

#### Training

```yaml
# 注意：这是 PTQ（无需训练），以下为校准/重建配置
Calibration Set: 32 张图 + 8 个视频（随机选取）
Reconstruction: 每个模块 ≥20000 iterations
Bit-width Candidates: 3 / 4 / 5（Figure 3，SAM-2B 的分配结果）
Avg Bit-width: Avg-6（6-bit 配置）与 Avg-4（4-bit 配置）
GPU: 4 × NVIDIA A6000
Framework: PyTorch 2.6.0, CUDA 12.4
# 其他：网络首尾层不量化（惯例）；第一帧/最后一帧权重保持 FP
```

#### Inference

```text
Input（图像/视频帧）
→ 量化 Image Encoder（混合精度，关键层高位宽）
→ Cross-attention with memory bank（超容量时触发 SIS 压缩）
→ 量化 Mask Decoder
→ 掩码输出
```

#### Complexity

```text
Params: 论文未报告（量化后按平均位宽 4-6 bit 存储，远小于 FP32）
FLOPs: 论文未报告
FPS / Latency: 论文未报告
Hardware: 4×A6000（实验用）；部署目标为资源受限设备
```

---

## 4. 实验

### 数据集与指标

|Dataset|Metric|Setting|
|---|---|---|
|MS-COCO 2017 val（5,000 图）|mAP|图像实例分割（Model-S/B/L）|
|SA-V（50,000+ 视频，600,000+ 时空掩码）|J, F, J&F|视频 promptable 分割（Model-B/L）|

### 主要结果

> 最值得关注的结果：**4-bit 混合精度全面碾压 4-bit 均匀量化，且经常反超对比方法的 6-bit 水平**——SA-V Model-B 上 Ours Avg-4 达 50.3 J&F，比 PTQ4SAM W6A6（66.5）在 4-bit 下的 40.8 高 9.5；COCO Model-B 4-bit mAP 34.2，超过 QDrop/PTQ4SAM 的 6-bit 结果（39.3/38.5 是 6-bit，34.2 是 4-bit，差距缩小到 ~5 以内）。

COCO 实例分割 mAP（FP：S 40.3 / B 41.1 / L 41.4）：

|方法|Model-B 6-bit|Model-B 4-bit|Model-L 6-bit|Model-L 4-bit|
|---|---|---|---|---|
|QDrop|39.3|25.1|37.1|29.4|
|PTQ4SAM|38.5|31.6|37.9|30.2|
|Ours (Avg)|**40.3**|**34.2**|**38.9**|**32.0**|

SA-V promptable 分割 J&F（FP：B 72.7 / L 73.7）：

|方法|Model-B 6-bit|Model-B 4-bit|Model-L 6-bit|Model-L 4-bit|
|---|---|---|---|---|
|AdaRound|47.2|25.6|48.1|27.3|
|QDrop|61.4|33.4|63.1|35.8|
|PTQ4SAM|66.5|40.8|67.1|40.4|
|Ours (Avg)|**68.7**|**50.3**|**68.6**|**49.6**|

### 消融实验

> 哪个模块贡献最大？**混合精度量化（MP）**——SA-V Model-B 4-bit 下 Base 40.8 → +MP 49.2（+8.4），而 SIS 单独只有 +1.1（41.9），两者叠加到 50.3。6-bit 下 MP 贡献 (+0.5) 与 SIS 贡献 (+1.2) 相当。说明：低位宽（4-bit）时参数精度是主要矛盾，位宽分配正确性决定成败；高位宽（6-bit）时参数已足够精确，记忆管理（SIS）的价值开始显现。

### 失败案例

- 论文 Limitation 自述：重要性度量**只针对 linear layer**，未覆盖卷积等结构；模型从 Base 扩到 Large 时量化性能下降，作者推测是**低精度误差在深层网络中累积**（error accumulation）
- 定性来看：6-bit 下 SIS 带来的 +1.2 增益远小于 4-bit 下 MP 的 +8.4，说明记忆压缩在"参数已大致保真"时才有边际收益；而当模型变大（L）时两类增益都收窄（68.6/49.6 vs 68.7/50.3），误差累积压制了优化空间

#### 我认为失败的原因

B→L 的退化本质上是**敏感度分布随深度变化**：深层网络里每层的量化误差都会传播放大，而本文的层重要性是独立计算的（没有建模层间依赖），贪心分配可能把误差集中到传播路径上的连续低重要层。这与 BRECQ 用 blockwise reconstruction 解决层间依赖的动机一致——本文每模块独立重建（≥20000 iter）没有 block 级联合优化。另一个隐患：SIS 的 cosine 相似度把"外观相似"当"冗余"，在多目标互相遮挡、同类目标多个出现的场景（如无人机俯视人群）容易误融合不同对象，论文没有此类失败分析。

---


### 论文图示（截图）

![Figure 4: Visual Segmentation Results](https://20020730.xyz/images/tracking/mixqsam2/fig4.webp)

## 5. 复现指南

**Repository**

```text
GitHub: 无官方代码
Commit: 无
Checkpoint: 无
```

**Environment**

```yaml
Python: 未提供（PyTorch 2.6.0 时代）
PyTorch: 2.6.0
CUDA: 12.4
GPU: 4 × NVIDIA A6000
```

**关键运行命令**

无官方代码，无法复现。替代信息：
- 作者单位：西安电子科技大学（School of Computer Science and Technology / School of Artificial Intelligence / School of Electronic Engineering），通讯作者 Xu Yang、Yanhua Yang
- 最接近的可复现基座：PTQ4SAM（CVPR 2024，有代码，SAM 的均匀位宽 PTQ）+ Q-MiniSAM2（IJCAI 2025，同一团队 Ren Xuanxuan 的 SAM2 量化评测工作，已有量化管线可扩展为混合精度）

#### 复现结果

未复现（无代码）。

#### 遇到的问题

无官方代码是本笔记复现部分的最大阻碍；复现需自行实现：重要性统计（校准集前向 + 通道能量）、贪心位宽分配、逐模块重建（≥20000 iter/模块，4 卡 A6000 成本不低），以及 SIS 的 bank 管理逻辑——全部逻辑论文描述已足够详细，工程上可行但工作量相当于重写一个 PTQ 框架。

---

## 6. 批判性思考

### 优点

- 统一的"重要性"视角贯穿参数与内容两个层面，叙事清晰、动机扎实
- 重要性度量零训练成本且与 OBS 有理论关联，4-bit 下增益显著（+8.4 J&F），工程上很实用
- 不修改架构、纯推理侧优化（PTQ + 记忆压缩），部署路径干净；校准成本低（32 图 + 8 视频）

### 局限

- 无官方代码，复现成本高
- 重要性度量仅限 linear layer；贪心位宽分配无最优性保证；层间误差依赖未建模（B→L 退化即证据）
- SIS 的 cosine 相似度仍本质上是时间邻近的变体，对"外观相似但语义不同"的帧会误判冗余；SIS 是"超容量才触发"的被动压缩，不保证记忆有界
- 实验基准集中在 SA-V/COCO，没有长视频（LVOS 类）与真实边缘设备（Jetson 等）的延迟/功耗实测，FPS 与 FLOPs 均未报告

### 我最关心的问题

1. 4-bit 下 50.3 的 J&F 距离 FP（72.7）仍有 22 个点，这个差距在视频分割里能否接受？论文没有给出 4-bit 的定性视频结果之外的误差分析
2. SIS 的 k=3 与触发阈值如何选择？对长视频是否稳定？
3. 量化与 SIS 的协同：量化后的特征做 cosine 相似度是否仍可靠（量化噪声会不会破坏相似度排序）？

### 可以迁移到我的研究中的部分

对 DAM4SAM（SAM2 基座、干扰物记忆、跨视角无人机目标从大变小时失效、需轻量化部署）的直接迁移点：

- **轻量化部署的最短路径**：DAM4SAM 要上无人机边缘端，混合精度 PTQ 是比蒸馏/剪枝更便宜的路线；"关键层保高位宽"的分配逻辑可以直接套用，且 SAM2 基座与本文同构，迁移成本低
- **重要性度量的复用**：S_l = mean(|W|⊙A) 可以作为 DAM4SAM 微调时"冻结哪些层/给哪些层更多学习率"的依据——用重要性分数做参数高效的层选择，比随机选择更有依据
- **SIS 用于干扰物记忆压缩**：干扰物记忆随时间膨胀的问题与 SIS 解决的问题同构——把"最冗余的干扰物记忆帧"合成为代表特征，保留"独特帧"（如干扰物首次出现、外观剧变帧），正好是"记忆管理 + 抗干扰"的落地机制
- **量化误差与尺度失效的叠加**：跨视角目标从大变小时，小目标像素少、特征能量低（A_j 小），被量化误差破坏的风险更高——这提示量化方案要按"目标尺度"保护对应层，是本文没有覆盖的维度

### 新想法

1. **尺度感知的位宽分配**：把重要性度量扩展为"目标尺度条件"——检测到小目标（低分辨率 ROI）时对其依赖的空间高频层临时提高有效精度（动态量化），直指跨视角从大变小的失效点
2. **对象级 SIS**：SIS 的相似度目前是帧级（整帧特征）；改成对象级（按对象 ROI 算），干扰物记忆与目标记忆分别压缩——"干扰物冗余帧融合、目标关键帧保留"，契合 DAM4SAM 双记忆结构
3. **量化-记忆联合重要性**：论文的两个重要性是分离的；可定义联合目标——"对冗余度最高的帧允许更激进的压缩"，让记忆压缩预算反哺参数精度，形成真正协同
4. **训练侧闭环**：微调时只在低位宽层加适配器（类似 LoRA 低秩约束），训练出的模型天然量化友好（quantization-aware fine-tuning），避免"训练 FP、部署量化"的精度落差

---

## 7. 深度阅读标注

本节暂无额外阅读标注。

---

## 8. 总结

### 三句话总结

1. **Problem：** SAM2 计算与显存开销大，且现有量化用均匀位宽、记忆压缩用时间启发式，都不考虑"什么信息真正重要"，资源受限设备上无法高保真部署
2. **Method：** 用统一的"重要性"框架协同优化：Weight-Activation Importance Score（S_l = mean(|W|⊙A)，OBS 对角近似）指导混合精度位宽分配；SIS 以内容相似度识别并 softmax 融合 top-k 最冗余记忆帧，保留独特帧
3. **Result：** SA-V 上 Model-B Avg-4 达 50.3 J&F（PTQ4SAM 4-bit 仅 40.8），4-bit 常反超对比方法 6-bit；COCO mAP 同样全面领先；且不修改架构

### 一句话评价

一篇"把重要性度量做便宜、把两件事串起来讲"的部署优化论文：4-bit 混合精度的增益真实且工程价值高，但 SIS 的内容感知有限、无开源代码、误差累积问题未解，属于"值得借鉴思想、不必完整复现"的实用型工作。

### 是否值得复现？

-  ⭐ 仅了解
    
-  ⭐⭐ 一般
    
-  ⭐⭐⭐ 值得作为 Baseline
    
-  ⭐⭐⭐⭐ 值得复现
    
-  ⭐⭐⭐⭐⭐ 与我的研究高度相关
    

**复现星级：⭐⭐（一般）。** 理由：论文逻辑完整但无官方代码，完整复现需自建 PTQ 框架（重要性统计 + 贪心分配 + 逐模块重建），成本高；且 SIS 的设计理念（记忆压缩）对我的干扰物记忆管理有启发，但实现细节（cosine + softmax 融合）需要按我的场景重新设计。更务实的路径：直接采用其"重要性度量指导位宽分配"的公式思想，在 PTQ4SAM 代码上扩展混合精度，不必复现整篇。

---
