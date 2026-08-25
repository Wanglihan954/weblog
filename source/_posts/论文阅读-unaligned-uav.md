---
title: "论文阅读｜Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach"
categories:
  - 文献阅读
  - Tracking
tags:
  - "文献笔记"
  - "AI论文"
  - "追踪"
  - "RGB-T"
  - "UAV"
  - "未对齐"
  - "对齐"
  - "视频目标跟踪"
  - "Tracking"
description: "无人机通常分别搭载可见光（RGB）与热红外（TIR）传感器，两个模态的分辨率、安装位置和视场差异会产生天然的空间失准。本文定义 unaligned UAV RGBT tracking 任务：使用未经人工后处理的原始未对齐 RGB/TIR 图像，同时预测两个模态中的目标框。…"
readmore: true
mathjax: true
date: 2026-08-21 20:30:00
updated: 2026-08-21 23:00:00
abbrlink: "ed78dc02"
---
> 本文基于论文、补充材料与公开代码整理。文中的“我的理解”和“批判性思考”属于个人分析；
> 论文插图均来自原论文或补充材料，仅用于学习与讨论。

## 论文信息

**Title:** Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach<br>
**Authors:** Yun Xiao, Yuhang Wang, Jiandong Jin, Wankang Zhang, Chenglong Li<br>
**Venue:** AAAI 2026（The Fortieth AAAI Conference on Artificial Intelligence）<br>
**Pages:** 11014–11022<br>
**DOI:** 10.1609/aaai.v40i13.38079<br>
**GitHub / Code&Datasets:** https://github.com/NOP1224/Unaligned_RGBT_Tracking（论文正文给出的入口；本次未做仓库级核验）<br>
**Project Page:** —<br>

### 摘要

无人机通常分别搭载可见光（RGB）与热红外（TIR）传感器，两个模态的分辨率、安装位置和视场差异会产生天然的空间失准。本文定义 **unaligned UAV RGBT tracking** 任务：使用未经人工后处理的原始未对齐 RGB/TIR 图像，同时预测两个模态中的目标框。作者构建 LUART（Largescale Unaligned UAV RGBT Tracking）基准，包含 1,453 对无人机视频序列、1.02 million 对图像帧、42 类目标、22 类挑战属性以及多种空间偏移尺度，并为两个模态分别提供边界框标注。

方法是 SFCATrack（Spatial-Feature Collaborative Alignment Tracker）。其第一阶段使用 Mixture of Shift-Estimation Experts（MSEE）在不同偏移尺度上自适应估计两个模态的空间 shift，并用预测结果重新定位 TIR search region；第二阶段使用 Cross-Modal Alignment and Fusion（CMAF）级联可变形卷积，在特征层纠正非线性失配并用轻量门控融合模态信息。论文在 LUART 与 LasHeR-Unaligned 上重训并比较 14 个代表性 RGBT tracker，报告 SFCATrack 在两套数据上均取得最佳结果。

> 事实边界：本文给出了数据集统计、模块公式、训练阶段与实验结果；没有在正文中给出参数量、FLOPs、FPS、逐偏移区间数值表或专门的失败案例章节。

<!-- more -->

---

## 论文资源

- **Zotero:** 未导入
- **PDF:** [Open PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38079/42041)
- **Paper:** [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38079)
- **GitHub / Code&Datasets:** https://github.com/NOP1224/Unaligned_RGBT_Tracking（论文列出；未核验）

---

## 1. 研究动机

### 要解决什么问题？

> 让无人机上的 RGB-T 跟踪器直接处理两个独立传感器产生的**原始未对齐视频**，而不是先把 TIR 图像刚性配准到 RGB，或假设两个模态共享一个目标框。

现有 UAV RGBT 数据与方法通常依赖严格人工对齐的序列，并且只提供一个跨模态共享的 ground-truth bounding box。真实无人机平台上，RGB 与 TIR 传感器的分辨率和安装位置不同，导致目标在两个模态中的位置并不相同；相对位移还会随无人机高度、相机运动和目标运动变化。本文将任务目标明确为：在不做人工后处理的 RGB/TIR 原图上，分别预测两个模态的目标框。

### 为什么 UAV 场景比已有未对齐设定更难？

- **失准类型更多：** LasHeR-Unaligned 主要体现为两个模态之间的目标平移；UAV 场景还会出现 TIR 目标出视野、旋转运动以及更明显的外观差异。
- **分辨率不同：** LUART 保留 RGB 1920×1080 与 TIR 640×512 的原始分辨率，跨分辨率特征学习和对应关系估计同时存在。
- **偏移范围更宽：** 数据覆盖约 0–20 pixels 的小偏移到超过 100 pixels 的大偏移，不能只靠一个固定的全局偏移处理。
- **目标更小、运动更复杂：** 论文指出 LUART 的平均目标尺寸显著小于 LasHeR-Unaligned，并覆盖多种 UAV 高度、运动模式、天气和光照条件。

### 现有方法的问题

- **只做融合、不先处理空间对应：** 大多数 RGBT tracker 重点设计跨模态融合，却默认输入已经对齐；直接拼接或交互失准特征会把互补信息变成噪声。
- **单一全局 shift 不够：** 论文以 AMNet 为例指出，统一预测一个全局 feature shift 并应用到所有 feature map 对小偏移有帮助，但难以处理严重的目标空间失准。
- **刚性/单应变换存在表达边界：** 图像级 warp 可以纠正整体位置，但目标区域仍可能有非线性形变；因此仅完成图像空间对齐并不能保证特征层逐点对应。
- **数据协议不贴近真实部署：** 人工对齐会隐藏传感器安装偏差、跨分辨率、TIR 目标出视野等问题，使模型在真实未对齐输入上的鲁棒性无法直接评估。

### 作者的核心思路

> 先在图像层用**尺度感知的 shift 专家**纠正主要空间偏移，再在特征层用**渐进可变形对齐 + 门控融合**处理剩余的非线性失配。SFCATrack 不把“对齐”当作一次性的预处理，而是把空间 warp 与特征交互放进同一个跟踪管线。

#### 我的理解（推断）

本文的关键重构不是单纯增加一个更强的融合模块，而是将跨模态跟踪拆成两个误差层级：

1. **可解释的全局误差：** 两个模态目标框的坐标差，用于重新定位 TIR search region；
2. **难以用一个几何变换解释的局部误差：** 目标形变、残余视差和模态差异，用 CMAF 的 deformable sampling 与 gate 处理。

这使 MSEE 的输出不只是辅助特征，而是直接改变后续输入区域；CMAF 则不再承担全部大范围搜索，而负责在粗对齐之后收敛局部残差。

---


**论文图示**

![Figure 1: Figure 1: Comprehensive Comparison between our LUART and LasHeR-Unaligned. Figure (a) shows the distribution of object pixel area of two ...](/images/tracking/unaligned-uav/fig1.webp)

## 2. 主要贡献

1. **新任务与基准：** 定义未对齐 UAV RGBT tracking，使用原始未对齐 RGB/TIR 图像，并构建 LUART 大规模基准；每个模态分别标注目标框。
2. **协同对齐跟踪器：** 提出 SFCATrack，将 MSEE 的图像空间对齐与 CMAF 的特征对齐、融合结合起来。
3. **系统评估：** 在 LUART 上重训并比较 14 个代表性 RGBT tracker，同时在 LasHeR-Unaligned 上进行验证；报告 LUART 的 22 类挑战属性和不同偏移尺度。

#### LUART 的基准价值

- **规模：** 1,453 对视频序列，约 1.02 million 对 RGB/TIR 图像帧，平均每个视频约 700 帧。
- **类别：** 42 个目标类别，包含行人、车辆、动物和工业机器等层级类别。
- **挑战：** 22 个序列级挑战属性，其中 15 个来自既有 RGBT tracking 设定，7 个是作者针对未对齐 UAV 场景加入的挑战；论文明确举例包括 TIR 目标出视野和旋转运动，但没有在正文逐项列出全部 7 个名称。
- **真实采集：** 使用 DJI Matrice 300 RTK、DJI Mavic 3T 和 Zenmuse H20T 等设备，在不同地点、季节和场景采集。
- **失准分布：** 偏移从小到大呈负相关/近似指数衰减；训练集和测试集的偏移分布相近。

#### 我认为真正的新意（推断）

> 论文把“跨模态对齐”做成了**图像空间校正与特征空间校正的协同问题**。MSEE 解决“搜索区是否包含对应目标”，CMAF 解决“包含之后两个模态的局部特征是否仍然对应”。这比只在融合层学习容错，或只用一个刚性 warp，更贴近 UAV 传感器的误差结构。

---

## 3. 方法

> **阅读说明**
> 论文正文给出 MSEE、CMAF 的结构、公式和训练阶段，并列出 Code&Datasets 入口；本次仅基于全文文本和 `paper_meta.json` 整理，未进行仓库源码核对。因此 3.4 的 code mapping 不填写未经验证的文件名、类名或行号。

### 3.1 整体框架

![Figure 4: Figure 4: Overall Framework of Our SFCATrack.](/images/tracking/unaligned-uav/fig4.webp)


**输入：** 未对齐的 RGB/TIR template 与 search region。整体跟踪器建立在 OSTrack 的 dual-branch tracking framework 上。

```text
RGB search X^V ─┐
                ├─ 共享 ResNet 特征提取 ─→ [F^V; F^I]
TIR search X^I ─┘                                      │
                                                       ▼
                          MSEE：shared expert + 5 个 scale experts
                                                       │
                    router 选择 active expert，回归 Δp=[Δx1,Δy1,Δx2,Δy2]
                                                       │
                    由两个模态框坐标差构造 homography，warp TIR search region
                                                       ▼
        RGB/TIR template + 对齐后的 RGB/TIR search region → patch embedding
                                                       │
                       12 层 Transformer backbone（各层插入 CMAF）
                                                       │
                MLP 融合 → 级联 deformable convolution → gated fusion
                                                       │
                     对齐/融合特征残差加入两模态分支 → OSTrack tracking head
                                                       ▼
                         分别输出 RGB 与 TIR 模态的目标框
```

#### 逐步流程

1. MSEE 从 RGB/TIR search region 提取共享 ResNet 特征，拼接两个模态的表示。
2. shared expert 建模共同的跨模态失准模式；router 从 5 个尺度专家中选择当前 active expert。
3. offset head 预测两个模态目标框四个坐标之间的差值，并由此构造 homography，将 TIR search region warp 到 RGB 参考坐标。
4. 对 template 和已经空间校正的 search region 做 patch embedding，加入可学习 positional encoding，送入 Transformer backbone。
5. CMAF 在每一层生成共享的 fused feature，以它作为 deformable convolution 的 offset guidance，逐步修正模态特征。
6. 轻量 gating network 生成动态融合权重；对齐后的融合特征以残差形式加入 RGB/TIR 输入特征，再传给下一层和最终 tracking head。

#### 关键设计判断

- **MSEE 的输出是输入区域级别的动作：** 不是只改变特征权重，而是直接 warp TIR search region。
- **CMAF 的输出是特征级别的动作：** 它在已经做过整体空间校正后处理局部、非线性或残余失配。
- **两者形成前后级联：** 图像层大范围校正减少 deformable convolution 的搜索负担，特征层残差对齐弥补单一 homography 的表达不足。

---

### 3.2 Core Module 1 — MSEE：Mixture of Shift-Estimation Experts

#### 为什么需要？

LUART 的模态偏移不是单一固定尺度。小偏移、中等偏移和大偏移的视觉对应难度不同，用同一个专家处理所有尺度会降低 shift estimation 的适应性；固定首帧偏移也不能覆盖随 UAV 运动变化的后续帧。

#### 特征提取与共享专家

给定 RGB 与 TIR search region $X^m$（$m∈{V,I}$），论文先用参数共享的 ResNet 提取特征：

$$F_m = Enc_m(X_m),\qquad m\in\{V,I\}.$$

共享参数的目的，是避免 MSEE 偏向某一个模态。两个特征沿通道维拼接：

$$F_{VI}=[F_V;F_I].$$

随后用一个 shared expert 提取两个模态共同的失准表示：

$$F_S=Expert_{shared}(F_{VI}).$$

#### 尺度专家与路由

论文设置 $N=5$ 个 scale experts。每个专家具有 SwiGLU 激活单元，并针对一种偏移尺度学习：

$$F_i=Expert_i(F_{VI}),\qquad i=1,\ldots,N.$$

scale expert 的训练区间为：

```text
Expert 1: [0, 20) pixels
Expert 2: [20, 35) pixels
Expert 3: [35, 50) pixels
Expert 4: [50, 75) pixels
Expert 5: [75, +∞) pixels
```

尺度感知 router 输入联合特征 $F_{\mathrm{VI}}$，输出专家置信度向量 $ω=[ω1,…,ωN]$。论文描述的推理选择是取最高置信度专家 `i*`，再与 shared expert 相加：

$$F_{mix}=F_S+F_{i^*}.$$

> 事实边界：论文说明 router 输出置信度并选择最高置信度专家；正文没有报告每个专家在测试集上的实际调用比例、router 熵或边界样本的选择稳定性。

#### Shift 回归与图像 warp

offset head 根据 $F_{\mathrm{mix}}$ 预测两个模态目标框的四个坐标差：

$$\Delta p=(\Delta x_1,\Delta y_1,\Delta x_2,\Delta y_2)=Head(F_{mix})\in\mathbb{R}^{4}.$$

其中四个量分别表示 RGB 与 TIR ground-truth bounding boxes 在左上角和右下角坐标上的差异。预测结果被转换为 homography matrix，并用于 warp TIR search region，从而把两个模态的搜索区域拉到更接近的空间位置。

#### MSEE 的三阶段训练

1. **共同表示初始化：** 在完整训练集上训练 feature extractor、shared expert 和 offset prediction head。
2. **尺度专家专门化：** 冻结 feature extractor，用 shared expert 初始化 5 个 scale experts；在对齐图像上模拟相应 offset range，分别训练各尺度专家。
3. **自适应路由：** 冻结 backbone 与 experts，在完整训练集上训练 router 和 offset prediction head，使模型能够从真实样本中选择合适的 scale expert。

MSEE 的 shift 损失为：

$$L_{msee}=L_1(\Delta p,\Delta p^{gt}),$$

其中 `Δp^gt` 是两个模态 ground-truth boxes 的相对坐标差。

#### 我的理解（推断）

MSEE 实质上是一个**带尺度先验的硬路由 MoE**：shared expert 保证共同结构，scale expert 承担偏移范围的专门化，router 将运行时样本分配给一个 active expert。它并没有让每个专家都对每一帧完整推理，因此“适应不同偏移尺度”同时包含表示能力与计算选择两层含义。

需要注意，四个 bbox 坐标差被压缩成一个 homography 后再 warp。这个表示对整体平移、尺度和部分旋转是有用的，但不等于完整的像素级光流或逐区域形变场；非线性残差由 CMAF 承担。

---

### 3.3 Core Module 2 — CMAF：Cross-Modal Alignment and Fusion

#### 为什么需要？

MSEE 只完成 search region 的图像级空间校正。传感器视角、目标形状和成像方式仍可能造成局部非线性失配；如果直接在这些特征上做模态融合，RGB/TIR 的互补信息可能被错误对应污染。CMAF 用共享融合特征指导 deformable convolution，继续做特征级对齐。

#### Patch embedding 与 backbone 输入

对 template $Z^m$ 和经过 MSEE 对齐的 search region $X^m$（$m∈{V,I}$）分别切分 `P×P` patches，并展平为 patch sequence。embedding 层叠加可学习位置编码：

$$E_r^m=Embed(P_r^m)+Pos_r,\qquad r\in\{x,z\}.$$

每个模态的 template/search embedding 拼接后进入 Transformer：

$$F_0^m=[E_z^m;E_x^m],$$

$$F_j^m=Layer_j(F_{j-1}^m),\qquad j=1,\ldots,12.$$

论文将 CMAF 插入所有 backbone layers。

#### 共享空间投影与融合

CMAF 首先用多模态 feature fusion layer $F_L$ 将 RGB 与 TIR 特征投影到共享空间：

$$F_{fuse}=F_L(F_V,F_I).$$

该 fusion layer 由三个 MLP 组成：先在空间和通道层面对两个模态分别投影，再拼接，最后用另一个 MLP 得到共享的 fused feature。这里的 fused feature 既用于融合，也作为后续 deformable convolution 的空间指导。

#### 级联 deformable convolution

对每个模态的特征进行级联可变形卷积：

$$\widetilde F_k^m=D_k(\widetilde F_{k-1}^m,F_{fuse}),\qquad k=1,\ldots,N.$$

$D_k$ 的第一个输入是待对齐特征，第二个输入 $F_{\mathrm{fuse}}$ 不被卷积，而是提供 offset guidance。论文将多次 deformable convolution 解释为 progressive alignment：每一次采样都在前一次结果上继续修正。

#### 动态门控融合

获得对齐后的多模态特征后，轻量 gating network $G$ 生成动态融合权重。融合结果随后以 residual 的形式加回 RGB 与 TIR 输入特征，并传入下一层：

```text
RGB/TIR features
      ↓
shared-space MLP fusion → fused guidance
      ↓
progressive deformable convolutions
      ↓
dynamic gated fusion
      ↓
residual add to modality-specific features
      ↓
next Transformer layer
```

#### 我的理解（推断）

CMAF 不是把两个模态强行变成一张图，而是保留 modality-specific branches，再用 fused feature 控制采样位置和融合权重。这样做的好处是把“哪里取信息”和“取多少信息”分开：deformable convolution 修正对应位置，gate 决定当前 RGB/TIR 信息的相对可信度。

它与 MSEE 的接口也很清晰：MSEE 先降低大位移，CMAF 再处理局部残差。若直接把 CMAF 用于极大偏移，deformable sampling 仍可能从错误的区域开始；这也是跨视角 UAV 迁移时不能跳过粗几何校正的原因。

---


**论文机制图**

![Figure 2: Figure 2: Object main category and subcategory statistics of our LUART dataset.](/images/tracking/unaligned-uav/fig2.webp)
![Figure 3: Figure 3: Number of frames at different offset scales in our LUART dataset.](/images/tracking/unaligned-uav/fig3.webp)
![Figure 7: Figure 7: Illustration of the relocation of the search region. (a) Initial RGB search region. (b) No Aligned TIR search region. (c) Align...](/images/tracking/unaligned-uav/fig7.webp)

### 3.4 论文与代码对照

> 论文正文在摘要末尾列出 Code&Datasets 入口，但全文材料没有给出仓库目录、文件名、类名或 commit。本表只映射论文模块与可核验的论文证据，不虚构源码路径。

| Paper Module | 论文位置 / 事实 | Code mapping（本次） | 状态 |
|---|---|---|---|
| SFCATrack overall | Methodology：基于 OSTrack 的 dual-branch tracker，串联 MSEE 与 CMAF | `SFCATrack` 具体入口文件未在论文全文给出 | 论文给出 Code&Datasets 入口；未做仓库核验 |
| MSEE shared expert | Sec. “Mixture of Shift Estimation Experts”：共享 ResNet、shared expert、共同失准表示 | `shared expert` 的具体实现文件未给出 | 未核验 |
| MSEE scale experts | Eq. (2)：5 个 SwiGLU scale experts，按五个 offset bins 专门化 | `Expert_i` 的具体实现文件未给出 | 未核验 |
| Scale-aware router | Eq. (4) 前后：输出 $ω$，选择最高置信度 active expert | router 与 top-1 selection 的具体实现文件未给出 | 未核验 |
| Offset head + image alignment | Eq. (4)：回归 `Δx1,Δy1,Δx2,Δy2`，构造 homography 并 warp TIR search region | offset head / homography warp 的具体实现文件未给出 | 未核验 |
| CMAF fusion layer | Eq. (7)：三个 MLP 投影 RGB/TIR 到 shared space | $F_L$ 的具体实现文件未给出 | 未核验 |
| Progressive deformable alignment | Eq. (8)：级联 deformable convolution，以 $F_{\mathrm{fuse}}$ 提供 offset guidance | $D_k$ 的具体实现文件未给出 | 未核验 |
| Gated fusion | Methodology：轻量 gating network 动态融合，对齐特征 residual add | gate 与 residual path 的具体实现文件未给出 | 未核验 |
| Training losses | Eq. (9)：MSEE 使用 L1；Eq. (10)：encoder+CMAF 使用 OSTrack 的分类、GIoU、L1 损失 | 配置、训练脚本和 loss 文件未给出 | 未核验 |

#### 论文与代码的已知边界

- 论文提供了仓库入口，但本次任务没有 clone 或读取仓库，因此不能确认代码是否已经公开、当前 API 是否与论文一致。
- 不能从论文安全推断 PyTorch 版本、依赖、配置文件、预训练权重、训练命令、默认 batch size 或硬件环境。
- 论文称 MSEE 的三阶段训练和 CMAF 的后续训练是 staged training；实际 checkpoint 命名、冻结参数范围和数据加载细节需要以 release 为准。

---

### 3.5 训练与推理

#### Training

论文采用分阶段训练：

```yaml
Dataset:
  LUART train: 1,010 sequences / 706,689 RGB-T image pairs
  partition: sequence-level

MSEE stage 1:
  train: feature extractor + shared expert + offset prediction head
  loss: L1(predicted offset, ground-truth relative displacement)

MSEE stage 2:
  freeze: feature extractor
  initialize: scale experts from shared expert
  train: five scale experts on simulated offset ranges
  bins: [0,20), [20,35), [35,50), [50,75), [75,+∞)

MSEE stage 3:
  freeze: backbone and experts
  train: router + offset prediction head on full training set

Tracker/CMAF stage:
  freeze: staged MSEE module
  train: backbone encoder + CMAF
  loss: L_total = L_cls + λ1 L_iou + λ2 L1
  L_cls: weighted focal loss
  L_iou: generalized IoU loss
  L1: bounding-box regression loss
```

论文没有在全文中报告 $λ1$、$λ2$ 的具体数值，也没有给出 batch size、epoch、learning rate、参数量或训练硬件。不要用其他 RGBT tracker 的默认配置替代这些缺失信息。

#### Inference

```text
输入当前 RGB/TIR template 与 search region
  → MSEE 提取共享/尺度特征
  → router 选择 active scale expert
  → 回归四个坐标 shift
  → 构造 homography，warp TIR search region
  → RGB/TIR patch embedding + 12-layer Transformer
  → 每层 CMAF：shared fusion guidance
  → cascade deformable alignment
  → dynamic gated fusion + residual update
  → OSTrack tracking head
  → 分别输出 RGB 与 TIR 目标框
```

#### 复杂度与可复现性

论文给出了 LUART 与 LasHeR-Unaligned 的精度，但没有在全文中报告 SFCATrack 的 FPS、参数量、FLOPs、单模块延迟或显存占用。因此无法仅凭论文判断 MSEE 的专家选择是否带来端侧速度收益，也不能把“轻量 gating”解释成完整实时性证据。

---

## 4. 实验

### 数据集与指标

|Dataset|规模 / 内容|Protocol|Metrics|
|---|---|---|---|
|LUART train|1,010 sequences，706,689 image pairs|按视频序列划分训练集；RGB/TIR 保留原始未对齐输入；两个模态分别有 box 标注|论文以 PR / NPR / SR 报告 OPE 结果|
|LUART test|443 sequences，310,669 image pairs；平均目标更小，包含 42 类目标和 22 类挑战|测试序列按目标类别、场景类别、挑战属性与偏移分布等标准划分|PR / NPR / SR|
|LasHeR-Unaligned|既有未对齐 RGBT benchmark，论文用于外部验证|在该数据上比较 SFCATrack 与已有方法|PR / NPR / SR|

#### LUART benchmark protocol

- 采集设备包括 DJI Matrice 300 RTK、DJI Mavic 3T 和 Zenmuse H20T；RGB 原始分辨率为 1920×1080，TIR 原始分辨率为 640×512。
- 五名专业标注员进行逐帧标注，并经过多阶段复核；由于两个模态的空间偏移，RGB 与 TIR 使用各自的 bounding box。
- 数据按视频序列划分 train/test，而不是按帧随机切分；划分时考虑目标类别、场景类别、挑战属性和位置偏移分布。
- 论文在 LUART test 上将 14 个代表性 RGBT trackers 重新训练后比较；这保证了表中结果不是直接把对齐数据集的 checkpoint 原样搬到未对齐输入上。
- 评价结果用 Precision Rate（PR）、Normalized Precision Rate（NPR）和 Success Rate（SR）报告。论文没有在正文展开“双模态分别预测框”如何汇总为最终指标的实现细节，复现时需以官方评测脚本为准。

### 主要结果

> 以下数字直接来自论文表格；括号中的差值是对表内数字的算术计算，不代表额外实验。

- **LUART（Table 2）：** SFCATrack 达到 **57.3 PR / 51.9 NPR / 44.6 SR**。相对表中的 Multi-modal Baseline（48.6 / 45.3 / 38.3），分别提升 **+8.7 / +6.6 / +6.3**。
- **LasHeR-Unaligned（Table 3）：** SFCATrack 达到 **60.7 / 55.1 / 47.9**，高于 CAFormer 的 **58.7 / 54.0 / 46.9**，差值为 **+2.0 / +1.1 / +1.0**。
- **挑战属性（Fig. 6）：** 论文文字明确指出在极端光照（HI）、小目标（SO）、跨模态干扰（TC/BC）以及水平/垂直运动（HM/VM）等场景表现出鲁棒性；图中为雷达图，没有给出每一类的完整数值表。
- **搜索区可视化（Fig. 7）：** 论文比较原始 RGB search region、未对齐 TIR search region、用首帧 offset 调整的 TIR search region 和 MSEE 调整结果，定性展示 MSEE 对当前搜索区的重新定位。

### 消融实验

#### MSEE 与 CMAF 的组件消融（LUART）

|Baseline|MSEE|CMAF|PR|SR|
|---|---:|---:|---:|---:|
|✓|—|—|48.6|38.3|
|✓|✓|—|53.2|41.5|
|✓|✓|✓|57.3|44.6|

- 加入 MSEE 后，相对 baseline 提升 **+4.6 PR / +3.2 SR**，说明图像空间 warp 已能显著改善搜索区对应。
- 再加入 CMAF 后，相对 baseline 提升 **+8.7 PR / +6.3 SR**，说明特征级非线性对齐和门控融合提供了额外收益。
- 论文没有把 MSEE 的“预测 shift 错误”与 CMAF 的“特征残差错误”分别报告，因此不能从该表推断二者对所有偏移尺度的贡献相同。

#### MSEE 内部消融（LUART）

|Baseline|Shared Expert|Scale Experts|PR|SR|
|---|---:|---:|---:|---:|
|✓|—|—|48.6|38.3|
|✓|✓|—|51.1|40.0|
|✓|✓|✓|53.2|41.5|

- shared expert 单独带来 **+2.5 PR / +1.7 SR**。
- 在 shared expert 上加入 scale experts 后，较 baseline 提升 **+4.6 PR / +3.2 SR**。
- 这个消融验证了“共同失准模式 + 偏移尺度专门化”的组合，但没有单独给出 router 的 oracle、soft mixture 或固定专家对照。

### 失败案例

#### 论文明确提供的限制事实

- 论文没有单独的 Failure Cases 章节，也没有逐帧失败可视化或逐偏移区间的 PR/NPR/SR 表。
- LUART 明确包含 TIR 目标出视野、旋转运动、极端光照、小目标、跨模态干扰等挑战，但正文只通过雷达图给出整体属性比较。
- MSEE 用四个 bbox 坐标差构造 homography；CMAF 用 deformable convolution 处理图像 warp 后仍存在的特征失配。
- LUART 的 offset 分布长尾延伸到超过 100 pixels，但 MSEE 的最大训练桶写成 `[75,+∞)`，该桶内部仍包含多种严重程度。

#### 基于方法结构的推断性失败模式

1. **TIR 目标完全出视野：** 如果当前 TIR search region 中没有目标像素，homography 只能移动已有内容，不能恢复传感器没有观测到的目标；CMAF 也只能在可见特征上做 deformable sampling。这是任务层面的信息缺失，不是简单融合可以保证解决的。
2. **大偏移与偏移桶边界：** router 使用 top-1 active expert，样本在 20、35、50 或 75 pixels 附近时可能落入相邻专家；论文没有报告边界样本的稳定性，也没有报告 `[75,+∞)` 内不同偏移的细分性能。
3. **旋转与非线性视差：** 四个 box-coordinate differences 被压缩成 homography 参数，可能无法精确表达局部旋转、目标形变和 UAV 视角造成的区域级视差。CMAF 是补救机制，但论文没有针对每种形变给出独立消融。
4. **小目标与跨分辨率：** RGB 1920×1080、TIR 640×512，且 LUART 的平均目标尺寸小于 LasHeR-Unaligned；小目标在 TIR 中的可用纹理更少，shift regression 和 deformable offset 都可能受量化误差影响。这个判断是结构推断，不是论文报告的单项失败率。
5. **首帧/模板失准：** 若初始 template 或 search region 已严重失准，后续 MSEE 可能在错误的跨模态对应上回归 shift；论文没有讨论首帧质量、模板污染和重检测机制。
6. **未量化的效率风险：** 论文没有给出参数量、FLOPs、FPS 或各专家调用比例，因此不能确认 MSEE 的路由节省了多少推理开销，也不能排除 CMAF 在 12 层全部启用带来的部署成本。

---


### 论文图示（截图）

![Figure 5: Figure 5: Evaluation result on LUART test set using preci- sion and success plots, where the scores are presented in the legend. All trac...](/images/tracking/unaligned-uav/fig5.webp)
![Figure 6: Figure 6: Comparisons of SFCATrack (Ours) and the com- peting methods under different attributes in our LUART test- ing set.](/images/tracking/unaligned-uav/fig6.webp)

## 5. 复现指南

### 输入材料与入口

```text
Paper metadata: 学习/文献阅读/paper_meta.json
Full text:      学习/文献阅读/.papers_fulltext/unaligned-uav.txt
Paper PDF:      https://ojs.aaai.org/index.php/AAAI/article/view/38079/42041
Paper page:     https://ojs.aaai.org/index.php/AAAI/article/view/38079
Code&Datasets:  https://github.com/NOP1224/Unaligned_RGBT_Tracking（论文正文列出，未核验）
```

### 可复现的论文级配置

```yaml
Task: unaligned UAV RGBT tracking
Tracker: SFCATrack, based on OSTrack
Dataset: LUART
Train split: 1,010 sequences / 706,689 image pairs
Test split: 443 sequences / 310,669 image pairs
Input: original unaligned RGB/TIR images
Annotations: separate RGB and TIR bounding boxes
MSEE experts: 1 shared expert + 5 scale experts
Offset bins: [0,20), [20,35), [35,50), [50,75), [75,+∞)
MSEE loss: L1 offset regression
CMAF/tracker loss: weighted focal + generalized IoU + L1 (OSTrack formulation)
Metrics: PR / NPR / SR
```

### 建议的复现顺序（由论文内容直接展开）

1. 从论文给出的 Code&Datasets 入口取得 LUART、LasHeR-Unaligned 处理方式、标注格式和官方评测脚本；先核对 1,453/1.02M、1,010/706,689、443/310,669 等统计。
2. 保留 RGB 1920×1080 与 TIR 640×512 的原始分辨率关系，确认训练样本是视频序列级划分，而不是随机帧划分。
3. 实现 MSEE stage 1：共享 ResNet、shared expert、offset head，并用两个模态 box 的相对坐标差训练 L1。
4. 实现 MSEE stage 2：从 shared expert 初始化 5 个 scale experts，按论文给出的五个 offset bins 在对齐图像上模拟偏移并专门化训练。
5. 实现 MSEE stage 3：冻结 backbone 与 experts，训练 router 和 offset head，在完整训练集上选择 active expert。
6. 固定 MSEE 后实现 CMAF：12 层插入 shared-space MLP、fused guidance、级联 deformable convolution、gated fusion 和 residual update。
7. 先复现 Table 4 的 baseline/MSEE/CMAF 三行，再复现 Table 5 的 shared-only/scale-experts 消融，最后运行 LUART 与 LasHeR-Unaligned 的完整 OPE。

### 本次复现状态

- **未运行训练或测试。** 本次任务仅阅读全文、元数据与现有笔记，不下载数据、不执行仓库代码。
- **硬阻塞：** 论文正文没有提供安装命令、依赖版本、checkpoint、训练超参数完整表、评测脚本用法或硬件配置。
- **待核对项：** Code&Datasets 仓库当前内容、LUART 标注文件格式、双模态框的评测汇总方式、MSEE homography 的具体参数化、CMAF deformable block 数量 $N$、以及论文中的 $λ1/λ2$。

---

## 6. 批判性思考

### 优点

- **任务定义更真实：** 保留独立传感器的原始分辨率和空间偏移，分别标注两个模态，直接暴露真实部署中的对应关系问题。
- **对齐层级清楚：** MSEE 负责把目标带回同一个 search region，CMAF 负责修正区域内的非线性特征失配；两个模块的职责可解释且可分别消融。
- **尺度专家有明确训练协议：** 5 个 offset bins、shared expert 初始化和三阶段冻结策略让 MSEE 的设计不只是概念上的 MoE。
- **基准覆盖面大：** 1,453 对序列、1.02M 图像帧、42 类目标和 22 类挑战，且 train/test 在偏移分布上保持相似，适合检验大规模未对齐跟踪。
- **消融支持协同设计：** LUART 上 baseline→MSEE→CMAF 的 PR/SR 逐级提升，说明 image alignment 与 feature alignment 不是互相替代的重复模块。

### 局限

- **效率证据不完整：** 没有参数量、FLOPs、FPS、显存或路由调用率；“scale-aware adaptive selection”在精度上的证据充分，但在计算收益上的证据不足。
- **缺少细粒度协议结果：** 22 类挑战只有雷达图，offset 分布只有统计图，没有按偏移大小、TIR 出视野、旋转运动或小目标分别给出数值。
- **几何参数化较弱：** 用四个 box-coordinate differences 构造 homography 是强而简洁的先验，但对区域级视差、局部旋转和遮挡的表达能力需要 CMAF 补足；论文没有提供这类边界条件的专门实验。
- **路由分析不足：** 没有 fixed expert、soft mixture、oracle expert、router entropy 或专家调用频率对照，无法判断性能增益来自专家专门化、hard routing 还是二者共同作用。
- **外部验证仍有限：** LUART 是作者新建数据集，主要结论依赖作者定义的采集协议和重训设置；LasHeR-Unaligned 提供了补充验证，但仍不足以覆盖所有跨平台 UAV 视角变化。
- **复现信息不全：** 论文未给完整运行命令、环境、checkpoint 和关键超参数，代码入口也需要实际 release 核对。

### 我最关心的问题

1. **MSEE 的 shift 表达是否足够？** 四个 bbox 坐标差构造的 homography 在 TIR 目标部分出视野、旋转和跨视角视差下还能否稳定定位 search region？
2. **router 是否在边界处稳定？** `[20,35,50,75]` pixels 的分桶是工程选择还是由验证集确定？对接近边界的样本，top-1 路由是否频繁跳变？
3. **CMAF 的 deformable offset 是否有可解释的几何约束？** 如果 $F_{\mathrm{fuse}}$ 本身已被严重失准特征污染，CMAF 是否会把错误对应进一步传播到两个分支？
4. **双框评测如何汇总？** 论文强调分别预测 RGB/TIR boxes，但正文没有展开 PR/NPR/SR 对双模态输出的计算和汇总方式，这是复现时必须核对的协议细节。
5. **性能与计算如何联合衡量？** MSEE 的 active expert 选择是否实际减少延迟，CMAF 在 12 层全部启用的代价是多少？

### 可以迁移到 DAM4SAM memory management 的部分

> 以下均为迁移建议，不是本文已验证的结果。

- **Shift-conditioned memory write gate：** 在 DAM4SAM 写入新 memory 前增加一个轻量的跨视角/跨模态 shift estimator。估计偏移小且 expert 置信度高时正常写入；偏移大、TIR/辅助视角出视野或 router 不确定时，保留旧 memory，避免错误特征污染长期库。
- **粗对齐后再写入：** 借鉴 MSEE→CMAF 的顺序，先对当前帧特征做全局 shift/homography 校正，再做局部 deformable alignment，最后把校正后的 canonical feature 写入 memory。不要把明显未对齐的原始特征直接混入长期 memory。
- **memory read 的 shift compensation：** 当前查询与历史 memory 的视角有已知偏移时，用估计的 shift 先对 query/key 的采样位置做预移位，再执行 SAM2/DAM4SAM 的 memory attention；CMAF 式局部采样可作为粗几何之后的残差对齐。
- **按失准尺度分层管理：** 将 memory 条目按小、中、大、长尾 shift 或视角差分桶，检索时优先访问当前难度对应的 memory 子库；这样比把所有帧放入一个 FIFO bank 更容易控制长视频中的混合误差。
- **把路由不确定性变成 memory 健康信号：** 记录 shared expert 与 active expert 的置信度、shift 大小和当前/历史特征相似度；高不确定性帧只用于短期缓存或待确认队列，不直接更新稳定模板。
- **双库而非单库：** 对 DAM4SAM 可分别维护 canonical target memory 与 view-specific residual memory。前者经过全局对齐，后者保留局部外观变化；类似 MSEE/CMAF 的两层职责，避免用单一 memory 同时承载几何与外观残差。

#### 迁移边界

MSEE 监督的是 RGB/TIR 两个 bounding boxes 的相对坐标，DAM4SAM 的 memory 对齐可能面对跨帧、跨视角和遮挡，不一定存在同样的双框 ground truth。因此 MSEE 的四参数 homography 更适合作为**粗初始化或门控信号**，不能直接假设它能解决跨视角 UAV 的视差与非刚性变化。

### 可以迁移到 cross-view UAV tracking 的部分

- **三层误差分解：** 将跨视角失配拆为全局平移/视差、尺度与旋转、局部透视或非线性残差；先用低成本几何模块把目标带入共同 search region，再让 deformable feature alignment 处理局部残差。
- **视角差感知路由：** 为小视角差保留轻量 shared/匹配专家；只有出现大视差、遮挡、低分辨率或跨模态质量下降时才激活更重的细节专家。路由输入可加入目标尺度、可见区域比例和跨视角匹配置信度。
- **几何引导的跨视角采样：** 先用相机标定、稀疏匹配或 box-level shift 生成 coarse transform，再将其作为 deformable sampling 的初始位置；可学习偏移只修正局部误差，避免在大视差上盲目搜索。
- **跨视角 benchmark protocol：** 保留原始多视角图像，不做隐式人工配准；为每个视角分别标注 box/mask，按序列和场景划分 train/test，并报告视差区间、目标尺度、遮挡、出视野和旋转属性下的 PR/NPR/SR 或对应跟踪指标。
- **失败驱动的 memory 管理：** 把“视角变化导致的对齐不确定性”接到 DAM4SAM 的写入策略，而不是只把它当作当前帧的定位误差；当 coarse transform 与 local residual 长期不一致时触发重检测或 memory refresh。

### 新想法

1. **Shift-aware DAM4SAM Memory Bank：** 为 memory 条目增加 `shift_bin`、视角差、可见比例和对齐置信度，检索时按当前查询的几何状态筛选，而不是仅按时间或 cosine similarity 排序。
2. **双阶段 memory update：** `coarse warp → deformable residual alignment → confidence gate → write`。如果局部残差过大或 gate 置信度低，只更新短期 memory，不更新长期 canonical memory。
3. **Router disagreement 作为漂移预警：** 让 shared expert 与多个 scale/view experts 输出候选 shift，使用候选之间的分歧和 router entropy 作为漂移信号；分歧增大时冻结长期 memory 并提高重检测频率。
4. **跨视角双向一致性：** 在 view A→B 和 B→A 都执行 coarse transform 与 local alignment，加入 cycle consistency 检查；只把满足双向一致性的对应写入长期 memory。
5. **应补的实验矩阵：** 对 DAM4SAM 分别报告小/中/大 shift、视角差、TIR/RGB 出视野、旋转、遮挡和小目标条件下的 memory contamination rate、重检测率、跟踪 PR/SR 与延迟，而不是只报告一个总平均分。

---

## 7. 深度阅读标注

本节暂无额外阅读标注。

### 逐段精读

- **[事实｜Abstract，PDF p. 11014]** 任务要求在原始未对齐 RGB/TIR 图像上预测两个模态的目标框；LUART 有 1,453 对序列、42 类目标、22 类挑战和不同偏移尺度；SFCATrack 由 MSEE 与 CMAF 构成。
- **[事实｜Introduction，PDF pp. 11014–11015]** UAV 传感器差异不仅带来位置平移，还会产生 TIR 目标出视野与旋转运动；LasHeR-Unaligned 的主要失准形式更简单。
- **[事实｜Dataset Construction，PDF p. 11016]** LUART 使用无人机实拍、保留 RGB 1920×1080 与 TIR 640×512；训练集 1,010 个序列、706,689 对帧，测试集 443 个序列、310,669 对帧。
- **[事实｜Modality Misalignment Statistics，PDF p. 11016]** 偏移越大，帧数总体越少；训练集与测试集在不同 offset ranges 上的分布相近。
- **[事实｜MSEE Eq. (1)–(4)，PDF p. 11017]** shared ResNet 提取两模态特征；5 个 scale experts 与一个 shared expert 共同产生 shift estimation 表示；router 选择 active expert；offset 是两个 boxes 的四个坐标差。
- **[事实｜MSEE staged training，PDF p. 11017]** scale experts 用五个偏移区间专门化，最后冻结 backbone/experts，仅训练 router 与 offset head 完成自适应选择。
- **[事实｜CMAF Eq. (5)–(8)，PDF pp. 11017–11018]** patch embedding 后进入 12-layer Transformer；CMAF 每层用 shared fusion feature 引导级联 deformable convolution，再用 gating network 融合并 residual add。
- **[事实｜Table 4–5，PDF p. 11019]** LUART 上 baseline→MSEE→CMAF 的 PR/SR 为 48.6/38.3→53.2/41.5→57.3/44.6；shared expert 与 scale experts 的逐级收益也被单独报告。
- **[推断｜从 MSEE 与 CMAF 的接口]** MSEE 主要解决“搜索区域是否包含对应目标”，CMAF 主要解决“区域内特征是否逐点可融合”；这也是把该方法迁移到 memory management 时应保留的职责分界。
- **[推断｜从 LUART 的长尾偏移与 top-1 routing]** `[75,+∞)` 桶可能混合多种严重失准，若 router 置信度在边界处不稳定，错误 expert 选择可能导致 warp 误差；需要 release 后做分桶细化与专家调用分析。
- **[推断｜从 TIR out-of-view 挑战]** 出视野属于观测缺失，不应只用 alignment loss 处理；DAM4SAM 中应把它转化为 memory write suppression / re-detection 触发条件。

### 读后核对清单

- [ ] 核对官方仓库是否包含 MSEE 三阶段训练与五个 offset bins。
- [ ] 核对 homography 的具体参数化，以及四个 box-coordinate differences 如何映射到 warp。
- [ ] 核对 CMAF deformable convolution 的级联次数、offset 生成方式和 gate 输入。
- [ ] 核对 LUART 双模态预测框的评测汇总方式。
- [ ] 按 offset range、TIR 出视野、旋转和小目标重新统计失败率与专家选择频率。

---

## 8. 总结

### 三句话总结

1. **Problem：** UAV 的独立 RGB/TIR 传感器造成动态、跨分辨率、跨视角的空间失准；传统依赖人工对齐和共享框的 RGBT tracker 无法直接处理原始未对齐输入。
2. **Method：** SFCATrack 用 MSEE 的 shared expert + 五个尺度专家估计四维 box shift 并 warp TIR search region，再用 CMAF 在 12 层 backbone 中以融合特征引导级联 deformable alignment 和 gated fusion。
3. **Result：** LUART 包含 1,453 对序列、1.02M 对帧、42 类目标和 22 类挑战；SFCATrack 在 LUART 上达到 57.3/51.9/44.6（PR/NPR/SR），在 LasHeR-Unaligned 上达到 60.7/55.1/47.9，并在组件消融中显示 MSEE 与 CMAF 的互补收益。

### 一句话评价

一个把“未对齐 UAV RGBT 跟踪”明确拆成**全局 shift 校正 + 局部特征对齐**的实用基线；LUART 的任务与数据协议很有价值，但缺少效率指标、细粒度失败分析和完整复现配置，结论强度仍受源码与评测协议核验限制。

### 是否值得复现？

**复现理由：** 三星半（若只使用整数星则记为 ⭐⭐⭐）。LUART 的双模态独立标注和真实无人机失准场景对跨视角 UAV tracking 很有参考价值，MSEE 的 shift-aware routing 与 CMAF 的 coarse-to-fine alignment 也适合迁移到 DAM4SAM 的 memory gating；但论文没有给出完整运行配置，且本次尚未核验 Code&Datasets release。建议优先复现 Table 4/5 的 MSEE、CMAF 消融，再做按偏移/出视野/旋转条件的失败矩阵，而不是一开始直接追求完整训练复现。

---

### 参考资料

- [AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/38079)
- [AAAI paper PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38079/42041)
- [Code&Datasets entry listed in the paper](https://github.com/NOP1224/Unaligned_RGBT_Tracking)（本次未核验仓库内容）
