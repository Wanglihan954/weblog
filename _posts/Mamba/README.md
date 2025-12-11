---
title: Introduction
categories:
  - Mamba
author: 宁翰
email: 314375980@qq.com
tags:
  - python
  - Mamba
  - DeepLearning
readmore: true
hideTime: true
abbrlink: 140cab1e
date: 2025-12-03 20:00:39
---



## What is Mamba

Mamba 是一种基于结构化状态空间序列模型（SSMs）的新兴架构，旨在高效捕捉序列数据中的复杂依赖性，成为 Transformer 的强大竞争对手。受经典状态空间模型启发，Mamba 融合了[循环神经网络](https://zhida.zhihu.com/search?content_id=254403907&content_type=Article&match_order=1&q=循环神经网络&zhida_source=entity)（RNN）和[卷积神经网络](https://zhida.zhihu.com/search?content_id=254403907&content_type=Article&match_order=1&q=卷积神经网络&zhida_source=entity)（CNN）的特点，通过递归或卷积操作实现计算成本与序列长度的线性或近线性扩展，显著降低计算复杂度。

具体而言，Mamba 的核心优势包括：

1. **选择机制**：引入简单而有效的选择机制，通过输入参数化 SSM 参数，过滤无关信息，保留必要数据。
2. **硬件感知算法**：采用递归扫描而非卷积计算，优化硬件性能，在 A100 GPU 上实现高达 3 倍的加速。
3. **建模能力**：在保持与 Transformer 相当的建模能力的同时，具备近线性的可扩展性，适用于复杂和长序列数据。

这些特性使 Mamba 成为处理多领域任务的理想基础模型，已在计算机视觉、自然语言处理和医疗保健等领域展现出卓越性能。例如，Vim 模型在高分辨率图像特征提取中比 DeiT 快 2.8 倍，节省 86.8% 的 GPU 内存；而在语言建模任务中，改进的选择性 SSM 架构实现了 2-8 倍的加速。Mamba 的高效性和灵活性使其有望在多个研究和应用领域引发革命性变革。
![image.png](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/20251129142057745.png)



## Mamba的前世今生

>1. RNN作为第一代序列模型，奠定了序列处理的基础；
>2. HiPPO作为理论框架，为RNN的长程依赖问题提供了数学上的最优记忆解决方案；
>3. S4将HiPPO的连续时间理论工程化，通过离散化处理使其能够处理实际中的离散序列数据，同时引入了可训练的参数矩阵；
>4. Mamba（S6）在S4基础上革新性地引入了动态选择性机制，使模型参数能够根据输入内容自适应调整，从而实现了接近注意力机制的内容感知能力，同时保持了线性计算复杂度。

## Mamba的**基础知识**

Mamba与循环框架的循环神经网络（RNNs）、并行计算和Transformer的注意力机制以及状态空间模型（SSMs）的线性属性密切相关。因此，本节旨在介绍这三种突出架构的概述。

