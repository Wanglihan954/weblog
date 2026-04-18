---
title: A Gentle Introduction to HiPPO and its Friends
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
mathjax: true
abbrlink: 74aed18
date: 2025-12-26 20:20:20
---

## 序言

考虑一个很长的一维序列，当我们希望模型具有 “记忆” 能力的时候，我们实际上期望模型能在当前时间步对很久以前的时间步上的数据点具有无损恢复的能力。然而，模型的大小是不可能随序列长度增长的，也就是说我们需要使用有限的参数恢复出无限多时间步的数据点。

很显然无损恢复是不可能的。但是，我们可以用一种具有渐进收敛性的方法，尽可能减少这种损失。我们面临的第一个问题就是，怎样描述这个损失？一个非常自然的想法是直接把我们恢复出来的数据点和真实的数据点的距离求 L2 范数。Naive 的 L2 范数假设了所有历史数据点是同等重要的。

到这里，我们起码有了一种最简单的求损失的方法。如果我们把所有真实数据点看作对时间步的函数，那么我们上面求损失的过程正是函数逼近的过程，即我们不知道真实函数的表达式，但我们获取了它的若干采样数据点，我们可以依赖这些数据点，选取一个已知表达式的函数来逼近它。这个用于逼近的函数包含有限多的待优化的参数，而参数的数量不随序列长度变化。
## 1. 核心思想

HiPPO 提供了一个数学框架，通过将输入信号 $f(t)$ 投影到正交多项式基上，来实现在固定维度的状态中保留无限长的历史信息。

### 核心机制：

- **投影 (Projection)**：将过去的所有输入信号投影到由正交多项式构成的系数空间。
- **在线更新 (Online Update)**：随着新输入 $f(t)$ 的到来，实时更新这些系数，而无需重新计算整个历史。


## 2. 数学框架

HiPPO 的核心是一个线性时不变系统 (LTI)：

$$\dot{c}(t) = A c(t) + B f(t)$$

其中：

- $c(t) \in \mathbb{R}^N$ 为存储记忆的系数向量。
  
- $f(t)$ 为输入信号。
  
- $A \in \mathbb{R}^{N \times N}$ 和 $B \in \mathbb{R}^{N \times 1}$ 是预先定义的矩阵。
  

### LegS (Legendre Measures)

对于 Legendre 度量（给予历史信息相同的权重），其 $A$ 矩阵定义如下：

$$A_{nk} = \begin{cases} -(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ -(n+1) & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

$$B_n = (2n+1)^{1/2}$$

---

## 3. 离散化 (Discretization)

为了在计算机中处理，需要将连续的微分方程转化为离散的递归形式。通常采用 **双线性变换 (Bilinear Transform)**：

$$c_t = A_d c_{t-1} + B_d f_t$$

**离散化公式：**

1. $A_d = (I - \frac{\Delta t}{2} A)^{-1}(I + \frac{\Delta t}{2} A)$
   
2. $B_d = (I - \frac{\Delta t}{2} A)^{-1} \Delta t B$
   

---

## 4. 架构模型实现 

根据[不会魔法的小圆的博客](https://anti-entrophic.github.io/posts/10038.html)中的架构设计，HiPPO 作为一个记忆单元嵌入到Gated RNN的递归结构中：
![image.png](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/20251226185343976.png)
### 更新步骤：

1. **融合输入**：$h_t = \tau(h_{t-1}, [c_{t-1}, x_t])$
   
    - $\tau$ 可以是普通的 RNN 单元或 GRU。
      
    - 它整合了当前输入 $x_t$ 和来自 HiPPO 的长短期记忆 $c_{t-1}$。
    
2. **生成投影信号**：$f_t = \text{Linear}(h_t)$
   
    - 从当前的隐藏状态中提取关键信息。
    
3. **更新 HiPPO 记忆**：$c_t = A_d c_{t-1} + B_d f_t$
   
    - 利用固定的 HiPPO 矩阵将信号 $f_t$ 压缩进记忆空间。
## 5. 代码实现 (PyTorch)

### HiPPO 矩阵生成

```Python
# 1. HiPPO 核心逻辑实现
def get_HiPPO_LegS(N):
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
    A = P.unsqueeze(1) * P.unsqueeze(0)
    A = -torch.tril(A, -1) - torch.diag(torch.arange(N, dtype=torch.float32) + 1.0)
    return A, P

def discretize_legs(A, B, dt):
    I = torch.eye(A.shape[0])
    BL = torch.inverse(I - (dt / 2.0) * A)
    Ad = BL @ (I + (dt / 2.0) * A)
    Bd = (BL @ (dt * B).view(-1, 1)).squeeze()
    return Ad, Bd

```

### 混合模型前向传播

```Python
# --- 根据图片架构实现的新模型 ---
class HiPPO_Hybrid_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hippo_dim, dt=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hippo_dim = hippo_dim
        
        # 1. 记忆矩阵 A, B (固定)
​        Ad, Bd = discretize_legs(*get_HiPPO_LegS(hippo_dim), dt)
​        self.register_buffer('Ad', Ad)
​        self.register_buffer('Bd', Bd.unsqueeze(1))
​        
        # 2. 隐藏状态更新函数 tau
        # 输入包括: 当前输入 x_t (input_dim) 和 上一时刻记忆 c_{t-1} (hippo_dim)
        # 上一时刻隐藏状态 h_{t-1} 由 RNNCell 内部处理
​        self.rnn_cell = nn.RNNCell(input_dim + hippo_dim, hidden_dim)
​        
        # 3. 特征提取层 f_t = L_f(h_t)
        # 将隐藏状态映射到 1 维信号，存入 HiPPO 记忆
​        self.f_proj = nn.Linear(hidden_dim, 1)
​        
        # 4. 输出映射 (可选)
​        self.output_layer = nn.Linear(hidden_dim, 1)

​    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
​        batch_size, seq_len, _ = x.shape
​        device = x.device
​        
        # 初始化隐藏状态和记忆状态
​        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
​        c_t = torch.zeros(batch_size, self.hippo_dim, device=device)
​        
​        outputs = []
​        
​        for i in range(seq_len):
​            x_t = x[:, i, :]
​            
            # --- 步骤 1: 更新 h_t ---
            # h_t = tau(h_{t-1}, [c_{t-1}, x_t])
            # 将 x_t 和 c_t 拼接作为当前输入
​            rnn_input = torch.cat([x_t, c_t], dim=-1)
​            h_t = self.rnn_cell(rnn_input, h_t)
​            
            # --- 步骤 2: 提取特征 f_t ---
            # f_t = L_f(h_t)
​            f_t = self.f_proj(h_t)
​            
            # --- 步骤 3: 更新 HiPPO 记忆 c_t ---
            # c_t = A c_{t-1} + B f_t
            # 注意：Ad 是 [N, N], Bd 是 [N, 1]
​            c_t = torch.matmul(c_t, self.Ad.T) + f_t @ self.Bd.T
​            
            # 保存输出
​            y_t = self.output_layer(h_t)
​            outputs.append(y_t.unsqueeze(1))
​            
​        return torch.cat(outputs, dim=1)
```
### 随机函数逼近可视化
~~~python
# --- 实验设置 ---
L = 500  # 序列长度
N = 64   # 隐藏维度

# --- 替换部分：生成类似图中黑色的噪声序列 (White Noise / Random Signal) ---
# 使用标准正态分布生成高频信号，并模拟图中的波动幅度
torch.manual_seed(42) # 保持结果可复现
noise_signal = torch.randn(L) * 0.5 

# 模拟图中 Input f 的特性：让序列在 [-1.5, 1.5] 之间震荡
inputs_raw = noise_signal.view(1, L, 1)

# 构造错位预测任务
inputs = inputs_raw[:, :-1, :]  # t 时刻的噪声
target = inputs_raw[:, 1:, :]   # 预测 t+1 时刻的噪声

# 横坐标对齐图片 (0 到 1)
t_plot = torch.linspace(0, 1, L-1) 
# ------------------------------------------------------------
# 4. 训练与对比
models = {
    "HiPPO-Hybrid-RNN": HiPPO_Hybrid_RNN(input_dim=1, hidden_dim=128, hippo_dim=N, dt=1.0/L),
    "Custom-GRU": GRUModel(1, N, 1), # 请确保你有定义该类
}
plt.figure(figsize=(12, 6))
# 绘制黑色虚线作为输入基准，匹配你提供的图片风格
plt.plot(t_plot.numpy(), inputs.squeeze().numpy(), 'k-', alpha=0.8, linewidth=1, label='Input function f (Noise)')
import tqdm
for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 针对噪声序列，建议至少迭代 50-100 次，否则模型无法捕捉到任何预测逻辑
    for epoch in tqdm.tqdm(range(100)):
        pred = model(inputs)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 可视化结果
    with torch.no_grad():
        final_pred = model(inputs).squeeze().numpy()
        plt.plot(t_plot.numpy(), final_pred, label=f'{name} Prediction', alpha=0.8)

plt.title("HiPPO vs GRU: High-Frequency Noise Prediction")
plt.xlabel("Time (normalized)")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
~~~
![image.png](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/20251226200829763.png)

## 6. 总结

HiPPO 的优势在于其**数学确定性**。通过将 A 和 B 矩阵初始化为 LegS 形式并保持冻结，模型被赋予了一个“内置”的正交投影机制，这使得它在处理超长序列（如几千个时间步）时比单纯学习权重的 RNN 更加稳健。