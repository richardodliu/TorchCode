## 激活函数

### 1. ReLU

介绍：ReLU 是逐元素激活函数，定义为 $\max(0,x)$。它保留正值、截断负值，常用于缓解 sigmoid/tanh 在正区间的梯度饱和问题。实现时不要把 Tensor 转成 Python 标量或列表，直接用布尔 mask 保持 autograd。

公式：$\operatorname{ReLU}(x)=\max(0,x)$；$\frac{d}{dx}\operatorname{ReLU}(x)=\mathbf{1}_{x>0}$。

- Q：ReLU 的导数是什么？
  A：$x>0$ 时导数为 1，$x<0$ 时导数为 0；$x=0$ 处不可导，框架通常取一个约定的次梯度。
- Q：什么是 dead ReLU？
  A：如果神经元长期落在负半轴，梯度一直为 0，参数很难再更新，这种现象叫 dead ReLU。
- Q：为什么 $x\cdot\mathbf{1}_{x>0}$ 仍然支持 autograd？
  A：mask 本身不需要梯度，但乘法对 `x` 的正区间保留梯度，负区间梯度被置 0。

```python
def relu(x: Tensor) -> Tensor:
    return x * (x > 0).to(x.dtype)


def relu_grad(x: Tensor) -> Tensor:
    return (x > 0).to(x.dtype)
```

### 2. Sigmoid

介绍：Sigmoid 是把实数映射到 $(0,1)$ 的激活函数，公式是 $\sigma(x)=\frac{1}{1+e^{-x}}$。它常用于二分类输出层，把 logit 转成正类概率；隐藏层中较少使用，因为输入绝对值较大时容易梯度饱和。

公式：$\sigma(x)=\frac{1}{1+e^{-x}}$；$\sigma'(x)=\sigma(x)(1-\sigma(x))$。

- Q：Sigmoid 的输出范围是什么？
  A：输出范围是 $(0,1)$，因此常被解释为概率。
- Q：Sigmoid 的主要缺点是什么？
  A：输入很大或很小时会饱和，梯度接近 0，容易导致梯度消失。
- Q：Sigmoid 常用在哪里？
  A：二分类输出层、多标签分类的每个独立标签概率、门控结构中的 gate。

```python
def sigmoid(x: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def sigmoid_grad(x: Tensor) -> Tensor:
    y = sigmoid(x)
    return y * (1.0 - y)
```

### 3. Tanh

介绍：Tanh 把实数映射到 $(-1,1)$，公式是 $\frac{e^x-e^{-x}}{e^x+e^{-x}}$。相比 Sigmoid，Tanh 是零中心输出，在一些 RNN 或传统神经网络中更常见，但同样存在饱和区梯度变小的问题。

公式：$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$；$\tanh'(x)=1-\tanh^2(x)$。

- Q：Tanh 和 Sigmoid 的主要区别是什么？
  A：Tanh 输出范围是 $(-1,1)$ 且零中心；Sigmoid 输出范围是 $(0,1)$ 且不是零中心。
- Q：Tanh 为什么也会梯度消失？
  A：当输入绝对值较大时，输出接近 -1 或 1，导数接近 0。
- Q：Tanh 常见在哪里？
  A：传统 RNN 状态更新、需要有界且零中心输出的中间层。

```python
def tanh(x: Tensor) -> Tensor:
    exp_pos = torch.exp(x)
    exp_neg = torch.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_grad(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1.0 - y.pow(2)
```

### 4. Leaky ReLU

介绍：Leaky ReLU 是 ReLU 的变体，正半轴保持 `x`，负半轴保留一个很小斜率 $\alpha x$，其中 $\alpha$ 对应代码里的 `negative_slope`。它用于缓解 ReLU 在负半轴梯度为 0 导致的 dead ReLU 问题。

公式：$\operatorname{LeakyReLU}_{\alpha}(x)=\begin{cases}x,&x>0\\ \alpha x,&x\le 0\end{cases}$；$\frac{d}{dx}\operatorname{LeakyReLU}_{\alpha}(x)=\begin{cases}1,&x>0\\ \alpha,&x\le 0\end{cases}$。

- Q：Leaky ReLU 和 ReLU 的区别是什么？
  A：ReLU 负半轴输出 0；Leaky ReLU 负半轴保留一个小斜率。
- Q：Leaky ReLU 能解决什么问题？
  A：能缓解 dead ReLU，让负半轴也有梯度可以更新参数。
- Q：`negative_slope` 一般取多少？
  A：常见默认值是 `0.01`，也可以作为超参数调整。

```python
def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return torch.where(x > 0, x, negative_slope * x)


def leaky_relu_grad(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, negative_slope))
```

### 5. Softplus

介绍：Softplus 是 ReLU 的平滑近似，公式是 $\operatorname{Softplus}(x)=\log(1+e^x)$。它的输出恒为正，$x$ 很大时接近 $x$，$x$ 很小时接近 0；导数是 Sigmoid，因此没有 ReLU 在 0 点不可导的问题。

公式：$\operatorname{Softplus}(x)=\log(1+e^x)$；$\frac{d}{dx}\operatorname{Softplus}(x)=\sigma(x)$。

- Q：Softplus 的公式是什么？
  A：$\operatorname{Softplus}(x)=\log(1+e^x)$。
- Q：Softplus 和 ReLU 有什么关系？
  A：Softplus 可以看作 ReLU 的平滑版本；大正数区域接近线性，大负数区域接近 0，但过渡更平滑。
- Q：Softplus 的导数是什么？
  A：导数是 $\sigma(x)=\frac{1}{1+e^{-x}}$，所以它在所有位置都可导。

```python
def softplus(x: Tensor) -> Tensor:
    return torch.log(1.0 + torch.exp(x))


def softplus_grad(x: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-x))
```

### 6. GELU

介绍：GELU（Gaussian Error Linear Unit）是平滑激活函数，可以理解为用高斯分布的累积分布函数对输入做门控。它常用于 Transformer、BERT、GPT 等模型，相比 ReLU 的硬截断更平滑。

公式：$\operatorname{GELU}(x)=x\Phi(x)=\frac{1}{2}x\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$。

- Q：GELU 的精确公式是什么？
  A：$\operatorname{GELU}(x)=\frac{1}{2}x\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$。
- Q：GELU 和 ReLU 有什么区别？
  A：ReLU 是硬截断负半轴；GELU 是平滑门控，负区间也可能保留很小输出。
- Q：GELU 常见在哪里？
  A：Transformer 的 MLP/FFN 中很常见，例如 BERT、GPT 系列和很多现代大模型变体。

```python
def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

### 7. SiLU / Swish

介绍：SiLU 也叫 Swish 的 $\beta=1$ 版本，公式是 $x\cdot\sigma(x)$。它是平滑门控激活，负半轴不会被硬截断，常见于现代 CNN 和 Transformer 变体中，例如 EfficientNet、SwiGLU 类结构。

公式：$\operatorname{SiLU}(x)=x\sigma(x)$；$\operatorname{SiLU}'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))$；$\operatorname{Swish}(x)=x\sigma(\beta x)$；$\operatorname{Swish}'(x)=\sigma(\beta x)+\beta x\sigma(\beta x)(1-\sigma(\beta x))$。

- Q：SiLU 的公式是什么？
  A：$\operatorname{SiLU}(x)=x\cdot\sigma(x)$。
- Q：SiLU 和 ReLU 有什么区别？
  A：SiLU 是平滑函数，负区间仍有非零输出；ReLU 是分段线性硬截断。
- Q：Swish 和 SiLU 的关系是什么？
  A：Swish 通常写作 $x\cdot\sigma(\beta x)$；当 $\beta=1$ 时就是 SiLU。

```python
def silu(x: Tensor) -> Tensor:
    return x * (1.0 / (1.0 + torch.exp(-x)))


def silu_grad(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + torch.exp(-x))
    return s + x * s * (1.0 - s)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * (1.0 / (1.0 + torch.exp(-beta * x)))


def swish_grad(x: Tensor, beta: float = 1.0) -> Tensor:
    s = 1.0 / (1.0 + torch.exp(-beta * x))
    return s + beta * x * s * (1.0 - s)
```

## 归一化 Norm

### 8. BatchNorm

介绍：BatchNorm 训练时使用当前小批量的均值和方差，并更新运行均值和运行方差；推理时使用保存下来的运行统计量。运行统计量不是模型梯度路径的一部分，更新时应 detach。

公式：$\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$；$y=\gamma\hat{x}+\beta$；$s_t=(1-m)s_{t-1}+m\hat{s}_t$。

- Q：BatchNorm 训练时和推理时用什么统计量？
  A：训练时用当前小批量的均值和方差，并更新运行统计量；推理时使用运行均值和运行方差。它和 LayerNorm 不同，训练和推理行为不完全一致。
- Q：BatchNorm 的动量系数表示什么？
  A：运行统计量的指数滑动更新比例：$s_t=(1-m)s_{t-1}+m\hat{s}_t$，其中 $m$ 是动量系数，$\hat{s}_t$ 是当前小批量统计量。
- Q：为什么小批量下 BatchNorm 可能不稳定？
  A：小批量统计量估计噪声大，均值和方差不稳定，会影响训练和推理一致性；这种场景下 LayerNorm/GroupNorm 往往更稳。

```python
def my_batch_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
) -> Tensor:
    if training:
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * var.detach())
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

### 9. GroupNorm

介绍：GroupNorm 把通道维分成若干组，在每个样本、每个组内计算均值和方差。它不依赖 batch 维统计量，因此小 batch、检测、分割等场景中常比 BatchNorm 更稳定。

公式：$\mu_g=\frac{1}{M}\sum_{i\in g}x_i$；$\hat{x}_i=\frac{x_i-\mu_g}{\sqrt{\sigma_g^2+\epsilon}}$；$y_i=\gamma_i\hat{x}_i+\beta_i$。

- Q：GroupNorm 和 BatchNorm 的核心区别是什么？
  A：BatchNorm 跨 batch 统计，GroupNorm 在单个样本的通道组内统计，因此不依赖 batch size。
- Q：GroupNorm 和 LayerNorm 有什么关系？
  A：GroupNorm 按通道分组；当组数为 1 时接近 LayerNorm，当组数等于通道数时接近 InstanceNorm。
- Q：小 batch 训练为什么常用 GroupNorm？
  A：小 batch 下 BatchNorm 统计噪声大，GroupNorm 不看 batch 维，训练和推理行为更一致。

```python
def my_group_norm(x: Tensor, gamma: Tensor, beta: Tensor, num_groups: int, eps: float = 1e-5) -> Tensor:
    B, C = x.shape[:2]
    assert C % num_groups == 0
    x_grouped = x.view(B, num_groups, C // num_groups, *x.shape[2:])
    dims = tuple(range(2, x_grouped.dim()))
    mean = x_grouped.mean(dim=dims, keepdim=True)
    var = x_grouped.var(dim=dims, keepdim=True, unbiased=False)
    x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.view_as(x)
    shape = (1, C) + (1,) * (x.dim() - 2)
    return x_norm * gamma.view(shape) + beta.view(shape)
```

### 10. LayerNorm

介绍：LayerNorm 对每个样本的最后一维计算均值和方差，然后做标准化和仿射变换。它不依赖小批量统计，因此训练和推理行为一致，常用于 Transformer。

公式：$\mu=\frac{1}{d}\sum_{i=1}^{d}x_i$；$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$；$y_i=\gamma_i\hat{x}_i+\beta_i$。

- Q：LayerNorm 归一化的是哪一维？
  A：常见 Transformer 中对每个 token 的 hidden dimension 归一化，也就是最后一维。
- Q：LayerNorm 训练和推理有什么区别？
  A：没有区别；它使用当前样本自身的统计量，不维护运行均值和运行方差。
- Q：LayerNorm 和 BatchNorm 的核心区别是什么？
  A：BatchNorm 依赖小批量统计，LayerNorm 依赖单个样本的特征维统计；因此 Transformer 更常用 LayerNorm，因为它不受批大小和序列长度变化影响。

```python
def my_layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

### 11. RMSNorm

介绍：RMSNorm 只按均方根缩放，不减均值：$\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}$。它比 LayerNorm 更简单，常见于 LLaMA、Gemma 等大模型。

公式：$y=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}$。

- Q：RMSNorm 和 LayerNorm 差在哪里？
  A：RMSNorm 不减均值，只用均方根控制尺度；LayerNorm 同时减均值并除标准差。
- Q：RMSNorm 为什么要加 `eps`？
  A：防止 RMS 非常小或为 0 时除零，同时提高数值稳定性。
- Q：为什么很多 LLM 使用 RMSNorm？
  A：它计算更简单、参数更少，在大模型中通常能保持稳定训练效果。

```python
def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

## 基础层

### 12. Softmax

介绍：Softmax 把 logits 沿指定维度归一化成概率分布。稳定实现需要先减去该维度最大值，再做指数和归一化；减常数不会改变结果，但可以避免 $\exp$ 上溢。

公式：$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$。

- Q：为什么 Softmax 要减去最大值？
  A：$\exp$ 对大数容易上溢；减去同一维度最大值不改变概率比例，但能把最大指数项变成 1。
- Q：Softmax 的 `dim` 应该怎么选？
  A：对哪一维做概率归一化，就把 `dim` 设为哪一维；分类 logits 通常沿类别维归一化。
- Q：temperature 对 Softmax 有什么影响？
  A：logits 除以较小 temperature 会让分布更尖锐，较大 temperature 会让分布更平滑。

```python
def my_softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 13. Linear Layer

介绍：线性层计算 $y=xW^T+b$。PyTorch 约定权重形状通常是 `(out_features, in_features)`，因此前向传播使用 $xW^T+b$。参数需要是可训练 Tensor，通常用 `nn.Parameter` 注册。

公式：$y=xW^T+b$。

- Q：为什么 Linear 的权重常写成 `(out_features, in_features)`？
  A：这样每个输出神经元对应权重矩阵的一行，前向计算写作 $xW^T+b$。
- Q：为什么要用 `nn.Parameter`？
  A：`nn.Parameter` 会被 `nn.Module` 注册为可训练参数，优化器才能通过 `model.parameters()` 找到它。
- Q：初始化为什么要除以 $\sqrt{d_{\mathrm{in}}}$？
  A：这是为了控制输出方差，避免输入维度变大时激活尺度过大。

```python
class SimpleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias
```

### 14. Embedding

介绍：Embedding 是一个可学习查表矩阵，输入 token id，输出对应向量。它等价于 one-hot 乘线性层，但直接索引更高效。

公式：$e_i=E_i$；$e_i=o_iE$。

- Q：Embedding 和 one-hot 乘矩阵有什么关系？
  A：Embedding 查表等价于 one-hot 向量乘 embedding 矩阵，但查表更省计算和内存。
- Q：Embedding 的梯度更新有什么特点？
  A：通常只有被索引到的行会产生梯度并更新。
- Q：语言模型里 embedding 常见的工程点有哪些？
  A：padding index、词表扩展、输入输出 embedding weight tying。

```python
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, indices: Tensor) -> Tensor:
        return self.weight[indices]
```

### 15. Dropout

介绍：Dropout 训练时随机置零部分激活，并对保留值乘 $1/(1-p)$ 保持期望不变；推理时直接返回输入。这种形式叫 inverted dropout。

公式：$\tilde{x}=\frac{m\odot x}{1-p}$；$m_i\sim\operatorname{Bernoulli}(1-p)$。

- Q：Dropout 训练和推理有什么区别？
  A：训练时随机置零并缩放保留项；推理时直接返回输入。
- Q：为什么训练时要除以 $1-p$？
  A：保持激活的期望不变，使推理时不需要额外缩放。
- Q：Dropout 会阻断梯度吗？
  A：被 mask 掉的位置梯度为 0，保留位置按缩放系数正常传递梯度。

```python
class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return x * mask / (1 - self.p)
```

### 16. Output Head

介绍：Output Head 是接在 backbone、encoder 或 decoder 后面的任务输出层，用来把隐藏表示映射到任务需要的输出空间。分类 head 输出类别 logits，语言模型 LM head 输出词表 logits，回归 head 输出连续值。训练时通常让 head 输出未归一化 logits，把 softmax、sigmoid 或数值稳定处理交给对应 loss。

公式：$z=hW^T+b$；分类任务 $p=\operatorname{softmax}(z)$；语言模型 $z_t=h_tE^T+b$。

- Q：Output Head 和 backbone 的区别是什么？
  A：backbone 负责提取通用表示，output head 负责把表示投影到具体任务空间，例如类别、词表或回归目标。
- Q：Output Head 里通常要不要手动加 softmax？
  A：训练时通常不加，直接输出 logits；`CrossEntropyLoss`、`BCEWithLogitsLoss` 等会把归一化和 log 合并得更稳定。
- Q：LM head 的 weight tying 是什么？
  A：让输出投影矩阵和输入 embedding 矩阵共享权重，减少参数量，并让输入 token 表示和输出词表打分使用同一套向量空间。

```python
class OutputHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout_p: float = 0.0, bias: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.proj = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(self.dropout(hidden))


def tied_lm_head(hidden: Tensor, embedding_weight: Tensor, bias: Tensor | None = None) -> Tensor:
    logits = hidden @ embedding_weight.T
    if bias is not None:
        logits = logits + bias
    return logits
```

## 损失函数

### 17. MSE Loss

介绍：MSE（Mean Squared Error，均方误差）用于回归任务，计算预测值和目标值差的平方平均。它对大误差惩罚更重，因此对异常值比较敏感。MSE 的梯度连续平滑，常用于线性回归和普通回归模型训练。

公式：$L=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$。

- Q：MSE 的公式是什么？
  A：$\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$。
- Q：MSE 为什么对异常值敏感？
  A：误差会被平方，大误差的影响被放大。
- Q：MSE 常用在什么任务？
  A：连续值回归任务，例如房价预测、分数预测、坐标回归等。

```python
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()
```

### 18. MAE Loss

介绍：MAE（Mean Absolute Error，平均绝对误差）用于回归任务，计算预测值和目标值绝对误差的平均。相比 MSE，MAE 对异常值更鲁棒，但在误差为 0 附近不可导，优化时可能不如 MSE 平滑。

公式：$L=\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|$。

- Q：MAE 的公式是什么？
  A：$\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|$。
- Q：MAE 和 MSE 的区别是什么？
  A：MSE 对大误差更敏感，MAE 对异常值更鲁棒。
- Q：MAE 的优化缺点是什么？
  A：绝对值函数在 0 处不可导，梯度不如 MSE 平滑。

```python
def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()
```

### 19. Smooth L1 Loss

介绍：Smooth L1 Loss 用于回归任务，它在误差较小时像 MSE 一样使用平方惩罚，在误差较大时像 MAE 一样使用线性惩罚。设误差 $d=|\hat{y}-y|$，当 $d<\beta$ 时损失为 $\frac{1}{2}d^2/\beta$，否则损失为 $d-\frac{1}{2}\beta$。它常用于目标检测中的边框回归。

公式：$d=|\hat{y}-y|$；$L=\begin{cases}\frac{1}{2}d^2/\beta,&d<\beta\\ d-\frac{1}{2}\beta,&d\ge\beta\end{cases}$。

- Q：Smooth L1 Loss 解决什么问题？
  A：它结合了 MSE 小误差处平滑、MAE 大误差处鲁棒的特点。
- Q：Smooth L1 Loss 和 MSE 的区别是什么？
  A：MSE 对大误差持续平方放大，Smooth L1 在大误差处变成线性惩罚，对异常值更鲁棒。
- Q：`beta` 控制什么？
  A：`beta` 控制从平方惩罚切换到线性惩罚的阈值，误差小于 `beta` 时更像 MSE，大于 `beta` 时更像 MAE。

```python
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    diff = torch.abs(pred - target)
    small_loss = 0.5 * diff ** 2 / beta
    large_loss = diff - 0.5 * beta
    loss = torch.where(diff < beta, small_loss, large_loss)
    return loss.mean()
```

### 20. Binary Cross Entropy

介绍：Binary Cross Entropy（BCE）用于二分类或多标签分类。二分类中每个样本的标签是 0 或 1，模型输入通常是未过 sigmoid 的 logits。手写理解时可以先用 sigmoid 把 logits 变成概率，再按二分类交叉熵公式计算；工程中优先用更稳定的 `BCEWithLogitsLoss`。

公式：$L=-\left[y\log p+(1-y)\log(1-p)\right]$。

- Q：BCE 适合什么任务？
  A：二分类和多标签分类；多标签分类中每个标签都可以独立用 BCE。
- Q：BCE 和多分类 Cross Entropy 有什么区别？
  A：BCE 对每个标签独立做 sigmoid 二分类；多分类 CE 用 softmax，类别之间互斥。
- Q：为什么 BCE 常直接输入 logits？
  A：logits 是模型线性输出，训练时通常交给 BCE 函数内部做 sigmoid；工程实现会把 sigmoid 和 log 合并得更稳定。

```python
def binary_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    probs = torch.sigmoid(logits)
    eps = 1e-12
    loss = -(targets * torch.log(probs + eps) + (1.0 - targets) * torch.log(1.0 - probs + eps))
    return loss.mean()
```

### 21. Cross-Entropy Loss

介绍：交叉熵用于多分类任务。直观理解是先用 softmax 把 logits 变成类别概率，再取真实类别概率的负对数。工程实现通常会把 softmax 和 log 合并得更稳定。

公式：$p_j=\frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}$；$L=-\sum_{j=1}^{K}y_j\log p_j$；$y_c=1,\ y_{j\ne c}=0$；$L=-\log p_c=-z_c+\log\sum_{k=1}^{K}e^{z_k}$。

- Q：Cross Entropy 的直观公式是什么？
  A：先算 $p=\operatorname{softmax}(z)$，再对 one-hot 标签做 $-\sum_j y_j\log p_j$；因为只有真实类别 $c$ 的 $y_c=1$，所以化简为 $-\log p_c$。
- Q：`nn.CrossEntropyLoss` 内部等价于什么？
  A：等价于 `log_softmax` 加 `NLLLoss`，输入应是未归一化 logits。
- Q：target 的 shape 和含义是什么？
  A：多分类中 target 通常是 `(B,)` 的类别索引，而不是 one-hot。

```python
def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    eps = 1e-12
    true_probs = probs[torch.arange(targets.shape[0], device=logits.device), targets]
    return -torch.log(true_probs + eps).mean()
```

### 22. Z Loss

介绍：Z Loss 这里指加在语言模型 LM head logits 上的辅助损失，不是 MoE router 的 z-loss。它惩罚每个 token 位置的 $\log\sum_v e^{z_v}$ 的平方，限制 logits 整体尺度过大，避免 softmax 过尖和训练数值不稳定。实际训练中通常把它乘一个很小系数加到语言模型 Cross Entropy 上。

公式：$L_{\mathrm{Z}}=\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\left(\log\sum_{v=1}^{V}e^{z_{b,t,v}}\right)^2$；$L=L_{\mathrm{CE}}+\lambda L_{\mathrm{Z}}$。

- Q：LM head Z Loss 惩罚的是什么？
  A：惩罚 LM head logits 的 `logsumexp` 平方，也就是限制归一化常数 $\log Z$ 不要过大。
- Q：Z Loss 和 Cross Entropy 是什么关系？
  A：Z Loss 通常是辅助项，乘很小权重后加到语言模型 Cross Entropy 上。
- Q：Z Loss 解决什么问题？
  A：抑制 logits 绝对尺度持续变大，降低 softmax 过尖、数值溢出和训练不稳定风险。

```python
def lm_head_z_loss(logits: Tensor) -> Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    return (log_z ** 2).mean()


def language_model_loss_with_z_loss(logits: Tensor, targets: Tensor, z_loss_weight: float = 1e-4) -> Tensor:
    vocab_size = logits.shape[-1]
    ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    z_loss = lm_head_z_loss(logits)
    return ce_loss + z_loss_weight * z_loss
```

### 23. Label Smoothing Cross Entropy

介绍：Label smoothing 把 one-hot 标签从“目标类概率为 1、其它类为 0”改成更平滑的分布。它能降低模型过度自信，常用于分类、Transformer 和视觉模型训练。

公式：$q_y=1-\epsilon+\epsilon/K$；$q_j=\epsilon/K$；$L=-\sum_j q_j\log p_j$。

- Q：Label smoothing 改了什么？
  A：目标类概率从 `1` 变成 $1-\epsilon$ 附近，非目标类分到少量概率。
- Q：它为什么能缓解过拟合？
  A：它不鼓励模型把训练样本预测到极端置信度，降低过度自信。
- Q：Label smoothing 会影响校准吗？
  A：通常能让输出概率更平滑，但 smoothing 过大会损害分类边界。

```python
def label_smoothing_cross_entropy(logits: Tensor, targets: Tensor, smoothing: float = 0.1) -> Tensor:
    num_classes = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    eps = 1e-12

    target_probs = torch.full_like(probs, smoothing / num_classes)
    target_probs[torch.arange(targets.shape[0], device=logits.device), targets] = (
        1.0 - smoothing + smoothing / num_classes
    )

    loss = -(target_probs * torch.log(probs + eps)).sum(dim=-1)
    return loss.mean()
```

### 24. Focal Loss

介绍：Focal Loss 在交叉熵上乘 $(1-p_t)^\gamma$，降低易分类样本的权重，让训练更关注难样本。它常用于类别不均衡的检测或二分类任务。

公式：$L=-\alpha(1-p_t)^\gamma\log p_t$。

- Q：Focal Loss 解决什么问题？
  A：解决大量简单负样本主导训练的问题，使模型更关注难样本。
- Q：`gamma` 的作用是什么？
  A：`gamma` 越大，易分类样本被降权越明显。
- Q：`alpha` 的作用是什么？
  A：用于平衡正负样本或不同类别的损失权重。

```python
def focal_loss_binary(logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    probs = torch.sigmoid(logits)
    eps = 1e-12
    bce = -(targets * torch.log(probs + eps) + (1.0 - targets) * torch.log(1.0 - probs + eps))
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    return loss.mean()
```

### 25. Load Balance Loss

介绍：Load Balance Loss 是 MoE 模型中常见的 router 辅助损失，用来避免大量 token 被路由到少数 expert，导致 expert 负载不均衡。它通常不会替代主任务 loss，而是乘一个较小系数加到总 loss 中。

公式：$p_{t,e}=\operatorname{softmax}(r_t)_e$；$f_e=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[a_t=e]$；$q_e=\frac{1}{T}\sum_{t=1}^{T}p_{t,e}$；$L_{\mathrm{LB}}=E\sum_{e=1}^{E}f_eq_e$。

- Q：Load Balance Loss 解决什么问题？
  A：防止 router 总是选择少数 expert，缓解 expert collapse 和训练吞吐不均衡。
- Q：$f_e$ 和 $q_e$ 分别表示什么？
  A：$f_e$ 是实际分配到第 $e$ 个 expert 的 token 占比，$q_e$ 是 router 给第 $e$ 个 expert 的平均概率。
- Q：Load Balance Loss 是主损失吗？
  A：不是，通常是辅助损失，需要乘较小权重后加到语言模型 loss 或任务 loss 上。

```python
def load_balance_loss(router_logits: Tensor) -> Tensor:
    num_experts = router_logits.shape[-1]
    probs = torch.softmax(router_logits, dim=-1)
    flat_probs = probs.reshape(-1, num_experts)

    chosen_experts = torch.argmax(flat_probs, dim=-1)
    hard_assignment = F.one_hot(chosen_experts, num_classes=num_experts).float()

    token_fraction = hard_assignment.mean(dim=0)
    router_probability = flat_probs.mean(dim=0)
    return num_experts * torch.sum(token_fraction * router_probability)
```

### 26. Pairwise Loss

介绍：Pairwise Loss 常用于成对偏好学习和 reward model 训练。标准 Pairwise Loss 可以从 Bradley-Terry（BT）模型的负对数似然完整推导出来：BT 模型把两个候选的分数差转成偏好概率，训练时最大化 preferred 样本 $i$ 胜过 dispreferred 样本 $j$ 的概率。带 margin 的 $L_{\mathrm{rank}}$ 是在 BT loss 基础上的排序损失扩展，要求 $o_i$ 不只是大于 $o_j$，而是至少高出 $m_{ij}$。

公式：$P(i\succ j)=\frac{\exp(o_i/\tau)}{\exp(o_i/\tau)+\exp(o_j/\tau)}=\sigma\left(\frac{o_i-o_j}{\tau}\right)$；$L_{\mathrm{BT}}=-\log P(i\succ j)=-\log\sigma\left(\frac{o_i-o_j}{\tau}\right)=\operatorname{softplus}\left(-\frac{o_i-o_j}{\tau}\right)$；$L_{\mathrm{rank}}=\operatorname{softplus}\left(-\frac{o_i-o_j-m_{ij}}{\tau}\right)$。

- Q：Bradley-Terry 模型怎么建模偏好概率？
  A：用两个候选的分数差建模，$P(i\succ j)=\sigma((o_i-o_j)/\tau)$。
- Q：margin $m_{ij}$ 的作用是什么？
  A：它是 BT loss 的排序扩展项，要求 preferred 分数至少比 dispreferred 高出一定间隔；当 $m_{ij}=0$ 时退化为标准 BT loss。
- Q：Pairwise Loss 和 DPO 的区别是什么？
  A：Pairwise Loss 通常训练显式打分模型或 reward model；DPO 直接优化 policy，并用 reference model 约束策略变化。

```python
def pairwise_loss(
    preferred_scores: Tensor,
    dispreferred_scores: Tensor,
    margin: float | Tensor = 0.0,
    temperature: float = 1.0,
) -> Tensor:
    score_gap = preferred_scores - dispreferred_scores - margin
    return F.softplus(-score_gap / temperature).mean()
```

### 27. InfoNCE Loss

介绍：InfoNCE Loss 是对比学习常用损失。给定 query、正样本 key 和一组负样本 key，它把“选中正样本”写成一个 softmax 分类问题。SimCLR、CLIP 等模型都可以看作使用了类似的对比学习目标。

公式：$s_i=\frac{q^Tk_i}{\tau}$；$L=-\log\frac{\exp(s_+)}{\sum_{j=1}^{K}\exp(s_j)}$。

- Q：InfoNCE 中正样本和负样本是什么？
  A：正样本是应该靠近 query 的样本，负样本是不应该靠近 query 的样本，常用同 batch 其它样本作为负样本。
- Q：temperature $\tau$ 的作用是什么？
  A：控制相似度分布的尖锐程度；较小 $\tau$ 会放大相似度差异，使学习信号更强但也可能不稳定。
- Q：InfoNCE 和 Cross Entropy 有什么关系？
  A：它本质上是对相似度 logits 做 Cross Entropy，目标类别是正样本所在位置。

```python
def info_nce_loss(query: Tensor, keys: Tensor, positive_index: Tensor, temperature: float = 0.07) -> Tensor:
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    logits = query @ keys.T / temperature
    return F.cross_entropy(logits, positive_index)
```

### 28. Triplet Loss

介绍：Triplet Loss 常用于度量学习和检索任务。每个训练样本由 anchor、positive、negative 组成，目标是让 anchor 到 positive 的距离小于 anchor 到 negative 的距离，并至少拉开一个 margin。

公式：$L=\max(0,d(a,p)-d(a,n)+m)$。

- Q：Triplet Loss 的 anchor、positive、negative 分别是什么？
  A：anchor 是基准样本，positive 是同类或相似样本，negative 是异类或不相似样本。
- Q：margin $m$ 的作用是什么？
  A：要求 negative 不只是比 positive 远，而是至少远出一个间隔。
- Q：Hard negative mining 是什么？
  A：选择距离 anchor 较近、容易混淆的 negative，让训练信号更强。

```python
def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0) -> Tensor:
    pos_dist = torch.norm(anchor - positive, p=2, dim=-1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=-1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()
```

### 29. KL Divergence

介绍：KL Divergence 衡量两个概率分布的差异，公式是 $D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$。它不对称，常用于知识蒸馏、VAE、分布匹配和策略优化。

公式：$D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$。

- Q：KL 散度是距离吗？
  A：不是严格距离，因为它不对称，且不满足三角不等式。
- Q：$D_{\mathrm{KL}}(P\|Q)$ 中哪个分布作为权重？
  A：由 `P` 加权，计算 `P` 对 `Q` 的相对信息差。
- Q：KL 常见数值稳定写法是什么？
  A：用 log probability 计算，避免直接对很小概率取 log。

```python
def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    p = torch.exp(log_p)
    return (p * (log_p - log_q)).sum(dim=-1).mean()
```

### 30. Knowledge Distillation

介绍：Knowledge Distillation 用 teacher model 的软概率分布训练 student model。相比只用 hard label，teacher 的 soft target 包含类别之间的相似性信息；常把蒸馏 KL loss 和普通监督 CE loss 加权求和。

公式：$q=\operatorname{softmax}(z_T/\tau)$；$p=\operatorname{softmax}(z_S/\tau)$；$L_{\mathrm{KD}}=\tau^2D_{\mathrm{KL}}(q\|p)$。

- Q：Knowledge Distillation 的 teacher 和 student 是什么？
  A：teacher 是较强或较大的模型，student 是要训练的小模型或部署模型。
- Q：temperature 在蒸馏中有什么作用？
  A：较大 temperature 会软化概率分布，让 student 学到更多类别间相似性。
- Q：为什么蒸馏 loss 常乘 $\tau^2$？
  A：temperature 会缩放梯度，乘 $\tau^2$ 用来补偿梯度尺度变化。

```python
def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 2.0) -> Tensor:
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss * (temperature ** 2)
```

## 评估指标

### 31. Precision / Recall

介绍：Precision 和 Recall 是分类任务常用评估指标，尤其适合类别不均衡场景。Precision 衡量“预测为正的样本里有多少是真的正样本”，Recall 衡量“真实正样本里有多少被找出来”。二者常通过 F1 score 做综合衡量。

公式：$P=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$；$R=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$。

- Q：Precision 的公式是什么？
  A：$P=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$，关注预测为正的结果是否准确。
- Q：Recall 的公式是什么？
  A：$R=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$，关注真实正样本是否被尽量找全。
- Q：什么时候更重视 Precision 或 Recall？
  A：误报代价高时更重视 Precision；漏报代价高时更重视 Recall。

```python
def precision_recall_f1(y_true: Tensor, y_pred: Tensor, positive_label: int = 1, eps: float = 1e-12):
    y_true_pos = y_true == positive_label
    y_pred_pos = y_pred == positive_label
    tp = (y_true_pos & y_pred_pos).sum().float()
    fp = (~y_true_pos & y_pred_pos).sum().float()
    fn = (y_true_pos & ~y_pred_pos).sum().float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1
```


### 32. 混淆矩阵

介绍：混淆矩阵用于统计分类模型的预测结果和真实标签之间的对应关系。二分类中常见四个量：TP（真正例）、FP（假正例）、TN（真负例）、FN（假负例）。Precision、Recall、F1、整体正确率等指标都可以从混淆矩阵推导出来。

公式：$C_{ij}=|\{n:y_n=i,\hat{y}_n=j\}|$。

- Q：TP、FP、TN、FN 分别是什么？
  A：TP 是真实为正且预测为正；FP 是真实为负但预测为正；TN 是真实为负且预测为负；FN 是真实为正但预测为负。
- Q：混淆矩阵的行列通常表示什么？
  A：常见约定是行表示真实类别，列表示预测类别；面试中要先说明自己的约定。
- Q：为什么类别不均衡时只看整体正确率不够？
  A：多数类占比很高时，模型即使只预测多数类也可能有高整体正确率，但少数类 Recall 很差。

```python
def confusion_matrix(y_true: Tensor, y_pred: Tensor, num_classes: int) -> Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=y_true.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def binary_confusion_counts(y_true: Tensor, y_pred: Tensor, positive_label: int = 1):
    y_true_pos = y_true == positive_label
    y_pred_pos = y_pred == positive_label
    tp = (y_true_pos & y_pred_pos).sum()
    fp = (~y_true_pos & y_pred_pos).sum()
    tn = (~y_true_pos & ~y_pred_pos).sum()
    fn = (y_true_pos & ~y_pred_pos).sum()
    return tp, fp, tn, fn
```

### 33. F1 Score

介绍：F1 Score 是 Precision 和 Recall 的调和平均数，公式是 $\frac{2PR}{P+R}$。它适合类别不均衡、同时关注误报和漏报的场景。F1 越高，表示模型在查准率和查全率之间取得了更好的平衡。

公式：$F_1=\frac{2PR}{P+R}$。

- Q：F1 为什么用调和平均？
  A：调和平均会惩罚 Precision 和 Recall 中较小的那个，只有二者都高时 F1 才高。
- Q：F1 和整体正确率有什么区别？
  A：整体正确率看全部样本预测正确比例；F1 更关注正类的 Precision 和 Recall，类别不均衡时更有参考价值。
- Q：Macro F1 和 Micro F1 区别是什么？
  A：Macro F1 先按类别算 F1 再平均，各类别权重相同；Micro F1 先汇总所有类别的 TP/FP/FN 再计算，更受样本多的类别影响。

```python
def f1_score_binary(y_true: Tensor, y_pred: Tensor, positive_label: int = 1, eps: float = 1e-12) -> Tensor:
    tp, fp, _, fn = binary_confusion_counts(y_true, y_pred, positive_label)
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)
```

### 34. ROC / AUC

介绍：ROC 曲线描述二分类模型在不同阈值下的 TPR 和 FPR 变化，横轴是 FPR，纵轴是 TPR。AUC 是 ROC 曲线下面积，衡量模型把正样本排在负样本前面的能力。AUC 与具体阈值无关，因此常用于评估排序质量。

公式：$\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$；$\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}$；$\mathrm{AUC}=\int \mathrm{TPR}\,d\mathrm{FPR}$。

- Q：TPR 和 FPR 分别是什么？
  A：$\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$，也就是 Recall；$\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}$。
- Q：AUC 的直观含义是什么？
  A：随机抽一个正样本和一个负样本，模型给正样本更高分的概率。
- Q：ROC-AUC 和 PR-AUC 什么时候更适合？
  A：类别较均衡时 ROC-AUC 常用；正类很稀少时 PR-AUC 往往更能反映模型对正类的识别能力。

```python
def roc_curve_binary(y_true: Tensor, y_score: Tensor, positive_label: int = 1, eps: float = 1e-12):
    y_true = (y_true == positive_label).float()
    order = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_true_sorted, dim=0)
    total_pos = y_true.sum().clamp_min(eps)
    total_neg = (1.0 - y_true).sum().clamp_min(eps)

    tpr = tps / total_pos
    fpr = fps / total_neg
    tpr = torch.cat([torch.zeros(1, device=y_true.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=y_true.device), fpr])
    thresholds = torch.cat([torch.tensor([float("inf")], device=y_score.device), y_score_sorted])
    return fpr, tpr, thresholds


def auc_trapezoid(x: Tensor, y: Tensor) -> Tensor:
    return torch.trapz(y, x)
```

### 35. PR-AUC

介绍：PR-AUC 是 Precision-Recall 曲线下面积，横轴通常是 Recall，纵轴是 Precision。它更关注正类识别质量，在正类很稀少、负类很多的场景中通常比 ROC-AUC 更敏感。

公式：$\mathrm{AUC}_{\mathrm{PR}}=\int_0^1 P(R)\,dR$。

- Q：PR 曲线的横纵轴是什么？
  A：常见画法是横轴 Recall，纵轴 Precision。
- Q：PR-AUC 和 ROC-AUC 哪个更适合极度类别不均衡？
  A：PR-AUC 通常更适合，因为它直接关注正类的查准率和查全率。
- Q：为什么阈值变化能得到 PR 曲线？
  A：不同阈值会改变预测为正的样本集合，从而改变 Precision 和 Recall。

```python
def pr_curve_binary(y_true: Tensor, y_score: Tensor, positive_label: int = 1, eps: float = 1e-12):
    y_true = (y_true == positive_label).float()
    order = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[order]
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_true_sorted, dim=0)
    precision = tps / (tps + fps + eps)
    recall = tps / y_true.sum().clamp_min(eps)
    precision = torch.cat([torch.ones(1, device=y_true.device), precision])
    recall = torch.cat([torch.zeros(1, device=y_true.device), recall])
    return recall, precision


def pr_auc_trapezoid(recall: Tensor, precision: Tensor) -> Tensor:
    return torch.trapz(precision, recall)
```

### 36. Perplexity

介绍：Perplexity 困惑度是语言模型常用指标，等于平均负对数似然的指数 $\exp(L)$。它可以理解为模型在每个位置平均面对多少个“有效选择”，数值越低通常表示模型对真实文本预测越好。

公式：$PPL=\exp(L)$。

- Q：Perplexity 和 Cross Entropy 的关系是什么？
  A：如果 $L$ 是平均 NLL，$PPL=\exp(L)$。
- Q：Perplexity 越低越好吗？
  A：通常越低说明模型给真实 token 的概率越高，但它不完全等价于生成质量。
- Q：比较 PPL 时要注意什么？
  A：要保证 tokenizer、数据集、上下文长度和 loss 计算方式一致。

```python
def perplexity(logits: Tensor, targets: Tensor) -> Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return torch.exp(loss)
```

## 正则化与泛化

### 37. 过拟合

介绍：过拟合指模型在训练集上表现很好，但在验证集或测试集上表现明显变差，本质是模型记住了训练数据中的噪声或偶然模式。常见缓解方法包括增加数据、数据增强、正则化、Dropout、Early Stopping、降低模型复杂度和交叉验证。

- Q：怎么判断模型过拟合？
  A：训练 loss 持续下降，但验证 loss 上升或验证指标变差，通常说明过拟合。
- Q：过拟合和模型容量有什么关系？
  A：模型容量越大越容易拟合训练噪声，但是否过拟合还取决于数据量、正则化和训练策略。
- Q：缓解过拟合有哪些常见方法？
  A：数据增强、L1/L2、Dropout、Early Stopping、降低模型复杂度和交叉验证。

```python
def is_overfitting(train_losses: list[float], val_losses: list[float], window: int = 3, min_gap: float = 0.0) -> bool:
    if len(train_losses) < window + 1 or len(val_losses) < window + 1:
        return False
    train_improving = train_losses[-1] < train_losses[-window - 1]
    val_worse = val_losses[-1] > val_losses[-window - 1]
    generalization_gap = val_losses[-1] - train_losses[-1]
    return train_improving and val_worse and generalization_gap > min_gap
```

### 38. 欠拟合

介绍：欠拟合指模型在训练集和验证集上表现都不好，说明模型没有学到数据中的主要规律。常见原因包括模型太简单、特征不足、训练不充分、正则化过强或学习率设置不合理。

- Q：欠拟合的典型现象是什么？
  A：训练 loss 和验证 loss 都较高，训练集指标也不理想。
- Q：怎么缓解欠拟合？
  A：增大模型容量、增加有效特征、训练更久、减弱正则化或调整学习率。
- Q：欠拟合和过拟合能同时出现吗？
  A：整体模型可能欠拟合，局部类别或小数据子集也可能过拟合；要结合曲线和分组指标判断。

```python
def is_underfitting(train_loss: float, val_loss: float, target_loss: float, gap_threshold: float = 0.1) -> bool:
    high_train_loss = train_loss > target_loss
    small_gap = abs(val_loss - train_loss) <= gap_threshold
    return high_train_loss and small_gap
```

### 39. Data Augmentation

介绍：Data Augmentation 通过对训练样本做保持语义或标签不变的变换来扩大有效数据分布，常用于缓解过拟合。图像中常见随机裁剪、翻转、颜色扰动；文本中可以做轻量扰动或回译；验证集和测试集通常不做随机增强。

公式：$(\tilde{x},\tilde{y})=a(x,y)$；$a\sim\mathcal{A}$。

- Q：Data Augmentation 为什么能缓解过拟合？
  A：它让模型看到同一语义下的更多输入变化，降低对训练样本细节和噪声的记忆。
- Q：验证集和测试集要做随机增强吗？
  A：通常不做随机增强，只做确定性的预处理；否则评估结果会引入额外随机性。
- Q：增强越强越好吗？
  A：不是。过强增强可能改变标签语义或制造分布偏移，导致训练目标变噪。

```python
def random_horizontal_flip(images: Tensor, p: float = 0.5) -> Tensor:
    flip_mask = torch.rand(images.shape[0], device=images.device) < p
    images = images.clone()
    images[flip_mask] = torch.flip(images[flip_mask], dims=[-1])
    return images
```

### 40. Bias-Variance Tradeoff

介绍：Bias-Variance Tradeoff 描述模型误差来源的权衡。高 bias 表示模型假设太强、容易欠拟合；高 variance 表示模型对训练数据扰动敏感、容易过拟合。好的模型需要在二者之间取得平衡。

公式：$\mathbb{E}[(\hat{y}(x)-y)^2]=\mathrm{Bias}^2+\mathrm{Var}+\sigma^2$。

- Q：高 bias 通常对应什么现象？
  A：训练集和验证集都表现差，模型太简单或训练不足。
- Q：高 variance 通常对应什么现象？
  A：训练集表现好，验证集表现差，对数据扰动敏感。
- Q：增加数据主要缓解 bias 还是 variance？
  A：通常更能缓解 variance，降低模型对特定训练样本的敏感性。

```python
def bias_variance(y_true: Tensor, predictions: Tensor) -> tuple[Tensor, Tensor]:
    mean_pred = predictions.mean(dim=0)
    bias2 = ((mean_pred - y_true) ** 2).mean()
    variance = ((predictions - mean_pred) ** 2).mean()
    return bias2, variance
```

### 41. L1 正则化

介绍：L1 正则在损失中加入参数绝对值之和，形式是 $\lambda\|\theta\|_1$。它倾向于把部分权重压到 0，因此常用于特征选择和稀疏模型。

公式：$L_{\mathrm{total}}=L+\lambda\|\theta\|_1$。

- Q：L1 正则为什么能产生稀疏性？
  A：绝对值惩罚在 0 附近有固定强度的收缩效果，容易把小权重推到 0。
- Q：L1 和 L2 的主要区别是什么？
  A：L1 倾向稀疏，L2 倾向让权重整体变小但通常不变成精确 0。
- Q：L1 正则常用在哪里？
  A：特征选择、稀疏线性模型和需要压缩无用权重的场景。

```python
def l1_regularization(parameters, lambda_l1: float = 1e-4) -> Tensor:
    params = list(parameters)
    penalty = torch.zeros((), device=params[0].device if params else None)
    for p in params:
        penalty = penalty + p.abs().sum()
    return lambda_l1 * penalty
```

### 42. L2 正则化

介绍：L2 正则在损失中加入权重平方和，形式是 $\lambda\|\theta\|_2^2$。它会惩罚过大的权重，使模型函数更平滑，常用于缓解过拟合。

公式：$L_{\mathrm{total}}=L+\lambda\|\theta\|_2^2$；$\frac{\partial}{\partial\theta}\lambda\theta^2=2\lambda\theta$。

- Q：L2 正则的梯度是什么？
  A：对 $\lambda\theta^2$ 求导得到 $2\lambda\theta$，会把权重往 0 拉。
- Q：L2 正则为什么能缓解过拟合？
  A：它限制权重过大，降低模型对训练噪声的敏感性。
- Q：L2 和 Weight Decay 完全一样吗？
  A：在普通 SGD 中常等价；在 Adam 这类自适应优化器中，解耦 weight decay 更常用。

```python
def l2_regularization(parameters, lambda_l2: float = 1e-4) -> Tensor:
    params = list(parameters)
    penalty = torch.zeros((), device=params[0].device)
    for p in params:
        penalty = penalty + p.pow(2).sum()
    return lambda_l2 * penalty
```

### 43. Weight Decay

介绍：Weight decay 直接在参数更新时衰减权重，例如 $\theta\leftarrow(1-\eta\lambda)\theta$。在 AdamW 中，weight decay 与梯度自适应更新解耦，因此比把 L2 正则混进 Adam 梯度更符合预期。

公式：$\theta\leftarrow(1-\eta\lambda)\theta$。

- Q：Weight decay 的直观作用是什么？
  A：每次更新都把权重向 0 缩小一点，限制参数过大。
- Q：AdamW 为什么叫 decoupled weight decay？
  A：它把权重衰减独立于 Adam 的一阶、二阶矩估计之外执行。
- Q：bias 和 norm 参数通常做 weight decay 吗？
  A：通常不做。bias 和 norm 的 scale/shift 参数量少，主要负责平移和尺度校准，衰减它们容易破坏归一化层的校准效果，正则收益也很小。

```python
def apply_weight_decay(param: Tensor, lr: float, weight_decay: float) -> Tensor:
    return param * (1.0 - lr * weight_decay)
```

### 44. LoRA

介绍：LoRA 冻结原始线性层，只训练低秩增量 `BA`。前向为 $W_0x+\frac{\alpha}{r}B A x$，可显著减少微调参数和优化器状态。

公式：$y=W_0x+\frac{\alpha}{r}BAx$。

- Q：LoRA 训练哪些参数？
  A：冻结原线性层，只训练低秩矩阵 `A` 和 `B`。
- Q：为什么 `B` 常初始化为 0？
  A：让初始 LoRA 分支输出为 0，模型初始行为等于原始基座模型。
- Q：LoRA 的 rank 有什么影响？
  A：rank 越大表达能力越强，但训练参数、显存和计算也越多。

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        for p in self.linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### 45. Early Stopping

介绍：Early Stopping 在验证集长期不提升时提前停止训练，避免模型继续拟合训练集噪声。它通常配合 patience 和 min_delta 使用，并保存验证集最优 checkpoint。

- Q：Early Stopping 监控训练集还是验证集？
  A：通常监控验证集 loss 或验证指标，因为目标是提升泛化能力。
- Q：patience 是什么？
  A：允许验证指标连续多少轮不提升，超过后停止训练。
- Q：Early Stopping 是正则化吗？
  A：可以看作训练过程层面的正则化，限制模型继续拟合训练噪声。

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
```

### 46. EMA

介绍：EMA（Exponential Moving Average）通常指对模型参数维护一份指数滑动平均的影子权重。训练时正常更新原模型参数，同时用当前参数更新 EMA 参数；验证或推理时可以临时使用 EMA 权重，常见作用是让评估结果更平滑、更稳定。

公式：$\theta_t^{\mathrm{EMA}}=\alpha\theta_{t-1}^{\mathrm{EMA}}+(1-\alpha)\theta_t$。

- Q：EMA 和优化器里的 momentum 是一回事吗？
  A：不是。优化器 momentum 平滑梯度或更新方向；模型 EMA 平滑的是模型参数本身。
- Q：EMA 通常什么时候更新？
  A：通常在每次 `optimizer.step()` 之后，用更新后的模型参数刷新 EMA 权重。
- Q：训练和推理分别用哪份权重？
  A：训练继续用原始权重反向传播；验证或推理时可以复制 EMA 权重到模型中评估。

```python
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                old_average = self.shadow[name]
                new_value = param.detach()
                old_average.mul_(self.decay).add_(new_value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])
```

## 初始化与训练稳定性

### 47. Fan-in Initialization

介绍：Fan-in 初始化的核心思想是根据每个输出单元接收的输入数量 `fan_in` 控制权重尺度，避免输入维度变大时输出方差过大。线性层中 $n_{\mathrm{in}}=d_{\mathrm{in}}$；卷积层中 $n_{\mathrm{in}}=C_{\mathrm{in}}k_hk_w$。很多初始化方法如 LeCun、Kaiming 都是 fan-in 思路的具体版本。

公式：$n_{\mathrm{in}}=d_{\mathrm{in}}$；$n_{\mathrm{in}}=C_{\mathrm{in}}k_hk_w$；$a=\frac{1}{\sqrt{n_{\mathrm{in}}}}$；$W\sim\mathcal{U}(-a,a)$。

- Q：`fan_in` 是什么？
  A：一个神经元接收的输入连接数；线性层是输入维度，卷积层是 $C_{\mathrm{in}}\times k_h\times k_w$。
- Q：为什么初始化要考虑 `fan_in`？
  A：输入连接越多，未缩放的加权和方差越大；按 `fan_in` 缩放能稳定前向激活尺度。
- Q：`fan_in` 和 `fan_out` 有什么区别？
  A：`fan_in` 关注前向传播输入数量，`fan_out` 关注反向传播梯度分散到多少输出连接。

```python
def calculate_fan_in_and_fan_out(weight: Tensor) -> tuple[int, int]:
    if weight.dim() < 2:
        fan_in = fan_out = weight.numel()
    elif weight.dim() == 2:
        fan_out, fan_in = weight.shape
    else:
        receptive_field_size = math.prod(weight.shape[2:])
        fan_in = weight.shape[1] * receptive_field_size
        fan_out = weight.shape[0] * receptive_field_size
    return fan_in, fan_out


def fan_in_uniform_init(weight: Tensor) -> Tensor:
    fan_in, _ = calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        weight.uniform_(-bound, bound)
    return weight
```

### 48. Xavier Initialization

介绍：Xavier 初始化也叫 Glorot 初始化，目标是同时稳定前向激活和反向梯度，常用于 tanh/sigmoid 等相对对称的激活函数。它同时考虑 `fan_in` 和 `fan_out`，常见正态版本标准差是 $\sqrt{\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}}$，均匀版本范围是 $[-\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})},\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})}]$。

公式：$\sigma=\sqrt{\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}}$；$a=\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}$。

- Q：Xavier 初始化适合什么激活？
  A：适合 tanh/sigmoid 这类较对称、不过度截断负半轴的激活。
- Q：Xavier 为什么同时看 `fan_in` 和 `fan_out`？
  A：它希望前向激活方差和反向梯度方差都尽量稳定。
- Q：Xavier 和 Kaiming 的核心区别是什么？
  A：Xavier 用 $n_{\mathrm{in}}+n_{\mathrm{out}}$，常配 tanh/sigmoid；Kaiming 主要按 `fan_in`，常配 ReLU。

```python
def xavier_uniform_init(weight: Tensor) -> Tensor:
    fan_in, fan_out = calculate_fan_in_and_fan_out(weight)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-bound, bound)
    return weight


def xavier_normal_init(weight: Tensor) -> Tensor:
    fan_in, fan_out = calculate_fan_in_and_fan_out(weight)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.normal_(0.0, std)
    return weight
```

### 49. LeCun Initialization

介绍：LeCun 初始化常用于 SELU、tanh 等网络，核心尺度是 $1/n_{\mathrm{in}}$，正态版本标准差为 $\sqrt{1/n_{\mathrm{in}}}$。它强调保持前向传播的激活方差稳定，是比 Xavier/Kaiming 更早的一类经典初始化方法。

公式：$\sigma=\sqrt{1/n_{\mathrm{in}}}$。

- Q：LeCun normal 的标准差是多少？
  A：$\sigma=\sqrt{1/n_{\mathrm{in}}}$。
- Q：LeCun 初始化常和什么激活搭配？
  A：常和 SELU、自归一化网络或 tanh 类网络一起讨论。
- Q：LeCun 和 Kaiming 的区别是什么？
  A：LeCun 使用 $1/n_{\mathrm{in}}$，Kaiming ReLU 版本使用 $2/n_{\mathrm{in}}$，因为 ReLU 会丢掉约一半负半轴信号。

```python
def lecun_normal_init(weight: Tensor) -> Tensor:
    fan_in, _ = calculate_fan_in_and_fan_out(weight)
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0.0, std)
    return weight


def lecun_uniform_init(weight: Tensor) -> Tensor:
    fan_in, _ = calculate_fan_in_and_fan_out(weight)
    bound = math.sqrt(3.0 / fan_in)
    with torch.no_grad():
        weight.uniform_(-bound, bound)
    return weight
```

### 50. Kaiming Initialization

介绍：Kaiming 初始化用于 ReLU 类网络，令权重标准差为 $\sqrt{2/n_{\mathrm{in}}}$，帮助前向和反向的方差在层间保持稳定。

公式：$\sigma=\sqrt{2/n_{\mathrm{in}}}$。

- Q：Kaiming 初始化适合什么激活？
  A：主要适合 ReLU 或 LeakyReLU 这类会截断一部分信号的激活。
- Q：`fan_in` 是什么？
  A：一个输出单元接收的输入数量；线性层通常是 `weight.shape[1]`，卷积层是 $C_{\mathrm{in}}k_hk_w$。
- Q：Kaiming 和 Xavier 的区别是什么？
  A：Kaiming 使用 $2/n_{\mathrm{in}}$，更适合 ReLU；Xavier 通常考虑 `fan_in` 和 `fan_out`，适合 tanh/sigmoid。

```python
def kaiming_init(weight: Tensor) -> Tensor:
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0.0, std)
    return weight
```

### 51. 梯度消失

介绍：梯度消失指反向传播时梯度在层间逐渐变小，靠近输入端的参数几乎得不到有效更新。常见原因包括网络很深、Sigmoid/Tanh 饱和、初始化过小、RNN 长序列反向传播等。常用缓解方法包括 ReLU、SiLU、残差连接、归一化层、合适初始化和门控结构。

公式：$\left\|\frac{\partial L}{\partial h_l}\right\|\to0$。

- Q：梯度消失最常见的表现是什么？
  A：训练 loss 下降很慢，前面层的梯度 norm 很小，参数几乎不更新。
- Q：为什么 Sigmoid/Tanh 容易导致梯度消失？
  A：它们在饱和区导数接近 0，多层链式相乘后梯度会快速衰减。
- Q：残差连接为什么能缓解梯度消失？
  A：残差提供更短的梯度路径，让梯度可以绕过部分非线性层直接传播。

```python
def min_grad_norm(parameters) -> float:
    norms = [p.grad.detach().norm().item() for p in parameters if p.grad is not None]
    return min(norms) if norms else 0.0
```

### 52. 梯度爆炸

介绍：梯度爆炸指反向传播时梯度变得非常大，导致参数更新过猛，loss 剧烈震荡甚至出现 NaN/Inf。常见于深层网络、RNN 长序列、不合适初始化或学习率过大。常用处理方法包括 gradient clipping、降低学习率、归一化层和更稳的初始化。

公式：$\left\|\frac{\partial L}{\partial h_l}\right\|\to\infty$。

- Q：梯度爆炸怎么发现？
  A：loss 突然变得很大或 NaN，参数/梯度 norm 快速增大。
- Q：最直接的处理方法是什么？
  A：gradient clipping，限制总梯度范数或逐元素梯度范围。
- Q：梯度爆炸和学习率有什么关系？
  A：学习率过大会放大梯度更新的影响，使训练更容易震荡或发散。

```python
def total_grad_norm(parameters, eps: float = 1e-12) -> float:
    grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum(g * g for g in grads) + eps)
    return total.item()


def has_bad_gradients(parameters) -> bool:
    for p in parameters:
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            return True
    return False
```

### 53. Gradient Clipping

介绍：Gradient clipping 用于控制梯度爆炸。按总 L2 norm 裁剪时，所有梯度乘同一个系数，因此整体方向不变，只限制范数不超过 `max_norm`。

公式：$g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2}\right)$。

- Q：Gradient clipping 解决什么问题？
  A：主要用于缓解梯度爆炸，防止一次更新步长过大。
- Q：clip by norm 和 clip by value 有什么区别？
  A：clip by norm 按整体范数缩放，保留方向；clip by value 逐元素截断，可能改变方向。
- Q：为什么返回裁剪前的 norm？
  A：便于日志监控和判断是否频繁发生梯度爆炸。

```python
def clip_grad_norm(parameters, max_norm: float) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
    return total_norm.item()
```

## 优化器与训练调度

### 54. SGD Optimizer

介绍：SGD（Stochastic Gradient Descent，随机梯度下降）是最基础的优化器。每次用当前 mini-batch 的梯度更新参数：$\theta_{t+1}=\theta_t-\eta g_t$。它实现简单、泛化能力常不错，但对学习率敏感，且在峡谷形损失面上可能震荡。

公式：$\theta_{t+1}=\theta_t-\eta g_t$。

- Q：SGD 的更新公式是什么？
  A：$\theta_{t+1}=\theta_t-\eta g_t$，其中 $\eta$ 是学习率，$g_t$ 是当前梯度。
- Q：SGD 为什么叫 stochastic？
  A：因为通常用随机采样的 mini-batch 估计全量数据梯度，而不是每步计算整个训练集梯度。
- Q：SGD 的主要缺点是什么？
  A：收敛可能慢，对学习率敏感，在高曲率方向容易震荡。

```python
class MySGD:
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### 55. SGD with Momentum

介绍：Momentum 在 SGD 的基础上加入速度项，累积历史梯度方向：$v_t=\mu v_{t-1}+g_t$，再用 `v` 更新参数。它能减少来回震荡，并在一致下降方向上加速收敛。

公式：$v_t=\mu v_{t-1}+g_t$；$\theta_{t+1}=\theta_t-\eta v_t$。

- Q：Momentum 解决了 SGD 的什么问题？
  A：减少震荡、加速稳定方向上的下降，尤其适合狭长峡谷形损失面。
- Q：Momentum 的更新公式是什么？
  A：$v_t=\mu v_{t-1}+g_t$，$\theta_{t+1}=\theta_t-\eta v_t$。
- Q：Momentum 系数常取多少？
  A：常见取值是 `0.9`，也可以根据任务调整。

```python
class MySGDMomentum:
    def __init__(self, params, lr=1e-2, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                p -= self.lr * self.velocity[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### 56. AdaGrad Optimizer

介绍：AdaGrad 是自适应学习率优化器，会累积每个参数历史梯度平方：$G_t=G_{t-1}+g_t^2$，更新时除以 $\sqrt{G_t}+\epsilon$。频繁出现大梯度的参数学习率会变小，稀疏特征场景中很有用，但训练后期学习率可能衰减过度。

公式：$G_t=G_{t-1}+g_t^2$；$\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{G_t}+\epsilon}$。

- Q：AdaGrad 为什么适合稀疏特征？
  A：每个参数有自己的累计梯度平方，少见特征对应参数不会被过快降低学习率。
- Q：AdaGrad 的主要缺点是什么？
  A：累计梯度平方单调增加，后期有效学习率可能变得太小。
- Q：AdaGrad 和 SGD 的核心区别是什么？
  A：SGD 所有参数共享学习率；AdaGrad 为每个参数自适应缩放学习率。

```python
class MyAdaGrad:
    def __init__(self, params, lr=1e-2, eps=1e-10):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.sum_sq_grad = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.sum_sq_grad[i] += p.grad * p.grad
                p -= self.lr * p.grad / (torch.sqrt(self.sum_sq_grad[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### 57. RMSProp Optimizer

介绍：RMSProp 改进了 AdaGrad 学习率持续变小的问题。它不累积所有历史梯度平方，而是使用指数滑动平均：$v_t=\alpha v_{t-1}+(1-\alpha)g_t^2$，再用 $\sqrt{v_t}$ 缩放梯度。

公式：$v_t=\alpha v_{t-1}+(1-\alpha)g_t^2$；$\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}$。

- Q：RMSProp 相比 AdaGrad 改进在哪里？
  A：RMSProp 使用梯度平方的滑动平均，不会让分母无限单调增大。
- Q：RMSProp 中 `alpha` 表示什么？
  A：梯度平方滑动平均的衰减系数，常见取值是 `0.99`。
- Q：RMSProp 和 Adam 的关系是什么？
  A：Adam 可以看作 Momentum 和 RMSProp 思路的结合，同时维护一阶动量和二阶动量。

```python
class MyRMSProp:
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * (p.grad * p.grad)
                p -= self.lr * p.grad / (torch.sqrt(self.square_avg[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```


### 58. Adam Optimizer

介绍：Adam 同时维护一阶动量 `m` 和二阶动量 `v`，并做 bias correction。更新形式是 $\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$。

公式：$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$；$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$；$\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$。

- Q：Adam 的 `m` 和 `v` 分别是什么？
  A：`m` 是梯度的一阶动量，`v` 是梯度平方的二阶动量。
- Q：为什么 Adam 需要 bias correction？
  A：初始 $m_t$ 和 $v_t$ 为 0，早期估计偏小，需要除以 $1-\beta^t$ 修正。
- Q：Adam 和 AdamW 的关键区别是什么？
  A：AdamW 把 weight decay 从梯度更新中解耦，通常更适合 Transformer 训练。

```python
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```


### 59. AdamW Optimizer

介绍：AdamW 是 Adam 的常用改进，核心是 decoupled weight decay。Adam 里如果把 L2 正则直接加进梯度，会和自适应学习率耦合；AdamW 在参数更新时直接加入权重衰减项，但不把权重衰减混进一阶、二阶矩估计。Transformer 和大模型训练中 AdamW 很常用。

公式：$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$；$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$；$\hat{m}_t=\frac{m_t}{1-\beta_1^t}$；$\hat{v}_t=\frac{v_t}{1-\beta_2^t}$；$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\theta_t\right)$。

- Q：AdamW 和 Adam 的核心区别是什么？
  A：AdamW 把 weight decay 从梯度更新中解耦，Adam 通常会把 L2 正则混入梯度。
- Q：为什么 AdamW 更适合 Transformer？
  A：解耦 weight decay 后正则强度更稳定，不会被 Adam 的自适应缩放扭曲。
- Q：AdamW 中哪些参数通常不做 weight decay？
  A：bias、LayerNorm/RMSNorm 的 scale 参数通常不做 weight decay。

```python
class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                adam_update = m_hat / (torch.sqrt(v_hat) + self.eps)
                weight_decay_update = self.weight_decay * p
                p -= self.lr * (adam_update + weight_decay_update)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```


### 60. Muon Optimizer

介绍：Muon 是一种面向神经网络隐藏层二维权重矩阵的优化器，名字通常解释为 MomentUm Orthogonalized by Newton-Schulz。它先维护 SGD momentum，再把 momentum update 通过 Newton-Schulz 迭代近似正交化，最后用正交化后的方向更新参数。Muon 通常用于 hidden weight matrix；embedding、输出层、bias、Norm 参数等一维或特殊参数一般继续用 AdamW。

公式：$G_t=\mu G_{t-1}+\nabla_\theta L$；$U=\operatorname{Polar}(G_t)$；$\theta_{t+1}=\theta_t-\eta U$。

- Q：Muon 的核心思想是什么？
  A：先做 momentum，再对二维权重的 update 做正交化，让更新方向更接近谱范数受控的矩阵方向。
- Q：Muon 适合优化哪些参数？
  A：主要适合隐藏层里的二维权重矩阵，例如 MLP/Attention 的 Linear weight；bias、Norm 参数、embedding、输出 head 通常不用 Muon。
- Q：Newton-Schulz 在 Muon 里做什么？
  A：用少量矩阵乘法近似计算正交化方向，避免直接做昂贵的 SVD。

```python
def zeropower_via_newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    # Approximate the orthogonal factor of G. Used by Muon for 2-D weight updates.
    assert G.dim() == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype=G.dtype)


class MyMuon:
    def __init__(self, params, lr=2e-2, momentum=0.95, ns_steps: int = 5, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("Muon is intended for 2-D weight matrices; use AdamW for bias/norm/embedding.")
                if self.weight_decay != 0:
                    p.mul_(1.0 - self.lr * self.weight_decay)
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                update = zeropower_via_newton_schulz(self.velocity[i], steps=self.ns_steps)
                # Scale keeps update magnitude roughly comparable across rectangular matrices.
                update = update * max(1.0, math.sqrt(p.size(0) / p.size(1)))
                p -= self.lr * update

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### 61. Learning Rate Warmup

介绍：Warmup 在训练初期把学习率从较小值逐步升到目标学习率，避免模型刚开始参数和梯度不稳定时更新过猛。Transformer 和大模型训练中常把 warmup 与 cosine decay 结合。

公式：$\eta_t=\eta_{\max}\frac{t}{T_w}\ (t\le T_w)$。

- Q：Warmup 解决什么问题？
  A：降低训练初期大步更新导致 loss 发散或不稳定的风险。
- Q：Warmup 后学习率通常怎么变化？
  A：常见做法是进入 cosine decay、linear decay 或保持一段时间。
- Q：warmup steps 太大会怎样？
  A：有效大学习率训练时间变短，可能收敛变慢或欠拟合。

```python
def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (base_lr - min_lr) * cosine
```

### 62. Cosine LR Scheduler

介绍：Cosine learning rate schedule 常与 warmup 搭配。训练初期线性升到最大学习率，之后按余弦曲线平滑下降到最小学习率。

公式：$\eta_t=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$。

- Q：为什么需要 warmup？
  A：训练初期参数和优化器状态不稳定，warmup 能避免学习率过大导致发散。
- Q：Cosine decay 的 `progress` 表示什么？
  A：表示 warmup 结束后当前 step 在剩余训练过程中的相对进度。
- Q：$t\ge T$ 时学习率应是多少？
  A：通常固定为 `min_lr`，避免继续下降或出现越界计算。

```python
def cosine_lr_schedule(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

## 建模流程与调参

### 63. 模型选择

介绍：模型选择是根据任务类型、数据规模、特征形态、可解释性、推理成本和验证集表现来选择合适模型。常见思路是先建立简单可靠的 baseline，再逐步尝试更复杂模型；不要只看训练集效果，最终应以验证集或交叉验证表现为主要依据。

公式：$M^*=\arg\min_{M\in\mathcal{M}}\hat{R}_{\mathrm{val}}(M)$。

- Q：怎么选择合适的模型？
  A：先看任务是分类、回归、排序还是生成，再结合数据量、特征类型、非线性强弱、解释性和部署成本选择候选模型。
- Q：为什么要先做 baseline？
  A：baseline 能提供最低可接受效果和排错参照，避免一开始就用复杂模型掩盖数据或评估问题。
- Q：为什么不能只选训练集 loss 最低的模型？
  A：训练集 loss 低可能只是过拟合；模型选择更关注验证集或交叉验证上的泛化表现。

```python
def select_best_model(results: list[dict], metric: str, higher_is_better: bool = True) -> dict:
    if not results:
        raise ValueError("results must not be empty")
    return max(results, key=lambda item: item[metric]) if higher_is_better else min(results, key=lambda item: item[metric])
```

### 64. 交叉验证

介绍：交叉验证把训练数据分成 K 份，每次用其中一份做验证，其余 K-1 份做训练，最后平均 K 次验证结果。它比单次训练/验证划分更稳定，适合数据量不大或模型选择、调参时估计泛化表现。

公式：$\hat{R}_{\mathrm{CV}}=\frac{1}{K}\sum_{k=1}^{K}R_k$。

- Q：交叉验证解决什么问题？
  A：减少单次划分带来的偶然性，让泛化评估更稳定。
- Q：什么时候用 K-fold，什么时候用留出验证集？
  A：数据较少时 K-fold 更稳；数据很多或训练成本很高时，留出验证集更省时间。
- Q：交叉验证中预处理要注意什么？
  A：标准化、特征选择等预处理必须在每个 fold 的训练部分 fit，再用于该 fold 的验证部分，避免数据泄漏。

```python
import numpy as np


def k_fold_indices(n_samples: int, n_splits: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_splits)
    splits = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, val_idx))
    return splits
```

### 65. 超参数调优

介绍：超参数调优是在训练前或训练外选择学习率、正则强度、树深、KNN 的 k、batch size 等不能直接由梯度学习到的配置。常见方法包括手动调参、网格搜索、随机搜索、贝叶斯优化和早停；调参目标应由验证集或交叉验证指标决定。

公式：$\lambda^*=\arg\min_{\lambda\in\Lambda}\hat{R}_{\mathrm{CV}}(\lambda)$。

- Q：调参通常先调哪些参数？
  A：优先调对结果最敏感的参数，例如学习率、正则强度、模型容量、树深、邻居数或 batch size。
- Q：网格搜索和随机搜索有什么区别？
  A：网格搜索穷举预设组合；随机搜索从空间中采样，维度较多时通常更高效。
- Q：为什么调参不能看测试集？
  A：测试集只用于最终一次泛化评估；反复用测试集调参会把测试集也变成训练信号，导致评估偏乐观。

```python
from itertools import product


def grid_search(param_grid: dict, train_eval_fn, higher_is_better: bool = False) -> tuple[dict, float]:
    keys = list(param_grid.keys())
    best_params = None
    best_value = None
    for values in product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        value = train_eval_fn(params)
        if best_value is None:
            best_params, best_value = params, value
        elif higher_is_better and value > best_value:
            best_params, best_value = params, value
        elif not higher_is_better and value < best_value:
            best_params, best_value = params, value
    return best_params, best_value
```

### 66. 学习率选择

介绍：学习率决定每次参数更新步长，是最重要的超参数之一。学习率太大容易 loss 震荡、发散或出现 NaN；学习率太小会收敛很慢、长时间欠拟合。常见做法是先用小范围实验或 learning rate finder 找量级，再结合 warmup、cosine decay 等调度策略训练。

公式：$\theta_{t+1}=\theta_t-\eta g_t$。

- Q：学习率太大通常有什么现象？
  A：loss 大幅震荡、验证指标不稳定，严重时梯度爆炸或 loss 变成 NaN。
- Q：学习率太小通常有什么现象？
  A：训练 loss 下降很慢，模型长期欠拟合，计算资源利用效率低。
- Q：怎么选择初始学习率？
  A：先用经验值或小规模搜索确定量级，再看训练曲线；大模型常配合 warmup，稳定后进入 decay。

```python
def suggest_lr_action(losses: list[float]) -> str:
    if len(losses) < 3:
        return "collect_more_steps"
    if not math.isfinite(losses[-1]) or losses[-1] > 2.0 * losses[0]:
        return "decrease_lr"
    if losses[-1] > 0.98 * losses[0]:
        return "increase_lr"
    return "keep_or_fine_tune"
```

### 67. Batch Size 选择

介绍：Batch size 影响显存占用、梯度噪声、吞吐和泛化。较大 batch 通常吞吐更高、梯度更稳定，但显存占用更大，可能需要更大学习率和 warmup；较小 batch 梯度噪声更大，有时泛化更好，但训练吞吐可能较低。

公式：$B_{\mathrm{eff}}=B_{\mathrm{device}}\times N_{\mathrm{device}}\times K_{\mathrm{acc}}$。

- Q：effective batch size 怎么算？
  A：单卡 micro-batch 乘设备数，再乘梯度累积步数。
- Q：batch size 变大时学习率怎么调？
  A：常见经验是适度增大学习率，并使用 warmup；是否线性放大要看模型、优化器和稳定性。
- Q：batch size 受哪些因素限制？
  A：主要受显存、序列长度、模型大小、激活保存、精度和梯度累积策略限制。

```python
def effective_batch_size(micro_batch_size: int, num_devices: int = 1, grad_accum_steps: int = 1) -> int:
    return micro_batch_size * num_devices * grad_accum_steps
```

## 传统机器学习

### 68. 标准化

介绍：标准化把特征变成均值约 0、方差约 1，常用于 KNN、SVM、Logistic Regression、线性模型和基于梯度的模型。它不限制特征取值范围，而是按均值和标准差重新缩放特征。

公式：$z=\frac{x-\mu}{\sigma}$。

- Q：标准化的目标是什么？
  A：让不同特征处在相近尺度上，避免大尺度特征主导距离、内积或梯度更新。
- Q：标准化和归一化有什么区别？
  A：标准化使用均值和标准差，结果不固定在某个区间；归一化常缩放到固定区间。
- Q：什么时候要只用训练集统计量？
  A：训练、验证和测试都应使用训练集上估计的均值和标准差，避免数据泄漏。

```python
import numpy as np


class StandardScalerNumpy:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### 69. 归一化

介绍：归一化通常指把特征缩放到固定区间，例如 $[0,1]$。它保留样本之间的相对大小关系，但对极端值敏感，因为最小值和最大值会直接决定缩放范围。

公式：$x^{\prime}=\frac{x-x_{\min}}{x_{\max}-x_{\min}}$。

- Q：归一化常用在哪里？
  A：常用于距离模型、需要固定输入范围的模型，或者图像像素等天然有上下界的数据。
- Q：归一化和标准化有什么区别？
  A：归一化通常缩放到固定范围；标准化变成均值约 0、方差约 1。
- Q：归一化为什么对异常值敏感？
  A：极端最小值或最大值会拉大分母，让大部分正常样本被压到很窄的区间。

```python
import numpy as np


def min_max_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    return (X - x_min) / (x_max - x_min + 1e-12)
```

### 70. One-hot Encoding

介绍：One-hot encoding 把离散类别映射成稀疏二进制向量，每个类别对应一个维度。它适合无序类别特征，但类别很多时会导致维度很高，也不能表达类别之间的语义相似性。

公式：$o_{i,c}=\mathbf{1}_{y_i=c}$。

- Q：One-hot 适合什么特征？
  A：适合无序离散类别，例如颜色、城市、类别 ID 等。
- Q：One-hot 的主要问题是什么？
  A：类别很多时维度很高且稀疏，模型也无法直接知道类别之间是否相似。
- Q：One-hot 和 Embedding 有什么关系？
  A：Embedding 可以看作 one-hot 向量乘一个可学习矩阵，但直接查表更高效。

```python
import numpy as np


def one_hot_encode(labels, num_classes=None):
    labels = np.asarray(labels, dtype=np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out
```

### 71. 多元线性回归

介绍：多元线性回归用于多个特征共同预测一个连续目标，模型形式是 $\hat y=Xw+b$，其中 `X` 的形状是 `(N, D)`，`w` 的形状是 `(D,)`，`b` 是标量。常见实现包括闭式解和梯度下降。闭式解适合中小规模数据，梯度下降更适合大数据或后续扩展到神经网络。

公式：$\hat{y}=Xw+b$；$\theta=(X^TX)^{-1}X^Ty$。

- Q：多元线性回归和一元线性回归的区别是什么？
  A：一元线性回归只有一个输入特征；多元线性回归有多个输入特征，本质仍是线性模型 $Xw+b$。
- Q：闭式解为什么要给 X 拼接一列 1？
  A：拼接一列 1 后，可以把 bias 合并进参数向量 $\theta=[w,b]$，统一写成 $\tilde{X}\theta$。
- Q：多重共线性会带来什么问题？
  A：特征之间强相关会让 $X^T X$ 病态或不可逆，参数估计不稳定；可用 `lstsq`、伪逆或 Ridge 正则缓解。

```python
import numpy as np


class MultipleLinearRegressionNumpy:
    def __init__(self):
        self.w = None
        self.b = 0.0

    def fit_closed_form(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        X_aug = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        self.w = theta[:n_features]
        self.b = theta[n_features]
        return self

    def fit_gradient_descent(self, X, y, lr=1e-2, epochs=1000):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for _ in range(epochs):
            pred = X @ self.w + self.b
            error = pred - y
            grad_w = (2.0 / n_samples) * (X.T @ error)
            grad_b = (2.0 / n_samples) * np.sum(error)
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def mse(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        return np.mean((pred - y) ** 2)
```

### 72. Logistic Regression

介绍：Logistic Regression 是线性二分类模型，用 $\sigma(w^Tx+b)$ 输出正类概率，并用二元交叉熵训练。虽然名字带 regression，但它主要用于分类。

公式：$p=\sigma(w^Tx+b)$；$L=-[y\log p+(1-y)\log(1-p)]$。

- Q：Logistic Regression 输出的是什么？
  A：输出正类概率，通常是 $\sigma(s)$，其中 $s=w^Tx+b$。
- Q：它和线性回归有什么区别？
  A：线性回归预测连续值并常用 MSE；逻辑回归预测概率并常用 BCE。
- Q：它的决策边界是什么形状？
  A：在原始特征空间中是线性超平面。

```python
class LogisticRegressionNumpy:
    def __init__(self, lr=1e-2, epochs=1000, reg=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for _ in range(self.epochs):
            probs = self._sigmoid(X @ self.w + self.b)
            error = probs - y
            grad_w = X.T @ error / n_samples + self.reg * self.w
            grad_b = error.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)
```

### 73. SVM

介绍：SVM 是最大间隔分类器。线性二分类 SVM 常用 hinge loss：$\max(0,1-y(w^Tx+b))$，其中标签通常取 $\{-1,+1\}$。训练目标是在分类错误和间隔不足时产生损失，同时用 L2 正则控制权重大小。

公式：$L=\max(0,1-y(w^Tx+b))$。

- Q：SVM 的 hinge loss 是什么？
  A：$\max(0,1-ys)$，其中 $s=w^Tx+b$；当样本分类正确且间隔大于 1 时 loss 为 0，否则产生线性惩罚。
- Q：SVM 和 Logistic Regression 的区别是什么？
  A：SVM 关注最大间隔，使用 hinge loss；Logistic Regression 直接建模概率，使用 log loss。
- Q：支持向量是什么？
  A：位于间隔边界上或违反间隔约束的样本，它们决定最终分类超平面。

```python
import numpy as np


class LinearSVM:
    def __init__(self, lr=1e-2, reg=1e-3, epochs=1000):
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        y = np.where(y <= 0, -1.0, 1.0)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for _ in range(self.epochs):
            scores = X @ self.w + self.b
            margins = y * scores
            active = margins < 1

            grad_w = self.reg * self.w
            grad_b = 0.0
            if np.any(active):
                grad_w -= (X[active].T @ y[active]) / n_samples
                grad_b -= np.sum(y[active]) / n_samples

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
```

### 74. KNN

介绍：KNN 是非参数、懒学习分类算法。训练阶段只保存样本；预测时计算测试样本到训练样本的距离，取最近的 k 个邻居投票。它简单直观，但预测成本较高，并且对特征尺度敏感。

公式：$\hat{y}=\operatorname{mode}\{y_i:i\in\mathcal{N}_k(x)\}$。

- Q：KNN 的训练过程是什么？
  A：几乎没有显式训练，只保存训练数据和标签，预测时再计算距离和投票。
- Q：k 值太小或太大会怎样？
  A：k 太小容易受噪声影响、方差高；k 太大可能过度平滑、偏差高。
- Q：为什么 KNN 前常做标准化？
  A：距离度量会被数值范围大的特征主导，标准化能让各维特征尺度更可比。

```python
import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=np.float64)
        x2 = np.sum(X * X, axis=1, keepdims=True)
        train2 = np.sum(self.X_train * self.X_train, axis=1, keepdims=True).T
        dist2 = x2 + train2 - 2 * X @ self.X_train.T
        return np.sqrt(np.maximum(dist2, 0.0))

    def predict(self, X):
        distances = self._distances(X)
        nearest = np.argsort(distances, axis=1)[:, : self.k]
        preds = []
        for idx in nearest:
            labels = self.y_train[idx]
            counts = np.array([np.sum(labels == c) for c in self.classes_])
            preds.append(self.classes_[np.argmax(counts)])
        return np.asarray(preds)
```

### 75. Naive Bayes

介绍：Naive Bayes 基于贝叶斯公式，并假设特征在类别条件下相互独立。Gaussian Naive Bayes 常用于连续特征，为每个类别的每个特征估计均值和方差。

公式：$\hat{y}=\arg\max_c P(c)\prod_j P(x_j|c)$。

- Q：Naive Bayes 的“朴素”指什么？
  A：指条件独立假设，即给定类别后各特征相互独立。
- Q：为什么它训练很快？
  A：只需要按类别统计先验、均值、方差或词频等数量。
- Q：条件独立假设不成立还能用吗？
  A：很多场景仍然能取得不错效果，尤其是文本分类等高维稀疏任务。

```python
class GaussianNaiveBayes:
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.prior_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.mean_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0) + 1e-12
            self.prior_[c] = X_c.shape[0] / X.shape[0]
        return self

    def _log_prob(self, X, c):
        mean = self.mean_[c]
        var = self.var_[c]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
        return np.log(self.prior_[c]) + log_likelihood

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        log_probs = np.column_stack([self._log_prob(X, c) for c in self.classes_])
        return self.classes_[np.argmax(log_probs, axis=1)]
```

### 76. Decision Tree

介绍：Decision Tree 通过递归选择特征和阈值切分数据，使子节点更“纯”。分类树常用 Gini impurity 或 entropy 作为划分标准，优点是可解释，缺点是容易过拟合。

公式：$G=1-\sum_c p_c^2$；$H=-\sum_c p_c\log p_c$。

- Q：分类树常用什么划分指标？
  A：Gini impurity、entropy/information gain。
- Q：决策树为什么容易过拟合？
  A：树太深时可以记住训练样本的细节和噪声。
- Q：怎么限制决策树复杂度？
  A：限制最大深度、最小样本数、最小增益，或进行剪枝。

```python
class DecisionTreeClassifierNumpy:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs * probs)

    def _best_split(self, X, y):
        best = None
        best_score = float("inf")
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                score = left.mean() * self._gini(y[left]) + right.mean() * self._gini(y[right])
                if score < best_score:
                    best_score = score
                    best = (feature, threshold)
        return best

    def _build(self, X, y, depth):
        values, counts = np.unique(y, return_counts=True)
        pred = values[np.argmax(counts)]
        if depth >= self.max_depth or len(values) == 1 or len(y) < self.min_samples_split:
            return {"pred": pred}
        split = self._best_split(X, y)
        if split is None:
            return {"pred": pred}
        feature, threshold = split
        left = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
            "pred": pred,
        }

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X, dtype=np.float64), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        while "feature" in node:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["pred"]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.asarray([self._predict_one(x, self.tree_) for x in X])
```

### 77. Random Forest

介绍：Random Forest 训练多棵决策树，并通过 bootstrap 采样和特征随机性降低树之间的相关性。预测时分类任务多数投票，回归任务平均，通常比单棵树更稳健。

公式：$\hat{y}=\operatorname{mode}\{T_b(x)\}_{b=1}^{B}$。

- Q：Random Forest 的随机性来自哪里？
  A：样本 bootstrap 采样和每次分裂时的特征子集采样。
- Q：它为什么能降低过拟合？
  A：多棵弱相关树做集成可以降低方差。
- Q：Random Forest 和 GBDT 的区别是什么？
  A：Random Forest 多棵树并行降低方差，GBDT 串行拟合残差或梯度降低偏差。

```python
class RandomForestClassifierNumpy:
    def __init__(self, n_trees=10, max_depth=4, sample_ratio=1.0, max_features=None, random_state=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        sample_size = max(1, int(n_samples * self.sample_ratio))
        max_features = self.max_features or max(1, int(np.sqrt(n_features)))
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n_samples, size=sample_size, replace=True)
            feature_idx = rng.choice(n_features, size=max_features, replace=False)
            tree = DecisionTreeClassifierNumpy(max_depth=self.max_depth)
            tree.fit(X[idx][:, feature_idx], y[idx])
            self.trees_.append((tree, feature_idx))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.stack([tree.predict(X[:, feature_idx]) for tree, feature_idx in self.trees_], axis=1)
        preds = []
        for row in all_preds:
            values, counts = np.unique(row, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.asarray(preds)
```

### 78. GBDT / XGBoost

介绍：GBDT 通过串行训练多棵树，每棵新树拟合当前模型的残差或负梯度。XGBoost 是 GBDT 的工程和目标函数增强版本，使用一阶、二阶梯度、正则化、列采样、缺失值处理等机制。

公式：$\hat{y}^{(m)}(x)=\hat{y}^{(m-1)}(x)+\eta h_m(x)$；$h_m(x)\approx-\frac{\partial L}{\partial \hat{y}(x)}$。

- Q：GBDT 每棵树拟合什么？
  A：拟合当前损失函数的负梯度；平方损失下就是残差。
- Q：GBDT 和 Random Forest 的训练方式有什么区别？
  A：GBDT 串行逐步纠错，Random Forest 多棵树相对独立并行训练。
- Q：XGBoost 相比普通 GBDT 常见增强是什么？
  A：二阶梯度、正则化目标、列采样、稀疏特征处理和高效工程实现。

```python
class RegressionStump:
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        best_error = float("inf")
        self.feature_ = 0
        self.threshold_ = 0.0
        self.left_value_ = y.mean()
        self.right_value_ = y.mean()
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                left_value = y[left].mean()
                right_value = y[right].mean()
                pred = np.where(left, left_value, right_value)
                error = np.mean((pred - y) ** 2)
                if error < best_error:
                    best_error = error
                    self.feature_ = feature
                    self.threshold_ = threshold
                    self.left_value_ = left_value
                    self.right_value_ = right_value
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.where(X[:, self.feature_] <= self.threshold_, self.left_value_, self.right_value_)


class SimpleGBDTRegressor:
    def __init__(self, n_estimators=20, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.init_ = 0.0
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.init_ = y.mean()
        pred = np.full_like(y, self.init_, dtype=np.float64)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = RegressionStump().fit(X, residual)
            update = tree.predict(X)
            pred += self.lr * update
            self.trees_.append(tree)
        return self

    def predict(self, X):
        pred = np.full(np.asarray(X).shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            pred += self.lr * tree.predict(X)
        return pred
```

### 79. K-Means

介绍：K-Means 是无监督聚类算法。它交替执行两步：把每个样本分配给最近的中心点，然后用每个簇内样本均值更新中心点。目标是最小化样本到所属中心的平方距离之和。

公式：$\min_{\mu_1,\ldots,\mu_K}\sum_i\min_k\|x_i-\mu_k\|_2^2$。

- Q：K-Means 的优化目标是什么？
  A：最小化簇内平方误差，即所有样本到其所属中心点的平方距离之和。
- Q：K-Means 为什么对初始化敏感？
  A：它是非凸优化，不同初始中心可能收敛到不同局部最优。
- Q：空簇怎么处理？
  A：可以保留旧中心，或随机选一个样本重新初始化该中心。

```python
import numpy as np


class KMeansNumpy:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        init_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[init_idx].copy()

        for _ in range(self.max_iter):
            dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1)

            new_centers = centers.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        self.labels_ = np.argmin(dist2, axis=1)
        self.centers_ = centers
        self.inertia_ = np.sum(dist2[np.arange(n_samples), self.labels_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        dist2 = np.sum((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        return np.argmin(dist2, axis=1)
```

### 80. PCA

介绍：PCA 是线性降维方法，通过寻找方差最大的正交方向，把高维数据投影到低维空间。它常用于可视化、降噪、压缩和缓解特征冗余。

公式：$w_1=\arg\max_{\|w\|_2=1}w^T\Sigma w$。

- Q：PCA 优化目标是什么？
  A：找到投影后方差最大的方向，等价于最小化线性重构误差。
- Q：PCA 前为什么要中心化？
  A：PCA 关注协方差结构，需要先去掉均值影响。
- Q：PCA 是有监督还是无监督？
  A：无监督，不使用标签，只使用特征矩阵。

```python
class PCANumpy:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        self.explained_variance_ = (s[: self.n_components] ** 2) / max(1, X.shape[0] - 1)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### 81. MLP

介绍：MLP（Multi-Layer Perceptron，多层感知机）由多层全连接层和非线性激活函数组成，是最基础的前馈神经网络。一个常见二层 MLP 结构是 `Linear -> ReLU -> Linear -> Softmax`。NumPy 手写 MLP 的重点是掌握矩阵乘法、前向传播、交叉熵损失、反向传播和参数更新。

公式：$h=\phi(xW_1+b_1)$；$\hat{y}=\operatorname{softmax}(hW_2+b_2)$。

- Q：MLP 为什么需要非线性激活？
  A：如果没有非线性，多层线性变换仍然等价于一个线性变换，模型表达能力不会随层数增加。
- Q：Softmax + Cross Entropy 的梯度是什么？
  A：若 $p$ 是 softmax 概率、$y$ 是 one-hot 标签，对 logits 的梯度是 $(p-y)/B$，这是分类 MLP 反向传播里最常用的简化结果。
- Q：MLP 的主要参数在哪里？
  A：每个全连接层都有权重矩阵和 bias，例如 $W_1,b_1$ 和 $W_2,b_2$；参数量随输入维度、隐藏层宽度和类别数线性/乘法增长。

```python
import numpy as np


class MLPClassifierNumpy:
    def __init__(self, input_dim, hidden_dim, num_classes, lr=1e-2, reg=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.reg = reg
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes)

    def _relu(self, x):
        return np.maximum(x, 0.0)

    def _softmax(self, logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        logits = a1 @ self.W2 + self.b2
        cache = {"X": X, "z1": z1, "a1": a1, "logits": logits}
        return logits, cache

    def loss_and_grads(self, X, y):
        y = np.asarray(y, dtype=np.int64)
        logits, cache = self.forward(X)
        probs = self._softmax(logits)
        n = y.shape[0]

        data_loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        reg_loss = 0.5 * self.reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        loss = data_loss + reg_loss

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        dW2 = cache["a1"].T @ dlogits + self.reg * self.W2
        db2 = dlogits.sum(axis=0)
        da1 = dlogits @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["X"].T @ dz1 + self.reg * self.W1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads

    def step(self, grads):
        self.W1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.W2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    def fit(self, X, y, epochs=200):
        losses = []
        for _ in range(epochs):
            loss, grads = self.loss_and_grads(X, y)
            self.step(grads)
            losses.append(loss)
        return losses

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return self._softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
```

## Attention

### 82. Scaled Dot-Product Attention

介绍：Scaled dot-product attention 的公式是 $\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$。缩放因子 $\sqrt{d_k}$ 用来控制点积方差，避免 softmax 输入过大后梯度变小。

公式：$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$。

- Q：Attention 的输入输出 shape 是什么？
  A：`Q:(B,S_q,d_k)`，`K:(B,S_k,d_k)`，`V:(B,S_k,d_v)`，输出是 `(B,S_q,d_v)`。
- Q：为什么除以 $\sqrt{d_k}$？
  A：点积方差随维度增大而增大，缩放能防止 softmax 过早饱和。
- Q：Attention 的主要复杂度在哪里？
  A：主要是 $QK^T$ 和 attention matrix，时间复杂度约 $O(S_qS_kd_k)$，显存常受 $S_qS_k$ 限制。

```python
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

### 83. QK Norm

介绍：QK Norm 是在计算 attention score 前对 Query 和 Key 做归一化，常用于稳定点积尺度，避免 attention logits 过大导致 softmax 过早饱和。常见做法是对每个 head 的最后一维做 L2 normalize 或 RMSNorm，再配合可学习缩放系数。

公式：$\tilde{Q}=\frac{Q}{\|Q\|_2+\epsilon}$；$\tilde{K}=\frac{K}{\|K\|_2+\epsilon}$；$A=\operatorname{softmax}(\gamma\tilde{Q}\tilde{K}^T)V$。

- Q：QK Norm 解决什么问题？
  A：控制 Q/K 点积的尺度，让 attention logits 更稳定，减少 softmax 饱和和训练不稳定。
- Q：QK Norm 通常归一化哪一维？
  A：通常对每个 token、每个 head 的 head dimension 做归一化，也就是 Q/K 的最后一维。
- Q：QK Norm 和除以 $\sqrt{d_k}$ 是什么关系？
  A：二者都在控制 attention score 的尺度；QK Norm 直接限制 Q/K 向量范数，常再配合可学习缩放系数。

```python
def qk_norm_attention(Q: Tensor, K: Tensor, V: Tensor, scale: float = 1.0, eps: float = 1e-6) -> Tensor:
    Q_norm = Q / (Q.norm(dim=-1, keepdim=True) + eps)
    K_norm = K / (K.norm(dim=-1, keepdim=True) + eps)
    scores = scale * torch.bmm(Q_norm, K_norm.transpose(1, 2))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

### 84. Multi-Head Attention

介绍：Multi-head attention 先把 Q/K/V 投影并拆成多个 head，每个 head 独立做 attention，再拼接并经过输出投影。核心是正确处理 `(B, S, D)` 和 `(B, H, S, d_k)` 之间的 reshape/transpose。

公式：$\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(H_1,\ldots,H_h)W^O$。

- Q：Multi-head attention 为什么要拆多个 head？
  A：多个 head 可以在不同子空间学习不同关系，再拼接融合，提高表达能力。
- Q：为什么 `d_model` 要能被 `num_heads` 整除？
  A：每个 head 的维度通常是 $d_{\mathrm{model}}/h$，必须是整数才能 reshape。
- Q：实现中为什么常用 `.contiguous().view(...)`？
  A：`transpose` 后 Tensor 内存可能不连续，`contiguous()` 确保后续 `view` 按预期工作。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        B, S_q, _ = Q.shape
        S_k = K.shape[1]
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(out)
```

### 85. Causal Self-Attention

介绍：Causal self-attention 在普通 attention 上加入上三角 mask，使位置 `i` 只能看见 $j\le i$ 的 token。这是 GPT 类自回归模型训练和生成的核心约束。

公式：$M_{ij}=0\ (j\le i)$；$M_{ij}=-\infty\ (j>i)$。

- Q：Causal mask 应该屏蔽哪些位置？
  A：对位置 `i`，所有未来位置 $j>i$ 都要屏蔽，通常用上三角 mask。
- Q：训练时为什么还能并行计算整段序列？
  A：虽然一次计算所有位置，但 mask 保证每个位置只看见过去和自己。
- Q：第一个 token 的 attention 有什么特殊性？
  A：第一个 token 没有历史，只能 attend 到自己。

```python
def causal_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

### 86. Attention Sink

介绍：Attention sink 指自回归模型中，后面的 token 往往会给序列开头的一个或几个 token 分配较多 attention 权重，即使这些 token 不一定有直接语义相关性。常见 sink token 包括 BOS 或最前面的若干 token。长上下文流式推理中常保留 sink token 的 KV cache，再配合滑动窗口保留最近 token。

公式：$\alpha_{i,j}=\operatorname{softmax}\left(\frac{q_i k_j^T}{\sqrt{d_k}}\right)_j$；$S_i=\sum_{j=1}^{s}\alpha_{i,j}$。

- Q：Attention sink 是什么现象？
  A：很多后续位置会稳定关注序列开头 token，使这些 token 像 attention 权重的“吸收点”。
- Q：Attention sink 和 BOS 有什么关系？
  A：BOS 常位于序列开头，容易成为 sink token；但 sink 也可能出现在前几个普通 token 上。
- Q：流式推理为什么保留 sink token？
  A：只保留最近窗口可能破坏模型已学到的开头锚点；保留少量 sink token 可以提升长序列生成稳定性。

```python
def attention_sink_mass(Q: Tensor, K: Tensor, sink_len: int = 4) -> Tensor:
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    seq_len = scores.size(-1)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights[..., :sink_len].sum(dim=-1)
```

### 87. Attention Mask

介绍：Attention mask 用来在 softmax 前屏蔽不允许关注的位置。Padding mask 屏蔽补齐 token，避免模型关注无效内容；causal mask 屏蔽未来 token，保证自回归生成只能看见当前位置及之前的信息。

公式：$A=\operatorname{softmax}\left(\frac{QK^T+M}{\sqrt{d_k}}\right)V$。

- Q：Attention mask 加在 softmax 前还是后？
  A：通常加在 softmax 前，把被屏蔽位置的 score 设成很小的负数。
- Q：Padding mask 解决什么问题？
  A：屏蔽 batch padding 出来的无效 token，避免它们参与 attention。
- Q：Causal mask 解决什么问题？
  A：屏蔽未来 token，防止自回归模型训练时信息泄漏。

```python
def masked_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)


def build_causal_mask(seq_len: int, device=None) -> Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)


def build_padding_mask(valid_lengths: Tensor, max_len: int) -> Tensor:
    positions = torch.arange(max_len, device=valid_lengths.device).unsqueeze(0)
    return (positions >= valid_lengths.unsqueeze(1)).unsqueeze(1)
```

### 88. Multi-Query Attention

介绍：Multi-Query Attention（MQA）让多个 Query head 共享同一组 Key/Value head。相比 MHA，它显著减少 KV cache 的 head 数和解码阶段的显存带宽，因此常用于大模型推理加速；代价是不同 query head 可用的 K/V 子空间更少。

公式：$Q\in\mathbb{R}^{B\times H\times S\times d}$；$K,V\in\mathbb{R}^{B\times 1\times S\times d}$；$A_h=\operatorname{softmax}\left(\frac{Q_hK^T}{\sqrt{d}}\right)V$。

- Q：MQA 和 MHA 的区别是什么？
  A：MHA 每个 query head 有独立 K/V；MQA 所有 query head 共享同一组 K/V。
- Q：MQA 为什么能节省 KV cache？
  A：KV cache 只需要保存 1 组 K/V head，而不是保存全部 query head 对应的 K/V。
- Q：MQA 的潜在缺点是什么？
  A：K/V 表达能力可能低于 MHA，因为所有 query head 共享同一组 K/V 子空间。

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, 1, S, self.d_k)
        v = self.W_v(x).view(B, 1, S, self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### 89. Grouped Query Attention

介绍：GQA 让 Query 使用较多 head，但 Key/Value 使用较少 head，并让多个 query head 共享同一组 KV。它可以减少 KV cache 显存和解码带宽，是 MHA 和 MQA 之间的折中。

公式：$g=H_q/H_{kv}$；$Q\in\mathbb{R}^{B\times H_q\times S\times d}$；$K,V\in\mathbb{R}^{B\times H_{kv}\times S\times d}$。

- Q：MHA、MQA、GQA 的区别是什么？
  A：MHA 每个 Q head 有独立 KV；MQA 所有 Q head 共享一个 KV；GQA 是多个 Q head 共享一组 KV。
- Q：GQA 为什么能减少 KV cache？
  A：KV head 数少于 Q head 数，缓存的 K/V 张量 head 维度更小。
- Q：实现中为什么要 repeat K/V？
  A：为了让较少的 KV head 对齐到较多的 Q head，每组 query head 共享对应 K/V。

```python
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### 90. Sliding Window Attention

介绍：Sliding window attention 只允许每个位置关注局部窗口内的 token，即 $|i-j|\le w$，其中 $w$ 是窗口半径。它把长序列 attention 的计算从全局二次复杂度压到近似局部复杂度。

公式：$M_{ij}=0\ (|i-j|\le w)$；$M_{ij}=-\infty\ (|i-j|>w)$。

- Q：Sliding Window Attention 的 mask 条件是什么？
  A：只保留 $|i-j|\le w$ 的位置，其余 score 设为 $-\infty$。
- Q：它相比全局 attention 节省什么？
  A：每个 token 只看局部窗口，长序列下计算和显存可从二次规模降到窗口相关规模。
- Q：$w=0$ 和超大窗口分别对应什么？
  A：$w=0$ 只看自己；窗口覆盖全序列时等价于普通 full attention。

```python
def sliding_window_attention(Q: Tensor, K: Tensor, V: Tensor, window_size: int) -> Tensor:
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

### 91. Linear Attention

介绍：Linear attention 用非负特征映射 $\phi(x)$ 替代 softmax，并把计算重排为 $\phi(Q)(\phi(K)^TV)$。这样不显式构造 $S^2$ attention matrix，适合长序列近似计算。

公式：$A=\frac{\phi(Q)(\phi(K)^TV)}{\phi(Q)\sum_i\phi(K_i)}$。

- Q：Linear Attention 的核心重排是什么？
  A：先算 $\phi(K)^TV$，再左乘 $\phi(Q)$，避免构造 $S^2$ 矩阵。
- Q：Linear Attention 和 Softmax Attention 完全等价吗？
  A：通常不完全等价；它使用 feature map 替代 softmax，是一种近似或替代形式。
- Q：分母为什么需要 $\phi(Q)\cdot\sum\phi(K)$？
  A：它起归一化作用，类似 softmax attention 中每行概率和为 1。

```python
def linear_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    Q_phi = F.elu(Q) + 1
    K_phi = F.elu(K) + 1
    KV = torch.bmm(K_phi.transpose(1, 2), V)
    K_sum = K_phi.sum(dim=1, keepdim=True)
    numerator = torch.bmm(Q_phi, KV)
    denominator = torch.bmm(Q_phi, K_sum.transpose(1, 2))
    return numerator / (denominator + 1e-6)
```

### 92. Flash Attention

介绍：Flash Attention 不改变 attention 公式，而是用分块和 online softmax 避免物化完整 $S^2$ 注意力矩阵，从而减少显存读写。

公式：$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$。

- Q：Flash Attention 改变 attention 结果吗？
  A：不改变；它与标准 softmax attention 数学等价，只改变计算和内存访问方式。
- Q：Online softmax 需要维护什么？
  A：每行的 running max、running sum 和累积加权 value。
- Q：Flash Attention 主要优化什么瓶颈？
  A：主要优化 HBM 显存读写和 $S^2$ attention matrix 的存储开销。

```python
def flash_attention(Q: Tensor, K: Tensor, V: Tensor, block_size: int = 32) -> Tensor:
    B, S, D = Q.shape
    output = torch.zeros_like(V)

    for i in range(0, S, block_size):
        qi = Q[:, i : i + block_size]
        bq = qi.shape[1]
        row_max = torch.full((B, bq, 1), float("-inf"), device=Q.device, dtype=Q.dtype)
        row_sum = torch.zeros(B, bq, 1, device=Q.device, dtype=Q.dtype)
        acc = torch.zeros(B, bq, V.size(-1), device=Q.device, dtype=Q.dtype)

        for j in range(0, S, block_size):
            kj = K[:, j : j + block_size]
            vj = V[:, j : j + block_size]
            scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            acc = acc * correction + torch.bmm(exp_scores, vj)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max

        output[:, i : i + block_size] = acc / row_sum

    return output
```

## Transformer

### 93. Sinusoidal Position Encoding

介绍：Sinusoidal Position Encoding 是 Transformer 原论文中的固定位置编码。它用不同频率的正弦和余弦函数表示位置：偶数维用 $\sin\left(p/10000^{2i/d}\right)$，奇数维用 $\cos\left(p/10000^{2i/d}\right)$。它不需要训练参数，并且能一定程度外推到比训练更长的位置。

公式：$PE_{p,2i}=\sin(p/10000^{2i/d})$；$PE_{p,2i+1}=\cos(p/10000^{2i/d})$。

- Q：为什么要用不同频率的 sin/cos？
  A：不同频率能表示不同尺度的位置变化，类似给每个位置一个多尺度编码。
- Q：Sinusoidal 位置编码有训练参数吗？
  A：没有，它是固定函数生成的。
- Q：Sinusoidal 位置编码怎么加到模型里？
  A：通常直接加到 token embedding 上，形状和 hidden size 一致。

```python
def sinusoidal_position_encoding(seq_len: int, d_model: int, device=None) -> Tensor:
    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
```

### 94. RoPE

介绍：RoPE 把向量的相邻维度两两成对，并按位置相关角度旋转。它保持向量范数，同时让 Q/K 点积自然包含相对位置信息。

公式：$R_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}$；$q_p^{\prime}=R_{\theta_p}q_p$。

- Q：RoPE 为什么只作用在 Q/K 上？
  A：位置关系通过 attention score 的 Q/K 点积体现，V 只承载被聚合的信息内容。
- Q：RoPE 是否改变向量范数？
  A：不会；二维旋转矩阵是正交变换，保持每对维度的范数。
- Q：RoPE 怎么体现相对位置？
  A：两个位置的旋转角差会进入 Q/K 点积，使 score 与相对距离相关。

```python
def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    B, S, D = q.shape
    assert D % 2 == 0
    pos = torch.arange(S, device=q.device, dtype=q.dtype).unsqueeze(1)
    dim = torch.arange(0, D, 2, device=q.device, dtype=q.dtype)
    freqs = 1.0 / (10000.0 ** (dim / D))
    angles = pos * freqs
    cos = torch.cos(angles).view(1, S, D // 2)
    sin = torch.sin(angles).view(1, S, D // 2)

    def rotate(x: Tensor) -> Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)

    return rotate(q), rotate(k)
```

### 95. Residual Connection

介绍：Residual connection 把子层输出和输入相加，形式是 $y=x+\operatorname{Sublayer}(x)$。它让梯度可以通过近似恒等路径传播，是深层 CNN、Transformer 和现代网络稳定训练的关键结构。

公式：$y=x+\operatorname{Sublayer}(x)$。

- Q：残差连接解决什么问题？
  A：缓解深层网络优化困难，让梯度更容易向前面层传播。
- Q：残差相加要求什么？
  A：`x` 和 $\operatorname{Sublayer}(x)$ 的 shape 必须一致，或通过投影把维度对齐。
- Q：Transformer 里残差连接放在哪里？
  A：通常每个 attention 子层和 FFN 子层外都有残差连接。

```python
class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)
```

### 96. Pre-Norm

介绍：Pre-Norm 指在 Transformer 子层之前做 LayerNorm，典型形式是 $x+\operatorname{Sublayer}(\operatorname{LN}(x))$。它让残差路径更接近恒等映射，梯度更容易跨层传播，因此深层 Transformer 和 GPT 类模型常用 Pre-Norm。

公式：$y=x+\operatorname{Sublayer}(\operatorname{LN}(x))$。

- Q：Pre-Norm 的计算顺序是什么？
  A：先对输入做 LayerNorm，再送入 attention 或 FFN 子层，最后和原输入做残差相加。
- Q：Pre-Norm 为什么训练更稳定？
  A：残差主路径不经过归一化层，梯度更容易沿恒等路径传播到浅层。
- Q：Pre-Norm 的常见缺点是什么？
  A：最终输出可能缺少统一归一化，很多模型会在所有 block 后再加一个 final LayerNorm。

```python
class PreNormBlock(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.sublayer(self.norm(x))
```

### 97. Post-Norm

介绍：Post-Norm 指在 Transformer 子层和残差相加之后做 LayerNorm，典型形式是 $\operatorname{LN}(x+\operatorname{Sublayer}(x))$。它更接近 Transformer 原论文结构，但在很深的模型里梯度传播通常不如 Pre-Norm 稳定。

公式：$y=\operatorname{LN}(x+\operatorname{Sublayer}(x))$。

- Q：Post-Norm 的计算顺序是什么？
  A：先计算子层输出并和输入做残差相加，再对相加结果做 LayerNorm。
- Q：Post-Norm 和 Pre-Norm 的核心区别是什么？
  A：LayerNorm 的位置不同；Post-Norm 放在残差之后，Pre-Norm 放在子层之前。
- Q：Post-Norm 为什么深层训练更难？
  A：梯度通过残差路径时仍要经过归一化层，深层堆叠时更容易出现训练不稳定。

```python
class PostNormBlock(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.sublayer(x))
```

### 98. SwiGLU MLP

介绍：SwiGLU 是现代 LLM 常用的前馈网络结构：$W_d(\operatorname{SiLU}(W_g x)\odot W_u x)$。它用门控分支增强表达能力。

公式：$\operatorname{SwiGLU}(x)=W_d(\operatorname{SiLU}(W_gx)\odot W_ux)$。

- Q：SwiGLU 的公式是什么？
  A：$W_d(\operatorname{SiLU}(W_g x)\odot W_u x)$。
- Q：SwiGLU 相比普通 FFN 多了什么？
  A：多了 gate 分支，用激活后的 gate 对 up 分支逐元素调制。
- Q：SiLU 的定义是什么？
  A：$\operatorname{SiLU}(x)=x\cdot\sigma(x)$。

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 99. Transformer Encoder Layer

介绍：Transformer encoder layer 由双向 self-attention、前馈网络、残差连接和归一化组成。Pre-norm 把 LayerNorm 放在子层前，深层训练更稳定；post-norm 把 LayerNorm 放在残差之后，更接近原论文结构。

公式：$x^{\prime}=x+\operatorname{MHA}(\operatorname{LN}(x))$；$y=x^{\prime}+\operatorname{FFN}(\operatorname{LN}(x^{\prime}))$。

- Q：Encoder layer 的两个主要子层是什么？
  A：多头 self-attention 和逐位置前馈网络 FFN。
- Q：Pre-norm 的优点是什么？
  A：梯度更容易穿过残差路径，深层 Transformer 训练通常更稳定。
- Q：Encoder 和 decoder-only block 的 mask 有什么区别？
  A：Encoder self-attention 通常双向可见，decoder-only 自注意力需要 causal mask。

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.ln1(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

### 100. Decoder-only Transformer Block

介绍：Decoder-only Transformer block 是 GPT 类语言模型的基本结构。典型结构是 pre-norm：先 `LayerNorm -> causal self-attention -> residual`，再 `LayerNorm -> MLP -> residual`。

公式：$x^{\prime}=x+\operatorname{MHA}_{\mathrm{causal}}(\operatorname{LN}(x))$；$y=x^{\prime}+\operatorname{MLP}(\operatorname{LN}(x^{\prime}))$。

- Q：Decoder-only Transformer block 的基本结构是什么？
  A：pre-norm causal self-attention 加残差，然后 pre-norm MLP 加残差。
- Q：Pre-norm 和 Post-norm 有什么区别？
  A：Pre-norm 在子层前做归一化，深层网络中梯度更稳定；Post-norm 在残差后归一化。
- Q：Decoder-only Transformer block 的 MLP 通常怎么设计？
  A：常见是 `Linear(d, 4d) -> Activation -> Linear(4d, d)`。

```python
class DecoderOnlyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _attn(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### 101. KV Cache

介绍：KV cache 用于自回归推理。历史 token 的 K/V 不会变化，因此每步只需要计算新 token 的 Q/K/V，并把新 K/V 追加到 cache，避免重复计算整段历史。

公式：$K_{1:t}=[K_{1:t-1};K_t]$；$V_{1:t}=[V_{1:t-1};V_t]$。

- Q：KV cache 保存的张量形状是什么？
  A：常见形状是 `(B, num_heads, S_past, d_k)`，分别保存历史 K 和 V。
- Q：Prefill 和 decode 有什么区别？
  A：Prefill 一次处理 prompt 多个 token，需要 causal mask；decode 通常每次处理一个新 token。
- Q：KV cache 的代价是什么？
  A：减少重复计算，但显著增加随 batch、层数、序列长度增长的显存占用。

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, cache=None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, S_new, _ = x.shape
        q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        S_total = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if S_new > 1:
            S_past = S_total - S_new
            mask = torch.triu(
                torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
                diagonal=S_past + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.W_o(out), (k, v)
```


### 102. Mixture of Experts

介绍：MoE 使用 router 为每个 token 选择 top-k expert，并对这些 expert 输出做加权求和。它增加参数容量，但每个 token 只激活少量专家。

公式：$p=\operatorname{softmax}(W_rx)$；$y=\sum_{e\in\mathcal{S}}p_e f_e(x)$。

- Q：MoE 的 router 输出是什么？
  A：对每个 token 输出各 expert 的 logits，用来选择 top-k expert。
- Q：MoE 为什么能增加参数但不同比例增加计算？
  A：因为每个 token 只激活 top-k 个 expert，而不是运行所有 expert。
- Q：MoE 训练常见问题是什么？
  A：expert 负载不均衡，部分 expert 过热、部分 expert 很少被选中。

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        logits = self.router(x_flat)
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            for expert_id, expert in enumerate(self.experts):
                mask = top_idx[:, k] == expert_id
                if mask.any():
                    output[mask] += weights[mask, k : k + 1] * expert(x_flat[mask])

        return output.reshape(orig_shape)
```

## 推理与解码

### 103. Greedy Decoding

介绍：Greedy decoding 每一步都选择当前概率最高的 token。它实现简单、速度快、结果确定，但容易陷入局部最优，生成文本可能重复或缺乏多样性。

公式：$y_t=\arg\max_i p_i$。

- Q：Greedy decoding 怎么选 token？
  A：每一步取 logits 最大的类别，也就是 `argmax`。
- Q：Greedy 的优点是什么？
  A：简单、确定、速度快，不需要维护多个候选。
- Q：Greedy 的缺点是什么？
  A：只看当前最优，可能错过整体更好的序列。

```python
def greedy_decode_step(logits: Tensor) -> Tensor:
    return torch.argmax(logits, dim=-1)
```

### 104. Temperature Sampling

介绍：Temperature sampling 先把 logits 除以 temperature 再采样。temperature 越小，分布越尖锐，输出越保守；temperature 越大，分布越平滑，随机性和多样性越强。

公式：$p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}$。

- Q：temperature 小于 1 会怎样？
  A：高概率 token 更突出，生成更确定、更保守。
- Q：temperature 大于 1 会怎样？
  A：概率分布更平，低概率 token 更容易被采到。
- Q：temperature 接近 0 时接近什么解码？
  A：接近 greedy decoding。

```python
def temperature_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    scaled_logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### 105. Top-k / Top-p Sampling

介绍：Top-k 只保留概率最高的 k 个 token；top-p 按概率排序后保留累计概率不超过 p 的候选集合。二者常用于 LLM 开放式生成。

公式：$\mathcal{V}_k=\operatorname{TopK}(p,k)$；$\mathcal{V}_p=\min\{S:\sum_{i\in S}p_i\ge p\}$。

- Q：Top-k 和 Top-p 的区别是什么？
  A：Top-k 固定候选数量，Top-p 固定累计概率阈值，候选数量会随分布变化。
- Q：temperature 先做还是过滤先做？
  A：通常先对 logits 做 temperature 缩放，再执行 top-k/top-p 过滤。
- Q：过滤后为什么要重新归一化概率？
  A：被过滤 token 的概率会置为 0，剩余概率和不一定是 1，所以采样前要重新归一化。

```python
def sample_top_k_top_p(logits: Tensor, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> int:
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)

    if top_k > 0:
        top_k = min(top_k, logits.numel())
        sorted_idx = torch.argsort(probs, descending=True)
        keep_top_k = torch.zeros_like(keep)
        keep_top_k[sorted_idx[:top_k]] = True
        keep = keep & keep_top_k

    if top_p < 1.0:
        sorted_idx = torch.argsort(probs, descending=True)
        keep_top_p = torch.zeros_like(keep)
        cumulative_prob = 0.0
        for idx in sorted_idx:
            keep_top_p[idx] = True
            cumulative_prob += probs[idx].item()
            if cumulative_prob >= top_p:
                break
        keep = keep & keep_top_p

    probs = probs * keep.float()
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()
```

### 106. Beam Search

介绍：Beam search 维护多个候选序列，每一步扩展并保留累计 logprob 最高的若干 beam。它适合较确定性的序列生成任务。

公式：$S(y_{1:t})=\sum_{i=1}^{t}\log p(y_i|y_{<i})$。

- Q：Beam search 的分数为什么用 logprob 累加？
  A：概率连乘容易下溢，取 log 后序列概率乘积变成加法。
- Q：Beam width 为 1 时是什么？
  A：退化为 greedy decoding，每步只保留当前最高分候选。
- Q：为什么常加 length penalty？
  A：累计 logprob 往往偏向短序列，length penalty 用来平衡长度偏置。

```python
def beam_search(log_prob_fn, start_token: int, max_len: int, beam_width: int, eos_token: int) -> list[int]:
    beams = [(0.0, [start_token])]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue
            log_probs = log_prob_fn(seq)
            top_scores, top_tokens = torch.topk(log_probs, beam_width)
            for s, tok in zip(top_scores.tolist(), top_tokens.tolist()):
                candidates.append((score + s, seq + [tok]))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        if beams[0][1][-1] == eos_token:
            completed.append(beams[0])
            break

    all_seqs = completed + beams
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    return all_seqs[0][1]
```

### 107. Speculative Decoding

介绍：Speculative decoding 用小 draft 模型先生成候选 token，再由大 target 模型并行验证。接受概率为 $\min(1,p_T/p_D)$，其中 $p_T$ 是 target 概率、$p_D$ 是 draft 概率，在不改变目标分布的前提下加速推理。

公式：$a=\min(1,p_T/p_D)$。

- Q：Speculative decoding 为什么不改变目标分布？
  A：接受/拒绝规则按 target 和 draft 概率比修正，保证最终采样服从 target 分布。
- Q：draft 模型越强有什么影响？
  A：接受率越高，target 模型并行验证带来的加速越明显。
- Q：拒绝某个 token 后为什么要停止当前草稿段？
  A：拒绝说明后续 draft token 的条件上下文已变化，需要重新从修正后的 token 继续生成。

```python
def speculative_decode(target_probs: Tensor, draft_probs: Tensor, draft_tokens: Tensor) -> list[int]:
    accepted = []
    for i in range(len(draft_tokens)):
        token = draft_tokens[i].item()
        ratio = target_probs[i, token] / max(draft_probs[i, token].item(), 1e-10)
        if torch.rand(1).item() < min(1.0, ratio.item()):
            accepted.append(token)
        else:
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            if adjusted.sum() > 0:
                adjusted = adjusted / adjusted.sum()
            else:
                adjusted = torch.ones_like(adjusted) / adjusted.numel()
            accepted.append(torch.multinomial(adjusted, 1).item())
            return accepted
    return accepted
```

## Tokenizer

### 108. BPE Tokenizer

介绍：BPE 从字符级 token 开始，反复合并语料中最高频的相邻 token pair。编码时按训练得到的 merge 顺序应用合并。

公式：$(a,b)^*=\arg\max_{(a,b)}\operatorname{count}(a,b)$。

- Q：BPE 训练时每轮合并什么？
  A：合并当前语料中出现频次最高的相邻 token pair。
- Q：为什么要加 `</w>`？
  A：标记词边界，避免跨词或词尾信息丢失。
- Q：BPE 如何处理没见过的词？
  A：可以退回到更小的子词、字符或字节单元，因此具备开放词表能力。

```python
class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus: list[str], num_merges: int):
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ("</w>",)
            vocab[symbols] = vocab.get(symbols, 0) + 1

        self.merges = []
        for _ in range(num_merges):
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_vocab = {}
            for word, freq in vocab.items():
                merged = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        merged.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_vocab[tuple(merged)] = new_vocab.get(tuple(merged), 0) + freq
            vocab = new_vocab

    def encode(self, text: str) -> list[str]:
        tokens = []
        for word in text.split():
            symbols = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2 :]
                    else:
                        i += 1
            tokens.extend(symbols)
        return tokens
```

### 109. WordPiece Tokenizer

介绍：WordPiece 是 BERT 常用的子词算法。训练时从字符或基础片段出发，每次选择最能提升语料似然的合并；工程上常用 pair score 近似衡量两个片段的关联强度。编码时使用 longest-match-first：从当前位置开始找词表中最长可匹配片段，非词首片段通常加 `##` 前缀。

公式：$s(a,b)=\frac{c(ab)}{c(a)c(b)}$；$(a,b)^*=\arg\max_{(a,b)}s(a,b)$。

- Q：WordPiece 和 BPE 的核心区别是什么？
  A：BPE 通常合并最高频相邻 pair；WordPiece 更强调合并后对语料似然或片段关联强度的提升。
- Q：`##` 前缀表示什么？
  A：表示该 token 是一个词内部的后续子词，不是词首片段，解码时要和前一个片段拼接。
- Q：WordPiece 遇到无法匹配的词怎么办？
  A：BERT 风格 WordPiece 通常输出 `[UNK]`；也可以设计字符级 fallback 来减少 unknown。

```python
class SimpleWordPieceTokenizer:
    def __init__(self, vocab: set[str], unk_token: str = "[UNK]"):
        self.vocab = set(vocab)
        self.unk_token = unk_token

    def encode_word(self, word: str) -> list[str]:
        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            matched = None

            while end > start:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    matched = piece
                    break
                end -= 1

            if matched is None:
                return [self.unk_token]

            pieces.append(matched)
            start = end

        return pieces

    def encode(self, text: str) -> list[str]:
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens: list[str]) -> str:
        words = []
        current_word = ""

        for token in tokens:
            if token == self.unk_token:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(token)
            elif token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token

        if current_word:
            words.append(current_word)
        return " ".join(words)
```

### 110. SentencePiece

介绍：SentencePiece 不是单个分词算法，而是一套不依赖空格预分词的子词 tokenizer 方案，常见训练方式包括 Unigram LM 和 BPE。它把空格当成普通符号处理，常用 `▁` 表示词边界，因此适合中文、日文、多语言以及没有天然空格分隔的文本。

公式：$z^*=\arg\min_{z_1,\ldots,z_m}\sum_{i=1}^{m}-\log P(z_i)$。

- Q：SentencePiece 和 BPE / WordPiece 是什么关系？
  A：SentencePiece 是 tokenizer 框架和训练工具，可以训练 Unigram LM 或 BPE；BPE、WordPiece 更像具体子词构造方法。
- Q：`▁` 符号有什么作用？
  A：它表示原始文本中的空格或词边界，解码时可以据此恢复空格。
- Q：Unigram LM 编码时怎么选择切分？
  A：在所有可能的子词切分中选择概率最大的一条路径，等价于最小化负对数概率之和，通常用动态规划或 Viterbi。

```python
class SimpleSentencePieceUnigram:
    def __init__(self, piece_probs: dict[str, float], unk_token: str = "<unk>"):
        self.piece_probs = dict(piece_probs)
        self.unk_token = unk_token

    def _normalize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return "▁" + text.replace(" ", "▁")

    def encode(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        n = len(normalized)
        dp = [float("inf")] * (n + 1)
        prev = [None] * (n + 1)
        dp[0] = 0.0
        unk_prob = self.piece_probs.get(self.unk_token, 1e-6)

        for i in range(n):
            if dp[i] == float("inf"):
                continue

            for piece, prob in self.piece_probs.items():
                if piece == self.unk_token:
                    continue
                if normalized.startswith(piece, i):
                    j = i + len(piece)
                    cost = -math.log(prob + 1e-12)
                    if dp[i] + cost < dp[j]:
                        dp[j] = dp[i] + cost
                        prev[j] = (i, piece)

            unk_cost = -math.log(unk_prob + 1e-12)
            if dp[i] + unk_cost < dp[i + 1]:
                dp[i + 1] = dp[i] + unk_cost
                prev[i + 1] = (i, self.unk_token)

        tokens = []
        pos = n
        while pos > 0:
            item = prev[pos]
            if item is None:
                return [self.unk_token]
            prev_pos, piece = item
            tokens.append(piece)
            pos = prev_pos

        return tokens[::-1]

    def decode(self, tokens: list[str]) -> str:
        text = "".join(token for token in tokens if token != self.unk_token)
        return text.replace("▁", " ").strip()
```

### 111. BOS / EOS Token

介绍：BOS（Beginning of Sequence）表示序列开始，EOS（End of Sequence）表示序列结束。自回归语言模型训练时，常把 BOS 放在输入最前面，把 EOS 作为需要预测的最后一个 token；生成时通常从 BOS 或 prompt 开始，遇到 EOS 就停止。

公式：$\tilde{x}=[x_{\mathrm{BOS}},x_1,\ldots,x_n,x_{\mathrm{EOS}}]$；$L=-\sum_{t=1}^{n+1}\log p(\tilde{x}_t|\tilde{x}_{<t})$。

- Q：BOS 的作用是什么？
  A：给模型一个明确的序列起点，尤其在没有 prompt 或需要统一格式时很有用。
- Q：EOS 的作用是什么？
  A：告诉模型序列已经结束；生成时采到 EOS 通常就停止继续解码。
- Q：训练语言模型时 input 和 target 怎么错位？
  A：通常 input 是 `[BOS, x_1, ..., x_n]`，target 是 `[x_1, ..., x_n, EOS]`。

```python
def add_bos_eos(token_ids: list[int], bos_id: int, eos_id: int) -> list[int]:
    return [bos_id] + list(token_ids) + [eos_id]


def make_lm_input_target(token_ids: list[int], bos_id: int, eos_id: int) -> tuple[Tensor, Tensor]:
    sequence = torch.tensor(add_bos_eos(token_ids, bos_id, eos_id), dtype=torch.long)
    input_ids = sequence[:-1]
    target_ids = sequence[1:]
    return input_ids, target_ids


def stop_at_eos(token_ids: list[int], eos_id: int) -> list[int]:
    output = []
    for token_id in token_ids:
        if token_id == eos_id:
            break
        output.append(token_id)
    return output
```

## CNN 与卷积结构

### 112. Conv2d

介绍：2D 卷积对输入局部窗口和卷积核做逐元素相乘再求和。最直观的手写方式是先 padding，再用循环遍历输出位置，取出输入窗口和卷积核相乘求和。输出尺寸由 kernel、stride、padding 决定。

公式：$y_{b,o,i,j}=\sum_c\sum_u\sum_v x_{b,c,i+u,j+v}W_{o,c,u,v}+b_o$。

- Q：Conv2d 输出尺寸怎么算？
  A：$H_{\mathrm{out}}=\left\lfloor\frac{H+2P-K_h}{S}\right\rfloor+1$，宽度同理。
- Q：手写 Conv2d 的核心步骤是什么？
  A：先 padding，再遍历 batch、输出通道、输出高宽位置，取局部窗口和对应卷积核相乘求和。
- Q：卷积层和全连接层的主要区别是什么？
  A：卷积使用局部连接和权重共享，参数量更少，适合图像空间结构。

```python
def _pair(x):
    return (x, x) if isinstance(x, int) else x


def my_conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0) -> Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

    B, _, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride_h + 1
    W_out = (W - kW) // stride_w + 1
    out = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)

    for b in range(B):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    window = x[b, :, h_start : h_start + kH, w_start : w_start + kW]
                    out[b, oc, i, j] = (window * weight[oc]).sum()
            if bias is not None:
                out[b, oc] += bias[oc]
    return out
```

### 113. Pooling

介绍：Pooling 用固定窗口对局部区域做下采样，常见有 max pooling 和 average pooling。Max pooling 保留局部最强响应，average pooling 保留局部平均信息；二者都能降低空间分辨率、扩大后续特征的感受范围。

公式：$y_{i,j}=\max_{(u,v)\in\Omega_{i,j}}x_{u,v}$；$\bar{y}_{i,j}=\frac{1}{|\Omega|}\sum_{(u,v)\in\Omega_{i,j}}x_{u,v}$。

- Q：Max Pooling 和 Average Pooling 有什么区别？
  A：Max pooling 取窗口最大值，更强调显著响应；average pooling 取平均值，更平滑。
- Q：Pooling 有可学习参数吗？
  A：普通 pooling 没有可学习参数，只是固定规则的局部聚合。
- Q：Pooling 为什么能降低计算量？
  A：它会减小特征图的高和宽，后续卷积或全连接层处理的元素更少。

```python
def max_pool2d(x: Tensor, kernel_size: int, stride: int | None = None) -> Tensor:
    if stride is None:
        stride = kernel_size
    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    out = torch.zeros(B, C, H_out, W_out, device=x.device, dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            window = x[:, :, h_start : h_start + kernel_size, w_start : w_start + kernel_size]
            out[:, :, i, j] = window.amax(dim=(-1, -2))
    return out
```

### 114. Padding / Stride / Output Size

介绍：卷积输出尺寸由输入尺寸、kernel size、padding、stride 和 dilation 决定。普通卷积不考虑 dilation 时，输出高为 $\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1$；stride 越大输出越小，padding 可以保留边界信息并控制输出尺寸。

公式：$L_{\mathrm{out}}=\left\lfloor\frac{L_{\mathrm{in}}+2P-D(K-1)-1}{S}\right\rfloor+1$。

- Q：卷积输出尺寸公式是什么？
  A：$L_{\mathrm{out}}=\left\lfloor\frac{L_{\mathrm{in}}+2P-D(K-1)-1}{S}\right\rfloor+1$。
- Q：stride 的作用是什么？
  A：控制滑动窗口每次移动的步长，stride 越大，下采样越明显。
- Q：padding 的作用是什么？
  A：在边界补值，减少边缘信息损失，也常用于保持输出尺寸。

```python
def conv_output_size(size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

### 115. Receptive Field

介绍：感受野表示输出特征上的一个位置能看到输入中的多大区域。多层卷积会逐层扩大感受野；kernel 越大、stride 越大、dilation 越大，感受野增长越快。

公式：$r_l=r_{l-1}+(K_l-1)\prod_{i=1}^{l-1}S_i$。

- Q：感受野为什么重要？
  A：它决定某层特征能融合多大范围的上下文，影响模型捕获局部或全局模式的能力。
- Q：堆叠两个 `3x3` 卷积相当于多大感受野？
  A：stride 为 1、dilation 为 1 时，两个 `3x3` 的有效感受野是 `5x5`。
- Q：dilation 怎么影响感受野？
  A：dilation 会拉开 kernel 采样点，使有效 kernel 变为 $K_{\mathrm{eff}}=D(K-1)+1$。

```python
def receptive_field_1d(layers: list[tuple[int, int, int]]) -> int:
    rf = 1
    jump = 1
    for kernel_size, stride, dilation in layers:
        effective_kernel = dilation * (kernel_size - 1) + 1
        rf += (effective_kernel - 1) * jump
        jump *= stride
    return rf
```

### 116. Depthwise Separable Convolution

介绍：Depthwise separable convolution 把普通卷积分解成 depthwise 卷积和 pointwise `1x1` 卷积。Depthwise 每个输入通道单独卷积，pointwise 再做通道混合，能显著减少参数量和计算量。

公式：$N_{\mathrm{std}}=C_{\mathrm{in}}C_{\mathrm{out}}K^2$；$N_{\mathrm{sep}}=C_{\mathrm{in}}K^2+C_{\mathrm{in}}C_{\mathrm{out}}$。

- Q：Depthwise 和 Pointwise 分别做什么？
  A：Depthwise 负责每个通道的空间卷积，pointwise 负责跨通道线性组合。
- Q：它为什么比普通卷积省参数？
  A：普通卷积参数是 $C_{\mathrm{in}}\cdot C_{\mathrm{out}}\cdot K^2$，深度可分离卷积约是 $C_{\mathrm{in}}\cdot K^2+C_{\mathrm{in}}\cdot C_{\mathrm{out}}$。
- Q：它常见于哪些模型？
  A：MobileNet、轻量级 CNN 和移动端视觉模型。

```python
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))
```

### 117. Dilated Convolution

介绍：Dilated convolution 又叫空洞卷积，在卷积核采样点之间插入间隔。它不增加参数量，却能扩大有效感受野，常用于语义分割、语音建模和需要大上下文的卷积网络。

公式：$K_{\mathrm{eff}}=D(K-1)+1$。

- Q：dilation 为 2 的 `3x3` 卷积有效 kernel 是多大？
  A：有效尺寸是 $2\cdot(3-1)+1=5$。
- Q：空洞卷积增加参数量吗？
  A：不增加，卷积核参数仍是原来的 $K\cdot K$ 个采样点。
- Q：空洞卷积的主要风险是什么？
  A：dilation 过大可能产生网格效应，局部连续信息覆盖不足。

```python
def dilated_conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1) -> Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W - dil_w * (kW - 1) - 1) // stride_w + 1
    out = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)

    for b in range(B):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    value = torch.zeros((), device=x.device, dtype=x.dtype)
                    for ic in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):
                                h_idx = i * stride_h + kh * dil_h
                                w_idx = j * stride_w + kw * dil_w
                                value = value + x[b, ic, h_idx, w_idx] * weight[oc, ic, kh, kw]
                    if bias is not None:
                        value = value + bias[oc]
                    out[b, oc, i, j] = value
    return out
```

## 序列模型

### 118. RNN

介绍：RNN 用隐藏状态递归处理序列，当前状态由当前输入和上一时刻状态共同决定。简单 RNN 的核心公式是 $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$，适合表达顺序依赖，但长序列上容易梯度消失或爆炸。

公式：$h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$。

- Q：RNN 为什么能处理变长序列？
  A：同一组参数在每个时间步复用，按时间维循环处理输入。
- Q：RNN 的隐藏状态表示什么？
  A：它是到当前时间步为止的历史信息摘要。
- Q：简单 RNN 的主要问题是什么？
  A：长序列反向传播时容易梯度消失或爆炸，难以捕获长期依赖。

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, S, _ = x.shape
        if h is None:
            h = torch.zeros(B, self.W_h.out_features, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = torch.tanh(self.W_x(x[:, t]) + self.W_h(h))
            outputs.append(h)
        return torch.stack(outputs, dim=1), h
```

### 119. LSTM

介绍：LSTM 通过输入门、遗忘门、输出门和候选记忆控制信息流。它显式维护 cell state，能更稳定地保留长期信息，是解决简单 RNN 长期依赖问题的经典结构。

公式：$i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$；$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$；$c_t=f_t\odot c_{t-1}+i_t\odot g_t$；$h_t=o_t\odot\tanh(c_t)$。

- Q：LSTM 有哪些门？
  A：输入门、遗忘门、输出门，以及候选记忆分支。
- Q：cell state 的作用是什么？
  A：作为较稳定的信息通道，减少长期依赖中的梯度衰减。
- Q：遗忘门控制什么？
  A：控制上一时刻 cell state 中哪些信息被保留或丢弃。

```python
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_proj = nn.Linear(input_dim, 4 * hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        i, f, g, o = (self.x_proj(x) + self.h_proj(h)).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c
```

### 120. GRU

介绍：GRU 是 LSTM 的简化门控循环单元，使用更新门和重置门控制隐藏状态。它没有单独的 cell state，参数量通常少于 LSTM，训练和推理更轻。

公式：$z_t=\sigma(W_z[x_t,h_{t-1}])$；$r_t=\sigma(W_r[x_t,h_{t-1}])$；$h_t=(1-z_t)\odot\tilde{h}_t+z_t\odot h_{t-1}$。

- Q：GRU 有哪些门？
  A：更新门和重置门。
- Q：GRU 和 LSTM 的主要区别是什么？
  A：GRU 没有独立 cell state，门更少，结构更简洁。
- Q：更新门控制什么？
  A：控制保留旧隐藏状态和写入新候选状态的比例。

```python
class GRUCellFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.x_z = nn.Linear(input_dim, hidden_dim)
        self.h_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.x_r = nn.Linear(input_dim, hidden_dim)
        self.h_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.x_n = nn.Linear(input_dim, hidden_dim)
        self.h_n = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = torch.sigmoid(self.x_z(x) + self.h_z(h))
        r = torch.sigmoid(self.x_r(x) + self.h_r(h))
        n = torch.tanh(self.x_n(x) + self.h_n(r * h))
        return (1 - z) * n + z * h
```

## 模型优化

### 121. Language Modeling Objective

介绍：Language Modeling Objective 是语言模型训练目标。Causal LM 训练时用当前位置之前的 token 预测当前位置 token；训练阶段可以 teacher forcing 并行计算所有位置的 loss，推理阶段则按生成出的 token 自回归继续预测。

公式：$L_{\mathrm{LM}}=-\sum_{t=1}^{T}\log p_\theta(x_t|x_{<t})$。

- Q：Causal LM 的训练目标是什么？
  A：用历史 token 预测下一个 token，最大化真实序列的条件概率。
- Q：Teacher forcing 是什么？
  A：训练时每个位置都使用真实前缀作为条件，而不是使用模型自己前一步生成的 token。
- Q：训练时并行和推理时自回归矛盾吗？
  A：不矛盾。训练时用 causal mask 并行算所有位置，推理时真实未来 token 不存在，只能逐步生成。

```python
def causal_lm_loss(logits: Tensor, input_ids: Tensor) -> Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    vocab_size = shift_logits.shape[-1]
    return F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
```

### 122. Loss Mask / Ignore Index

介绍：Loss Mask 用来控制哪些 token 位置参与 loss。语言模型训练中常把 padding、prompt 中不需要监督的位置、packing 后的无效位置设置为 ignore index；它和 attention mask 不同，attention mask 控制可见性，loss mask 控制是否计入训练目标。

公式：$L=-\frac{\sum_{t=1}^{T}m_t\log p_\theta(y_t|x_{\le t})}{\sum_{t=1}^{T}m_t}$。

- Q：Loss mask 和 attention mask 有什么区别？
  A：attention mask 控制 token 能看见谁；loss mask 控制哪些位置参与损失计算。
- Q：`ignore_index=-100` 常用来做什么？
  A：让 PyTorch 的 Cross Entropy 跳过对应标签位置，不把它们计入 loss。
- Q：padding token 为什么不参与 loss？
  A：padding 是为了对齐 batch 的无效内容，让它参与 loss 会让模型学习无意义目标。

```python
def masked_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=ignore_index)
```

### 123. DPO Loss

介绍：DPO 用 chosen/rejected 偏好对直接优化 policy。它比较 policy 相对 reference 在 chosen 和 rejected 上的 logprob 差，并用 `logsigmoid` 构造 pairwise loss。

公式：$L=-\log\sigma\left(\beta[(\ell_\theta^+-\ell_\theta^-)-(\ell_0^+-\ell_0^-)]\right)$。

- Q：DPO 的 chosen/rejected 表示什么？
  A：chosen 是偏好数据中更好的回答，rejected 是较差回答。
- Q：reference model 在 DPO 中起什么作用？
  A：提供基准策略，限制 policy 不要过度偏离原模型。
- Q：`beta` 控制什么？
  A：控制偏好优化强度和相对 reference 的约束尺度。

```python
def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
) -> Tensor:
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

### 124. GRPO Loss

介绍：GRPO 在同一 prompt 的多条 response 内做 reward 标准化，得到组内相对 advantage，再用 $-A\log\pi_\theta(y|x)$ 优化策略。advantage 通常 stop-gradient。

公式：$A_i=\frac{r_i-\mu_G}{\sigma_G+\epsilon}$；$L=-A_i\log\pi_\theta(y_i|x)$。

- Q：GRPO 的 group 是什么？
  A：通常同一个 prompt 下采样出的多条 response 属于同一 group。
- Q：为什么要做组内 reward 标准化？
  A：得到相对 advantage，减少不同 prompt 奖励尺度差异的影响。
- Q：为什么 advantage 要 detach？
  A：策略梯度只更新 logprob 对应的 policy，advantage 当作权重而不是反向传播目标。

```python
def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor, eps: float = 1e-5) -> Tensor:
    advantages = torch.empty_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - mean) / (std + eps)
    return -(advantages.detach() * logps).mean()
```

### 125. PPO Loss

介绍：PPO 使用新旧策略 logprob 的比率 $\exp(\ell_\theta-\ell_{\theta_{\mathrm{old}}})$ 构造 surrogate objective，并通过 clipping 限制单步策略更新幅度。

公式：$r_t=\exp(\ell_\theta-\ell_{\theta_{\mathrm{old}}})$；$L=-\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)$。

- Q：PPO 的 ratio 表示什么？
  A：$\exp(\ell_\theta-\ell_{\theta_{\mathrm{old}}})$，表示新旧策略对同一动作概率的比值。
- Q：clip 的作用是什么？
  A：限制单次策略更新幅度，避免新策略相对旧策略变化过大。
- Q：为什么 old logps 和 advantages 要 detach？
  A：PPO loss 只应该对当前 policy 的 logprob 求梯度。

```python
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor, clip_ratio: float = 0.2) -> Tensor:
    ratios = torch.exp(new_logps - old_logps.detach())
    adv = advantages.detach()
    unclipped = ratios * adv
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    return -torch.min(unclipped, clipped).mean()
```

## 训练机制与工程技巧

### 126. Backpropagation

介绍：Backpropagation 用链式法则从 loss 反向计算每个参数的梯度。深度学习训练的核心就是前向得到 loss，反向得到梯度，再由优化器更新参数。

公式：$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}$。

- Q：反向传播的数学核心是什么？
  A：链式法则，把输出对中间变量、参数的梯度逐层相乘传回去。
- Q：为什么需要保存前向中间结果？
  A：很多梯度计算依赖前向的输入、激活或 mask，需要缓存用于 backward。
- Q：反向传播和优化器是什么关系？
  A：反向传播负责算梯度，优化器根据梯度决定怎么更新参数。

```python
def linear_regression_backward(x: Tensor, y: Tensor, w: Tensor, b: Tensor):
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()
    grad_pred = 2.0 * (pred - y) / y.numel()
    grad_w = x.T @ grad_pred
    grad_b = grad_pred.sum(dim=0)
    return loss, grad_w, grad_b
```

### 127. Autograd / Computational Graph

介绍：Autograd 会在前向计算时动态构建计算图，每个可求导 Tensor 记录产生它的操作。调用 `backward()` 后，PyTorch 沿计算图反向应用链式法则，把梯度累加到叶子 Tensor 的 `.grad`。

公式：$\nabla_\theta L=\frac{\partial L}{\partial z}\frac{\partial z}{\partial\theta}$。

- Q：什么是叶子 Tensor？
  A：通常是用户创建且 `requires_grad=True` 的 Tensor，例如模型参数。
- Q：为什么梯度会累加？
  A：PyTorch 默认把新梯度加到已有 `.grad` 上，所以每步训练前要 `zero_grad()`。
- Q：`detach()` 有什么作用？
  A：返回一个不再连接当前计算图的 Tensor，阻断梯度继续回传。

```python
def autograd_example(x: Tensor) -> Tensor:
    w = torch.ones(x.shape[-1], 1, device=x.device, requires_grad=True)
    y = (x @ w).pow(2).mean()
    y.backward()
    return w.grad
```

### 128. Mixed Precision

介绍：Mixed Precision 混合精度训练把部分计算放到低精度中执行，以降低显存占用并提高吞吐，同时保留关键参数、归约或优化器状态的稳定性。它通常和自动类型转换、梯度缩放、FP32 master weight 等机制一起使用。

公式：$L'=cL$；$g=g'/c$。

- Q：混合精度主要节省什么？
  A：节省激活、梯度和部分计算的显存，并提升 Tensor Core 等硬件上的吞吐。
- Q：为什么混合精度不是所有东西都用低精度？
  A：有些归约、归一化、优化器状态或主权重需要更高精度来避免数值误差累积。
- Q：梯度缩放解决什么问题？
  A：把 loss 放大后反向传播，减少小梯度在低精度中下溢；更新前再把梯度缩回原尺度。

```python
def mixed_precision_train_step(model: nn.Module, x: Tensor, y: Tensor, optimizer, scaler=None) -> Tensor:
    device_type = "cuda" if x.is_cuda else "cpu"
    enabled = x.is_cuda
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, enabled=enabled):
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()
```

### 129. FP16

介绍：FP16 是 16 位浮点格式，通常包含 1 位符号位、5 位指数位和 10 位尾数位。它比 FP32 更省显存、计算更快，但指数范围较小，训练时更容易出现上溢或下溢，因此常和 loss scaling 搭配。

公式：$x=(-1)^s(1+f)2^{e-15}$。

- Q：FP16 为什么容易下溢？
  A：FP16 指数位少，能表示的极小正数范围有限，小梯度可能被舍入成 0。
- Q：FP16 为什么常需要 loss scaling？
  A：loss scaling 先放大 loss 和梯度，减少小梯度下溢；参数更新前再把梯度除回原尺度。
- Q：FP16 常见风险是什么？
  A：梯度下溢、激活或 loss 上溢、某些归约操作精度不足。

```python
def to_fp16_then_back(x: Tensor) -> Tensor:
    return x.to(torch.float16).to(torch.float32)


def unscale_gradients(parameters, scale: float) -> None:
    for p in parameters:
        if p.grad is not None:
            p.grad.div_(scale)
```

### 130. BF16

介绍：BF16 是 16 位浮点格式，通常包含 1 位符号位、8 位指数位和 7 位尾数位。它的指数范围接近 FP32，因此比 FP16 更不容易上溢或下溢；缺点是尾数位更少，单个数的精度更粗。

公式：$x=(-1)^s(1+f)2^{e-127}$。

- Q：BF16 相比 FP16 最大优势是什么？
  A：BF16 指数位更多，动态范围接近 FP32，更不容易出现上溢或下溢。
- Q：BF16 的缺点是什么？
  A：尾数位比 FP16 更少，数值精度更粗，细粒度小数表达能力更弱。
- Q：BF16 还需要 loss scaling 吗？
  A：通常不需要或需求弱很多，因为它的动态范围足够大；具体仍取决于硬件和训练稳定性。

```python
def to_bf16_then_back(x: Tensor) -> Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def bf16_matmul(a: Tensor, b: Tensor) -> Tensor:
    out = a.to(torch.bfloat16) @ b.to(torch.bfloat16)
    return out.to(torch.float32)
```

### 131. Gradient Checkpointing

介绍：Gradient checkpointing 用计算换显存。前向时不保存所有中间激活，只保存少量 checkpoint；反向时重新计算部分前向结果，从而降低显存占用。

- Q：Gradient checkpointing 节省什么？
  A：主要节省训练时保存激活的显存。
- Q：它的代价是什么？
  A：反向传播需要重算部分前向，训练时间会增加。
- Q：它适合什么场景？
  A：大模型、长序列、显存瓶颈明显且能接受额外计算的训练场景。

```python
def checkpointed_forward(module: nn.Module, x: Tensor) -> Tensor:
    from torch.utils.checkpoint import checkpoint

    return checkpoint(module, x, use_reentrant=False)
```

### 132. Gradient Accumulation

介绍：Gradient accumulation 把大 batch 拆成多个 micro-batch。每个 micro-batch 的 loss 除以累积步数后 backward，最后统一 `optimizer.step()`。

公式：$g=\frac{1}{K}\sum_{k=1}^{K}g_k$。

- Q：Gradient accumulation 为什么要把 loss 除以累积步数？
  A：为了让累积梯度尺度等价于一次大 batch 的平均 loss 梯度。
- Q：它和真实大 batch 完全等价吗？
  A：对普通层接近等价，但 BatchNorm、Dropout、数据并行同步等会带来差异。
- Q：什么时候调用 `optimizer.step()`？
  A：所有 micro-batch 都 backward 完成后，再调用一次 `optimizer.step()`。

```python
def accumulated_step(model: nn.Module, optimizer, loss_fn, micro_batches) -> float:
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y)
        total_loss += loss.item()
        (loss / n).backward()
    optimizer.step()
    return total_loss / n
```
