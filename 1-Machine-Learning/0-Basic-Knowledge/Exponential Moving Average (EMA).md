# Exponential Moving Average (EMA)

相关论文

- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)  
  *Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson*  
  **[`UAI 2018`] (`Cornell`)**



**目的**：使用EMA对模型的参数做平均，可以提高结果并增强鲁棒性。

**本质**：是一种给予近期数据更高权重的平均方法。



## Detail

举个例子，对于$n$个数据：$[x_1, x_2, \dots, x_n]$

普通的平均值为：$\bar{x}_n = \frac{1}{n} \sum_{i=1}^{n} x_i$

而用EMA表示为：$\hat{x}_n = \beta \cdot \bar{x}_{n-1}+(1-\beta) \cdot x_{n}$ 其中 $\bar{x}_{n-1}$ 表示前 $(n-1)$ 个数据的平均值，$\beta$ 是一个权重（一般为0.9-0.999）



迁移到深度学习中，数据 $x$ 就变成了模型在t时刻的网络 weight $\theta_t$
$$
\begin{aligned}
&\text{一般的： } \theta_{n}=\theta_{1}-\sum_{i=1}^{n-1} g_{i} \\
&\text{EMA： }\theta_{n}^{\prime}=\theta_{1}-\sum_{i=1}^{n-1}\left(1-\alpha^{n-i}\right) g_{i}
\end{aligned}
$$

> 相当于在每一步梯度下降的时候都乘以一个系数 $1-\alpha^{n-i}$，这就是一个 learning rate decay。因此在pytroch的官方代码中，这部分是在[optimizer](https://pytorch.org/docs/stable/optim.html)一章节介绍的。



## Code

```python
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
```



