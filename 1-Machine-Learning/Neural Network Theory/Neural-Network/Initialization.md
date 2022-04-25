> PyTorch 提供了很多预设好的 init 方法，详见：[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html)



## Introduction

最简单的就是(0,1)-均匀分布采样或者(0, 1)-正态分布采样了。

后来又新加入了 **Xavier**，**Kaiming** 两种经典方法：

- [Understanding the difficulty of training deep feedforward neural network](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)  
  *Xavier Glorot, Yoshua Bengio*  
  **[`AISTATS 2010`] (`UMontreal`)** citation=13951

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)  
  *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*  
  **[`ICCV 2015`] (`Microsoft`)** citation=13438

> Kaiming是针对Xavier初始化方法在relu这一类激活函数表现不佳而提出的改进。
>
> 两者均有均匀分布和正态分布两种形式。



### Notes

初始化的原理基本都是从 “方差一致性” 出发：

对于一个全连接层 $y_i = W_j^i x^j + b^i$，假设这些参数都是独立且零均值的。方差计算可写作 $\text{Var}(y^i) = d_j \text{Var}(W_j^i)\text{Var}(x^j)$，我们希望输入和输出的方差是相同的即 $\text{Var}(y^i) = \text{Var}(x^j)$，所以网络权重的方差应满足 $\text{Var}(W_j^i) = \frac{1}{d_j}$



经常看到 `fan_in` & `fan_out` 是什么意思？

> fan_in是某一层的 输入神经元个数，fan_out是输出神经元个数。
>
> 对于全连接层：fan_in=in_channel ；fan_out = out_channel
>
> 对于卷积层：fan_in = in_channel * kernel_height * kernel_width
>
> ​                       fan_out = out_channel* kernel_height * kernel_width
>
> fan_in使正向传播时，方差一致; fan_out使反向传播时，方差一致。



### Usage

```python
import torch.nn as nn

w = torch.empty(3, 5)
nn.init.uniform_(w)
```



```
model = SimpleCnn()
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # nn.init.normal_(m.weight.data)
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()


m.weight

m.bias
```









## Hypernetwork Initialization

随着超参数网络(Hypernetworks)的兴起，如何初始化它的参数也是一个值得研究的问题，最简单的做法就是也用上面或者其他基础初始化方法。下面论文研究了如何更好地初始化。

- [Principled Weight Initialization for Hypernetworks](https://openreview.net/pdf?id=H1lma24tPB)  
  *Oscar Chang, Lampros Flokas, Hod Lipson*  
  **[`ICLR 2020`] (`Columbia`)**

> 大致意思是之前的方法会导致主网络权重尺度不合适，进而导致训练不稳定。作者利用方差分析，来保证了主成分的统一。



代码在 [hypnettorch](https://github.com/chrhenning/hypnettorch)



