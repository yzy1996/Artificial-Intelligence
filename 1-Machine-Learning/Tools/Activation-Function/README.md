[toc]

# Activation Function

先上推荐链接：

[Visualising Activation Functions in Neural Networks](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/) 可以动态展示各种激活函数及导数的图像

http://spytensor.com/index.php/archives/23/ 中文版

[Wiki-Activation_function](https://en.wikipedia.org/wiki/Activation_function) 归纳对比了很多激活函数



## 为什么需要激活函数

> **The role of activation functions?**

follow a linear transformation，`nn.Linear()`线性变换$w^Tx+b$可训练的只有权重`wight`和偏差`bias`

不用激活函数时，网络只是一个全连接的线性模型，`问卷积层呢？` 而线性模型的能力有限，无法解决复杂任务，因此很多情况下需要网络模型具有非线性，而激活函数就带来了非线性。

为什么要叫**激活**函数呢，激活的含义是当神经元信号达到一定程度时才会激发有效。否则如果一直都是激活状态，训练的消耗就会很大。这其实是和生物学真实神经元是相似的。



## 反向传播中激活函数的具体作用

> 为了弄清楚，激活函数的梯度是怎么作用的

举一个很简单的例子
$$
z = w^\mathsf{T}x + b \Rightarrow p = \sigma(z) \Rightarrow L(p, y)
$$
反向传播中更新权重 $w$ 的步骤是：
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}
$$
其中第一项是损失函数的梯度，第二项是激活函数的梯度，第三项是线性函数的梯度



在这里可以看出，替换不同的激活函数会直接影响整个反向传播过程。



## 有哪些激活函数



### ReLU

> 先从最有名的ReLU开始

ReLU[^ReLU]全称是**Rectified Linear Unit**，翻译过来是**修正线性单元**，他的函数表达式是：
$$
f(x) = \max(x, 0) =
\left\{
\begin{array}{ll}
x & \text { if } x \geq 0 \\
0 & \text { if } x<0
\end{array}\right.
$$

> **为什么ReLU好？**

gradients are able to flow when the input to the ReLU function is positive

- 计算简单。无论是原函数还是导数运算都很简单，相比于sigmoid tanh
- 非饱和。sigmoid在两端导数趋近于0，也叫饱和区间，会出现梯度消失
- 稀疏性。因为负数部分的输出为0，稀疏性带来的好处是减少了参数的相互依存关系，缓解了过拟合



因为ReLU在小于0的部分梯度恒为0，导致了部分神经元死亡，也被称为Dying ReLU现象，为了缓解这个问题，后续提出了不少ReLU的改进版。可能针对不同任务有不同的表现，但大多数有得有失，不如原始版ReLU。



- Leaky ReLU (LReLU)[^LReLU]
  $$
  f(x)=
  \left\{
  \begin{array}{ll}
  x & \text { if } x \geq 0 \\
  \alpha x & \text { if } x<0
  \end{array}\right. \quad \text{where} \ \alpha = 0.01
  $$

- Parametric ReLU (PReLU)[^PReLU] 

  $$
  f(x)=
  \left\{
  \begin{array}{ll}
  x & \text { if } x \geq 0 \\
  \alpha x & \text { if } x<0
  \end{array}\right. \quad \text{where} \ \alpha \ \text{is a learnable parameter}
  $$
  
- Softplus[^Softplus] (smooth version of ReLU)
  $$
  f(x) = \log(1+\exp(x))
  $$

- Exponential Linear Unit (ELU)[^ELU]
  $$
  f(x)=
  \left\{
  \begin{array}{ll}
  x & \text { if } x \geq 0 \\
  \alpha (\exp(x) - 1) & \text { if } x<0
  \end{array}\right. \quad \text{where} \ \alpha = 1
  $$

- Scaled Exponential Linear Unit (SELU)[^SELU] 
  $$
  f(x)=\lambda
  \left\{
  \begin{array}{ll}
  x & \text { if } x \geq 0 \\
  \alpha (\exp(x) - 1) & \text { if } x<0
  \end{array}\right. \quad \text{where} \ \alpha \approx 1.6733, \lambda \approx 1.0507
  $$

  
  
  

### 其他

谷歌的 Searching for Activation Functions 这篇文章基于RL对激活函数进行了研究，分为了一元和二元函数两大类



Unary functions:

$$
\begin{array}{l}
x,-x,|x|, x^{2}, x^{3}, \sqrt{x}, \beta x, x+\beta, \log (|x|+\epsilon), \exp (x) \sin (x), \cos (x) \\
\sinh (x), \cosh (x), \tanh (x), \sinh ^{-1}(x), \tan ^{-1}(x), \operatorname{sinc}(x), \max (x, 0), \min (x, 0), \sigma(x) \\
\log (1+\exp (x)), \exp \left(-x^{2}\right), \operatorname{erf}(x), \beta
\end{array}
$$
Binary functions:

$$
\begin{array}{l}
x_{1}+x_{2}, x_{1} \cdot x_{2}, x_{1}-x_{2}, \frac{x_{1}}{x_{2}+\epsilon}, \max \left(x_{1}, x_{2}\right), \min \left(x_{1}, x_{2}\right), \sigma\left(x_{1}\right) \cdot x_{2} \\
\exp \left(-\beta\left(x_{1}-x_{2}\right)^{2}\right), \exp \left(-\beta\left|x_{1}-x_{2}\right|\right), \beta x_{1}+(1-\beta) x_{2}
\end{array}
$$


再单独拎出来谈几个

Sigmoid
$$
f(x) = \frac{1}{1 + e^{-x}}
$$
tanh
$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$


Gaussian Error Linear Unit (GELU)[^GELU]
$$
f(x) = x \cdot \Phi(x)
$$

Swish: 

$$
f(x)=x \cdot \operatorname{sigmoid}(\beta x)
$$





### SIREN

> 这里专门讲一下sine正弦激活函数，写这篇文章的起因是SIREN这篇文章

相关文献有：

Taming the waves: sine as activation function in deep neural networks

Implicit Neural Representations with Periodic Activation Functions



SIREN的初衷是为了应用到implicit neural representation上，而传统使用的ReLU-based MLP因为激活函数缺失了二阶信息，所以无法表征细节信息。那么就要找有二阶导数的激活函数---比如tanh & softplus

<img src="C:\Users\zhiyuyang4\AppData\Roaming\Typora\typora-user-images\image-20210210110750822.png" alt="image-20210210110750822" style="zoom:25%;" />

作者通过实验证明这两个的效果不够好，于是引入了周期性的激活函数

> **sine激活函数具有哪些特性呢？**

让每一层输出的分布一致，保证了稳定性

BN也是为了保证每一层的数据分布保持相同



SIREN的导数还是一个SIREN，这种能够继承的特性可以让



初始化是为了控制激活函数的分布

保证在初始化的时候最后一层输出不依赖网络的层数，层数多了容易梯度消失



不好的初始化会导致糟糕的结果

不回遭受梯度消失和爆炸



梯度幅度缓慢的增加

## 我们需要关注激活函数的哪些特性

> 针对每个特性这里只列举几个例子



**取值范围**

> 有限范围让训练更稳定；无限范围让训练更快

sigmoid 取值范围为  $(0, 1)$

tanh 取值范围为  $(-1, 1)$

**最大梯度**

> 影响训练过程中梯度消失（梯度会小于1）、梯度爆炸（梯度会大于1）

sigmoid 最大梯度为0.25，在0处取得

tanh 最大梯度为1，在0处取得

**单调性** monotonic

> 如果单调会收敛较快，因为会一直朝着一个方向去

**原点对称性**



**周期性**



## 一个好的激活函数具有哪些特性

参考[知乎回答](https://www.zhihu.com/question/67366051)

1. 非线性
2. 几乎处处可微，对于SGD算法来说，由于几乎不可能收敛到梯度接近零的位置，有限的不可微点对于优化结果不会有很大影响[1]。
3. 计算简单
4. 非饱和性
5. 单调性，存在争议性，为了保证导数符号不变，更容易收敛
6. 输出范围有限，保证对于一些较大的输入也会有比较稳定的输出
7. 接近恒等变换，即约等于x，为了保证输出的幅度不回随着深度的增加而显著增加，从而使网络更加稳定。但这一点又和非线性是冲突的，因此激活函数是部分满足这个条件，比如tanh在远点附近有线性区，
8. 参数少，大部分激活函数都没有参数，因为激活函数的额外参数会增加网络的大小，
9. 归一化，对应激活函数是SELU，是样本分布自动归一化到零均值，单位方差的分布





## 文献

[^ReLU]: Rectified Linear Units Improve Restricted Boltzmann Machines
[^LReLU]: Rectifier nonlinearities improve neural network acoustic models 2013
[^PReLU]: Delving deep into rectifiers: Surpassing humanlevel performance on imagenet classification

[^Softplus]: Rectified linear units improve restricted boltzmann machines
[^ELU]: Fast and accurate deep network learning by exponential linear units (elus)
[^SELU]:Self-normalizing neural networks
[^GELU]: Bridging nonlinearities and stochastic regularizers with gaussian error linear units





---

second-order derivatives