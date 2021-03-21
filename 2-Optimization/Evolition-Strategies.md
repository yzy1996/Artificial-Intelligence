<h1 align="center">Evolutionary Strategies (ES)</h1>
<div align="center">

</div>



Evolutionary Strategies (ES) are a popular family of **black-box** zeroth-order optimization algorithms which rely on search distributions to efficiently optimize a large variety of objective functions.

Examples of black-box optimization methods include [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing), [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) and [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder–Mead_method).



**Evolution Strategies (ES)** is one type of black-box optimization algorithms, born in the family of **Evolutionary Algorithms (EA)**





balance between exploitation and exploration of the optimization landscape



Bayesian Optimization 



ES rely on a search distribution



**Algorithm 1**  Generic ES procedure

**Input:** objective $f$, distribution $\pi_0$, population size $n$

**Repeat:** (Sampling) Sample $x_1, \dots, x_n \stackrel{\mathrm{i.i.d}}{\sim} \pi_{t}$

​				(Evaluation) Evaluate $f(x_1), \dots, f(x_n)$

​				(Update) Update $\pi_t$ to produce $x$ of potentially smaller objective values.

**Until:** convergence



## 进化策略的概述





## 进化策略的基本思想

我们不严格要求最优解 $x^*$ 是一个点，而是认为它是一个小区域，在这个小区域内采样得到的所有解都可以近似被看成最优解，因此**目的就是找到这个最优小区域**。肯定不可能一下子就找到这个最优区域，会通过初始化一个随机区域后，通过一步步迭代筛选调整，进而优化找到最优区域。

每一次迭代的区域我们称之为**搜索空间**，一般我们将它建模为一个正态分布，参数记作 $\theta = (\mu, \sigma, C)$ ，解服从这个正态分布记作 $x \sim p_{\theta}(x) = \mathcal{N}\left(\mu, \sigma^{2} C\right)$ 。这三个参数分别影响了：                                          

- $\mu$ 均值，决定了分布的中心位置，也即搜索空间的位置
- $\sigma$ 步长，决定了分布的整体方差，也即搜索空间的大小
- $C$ 协方差矩阵，决定了分布的形状，也即搜索空间不同维度的相对关系



各种ES算法的核心是**如何调整这些参数，使得产生好解的概率逐渐增大（沿好的搜索方向进行搜索的概率增大）**。



一般步骤可以被概括为：

1. **初始化** 给定目标函数，随机一个分布参数，设定种群大小等
2. **迭代** 
   - **采样** 从分布里采集一组解
   - **评估** 计算解的目标函数值
   - **选择** 根据目标函数值选择部分或全部解
   - **更新** 使用选择的解更新分布参数
3. **结束** 直到收敛停止



## 进化策略的分类

根据产生解和选择解的方式不同，主要分为三类

### (1+1)-ES

每次迭代只产生一个新解，通过和父代比较，较好的一个成为下一次迭代的父代

> 形式简单，更易于理论分析；
>
> 性能良好，某些variants代表了state-of-the-art；
>
> 集中在局部搜索(local search)；



### ($\mu + \lambda$)-ES

每次迭代产生 $\lambda$ 个新解，通过和父代比较，较好的 $\mu$ 个成为下一次迭代的父代

> 引入种群的思想，易于并行化；
>
> 围绕着最优点进行搜索，可能会长时间陷入某个局部范围无法出来；
>
> 主要用在多目标优化里面（MO-CMA-ES）；



### ($\mu , \lambda$)-ES

每次迭代产生 $\lambda$ 个新解，不和父代比较，较好的 $\mu$ 个成为下一次迭代的父代

> 所有解都只存活一代，避免长时间陷入某个范围；
>
> 每次只保留产生的最好的解，这种常用于理论分析。



前两种都是再最优解附近进行搜索，因此被称为精英进化策略(Elitist ES)；而第三种不在最优解附近搜索，因此被称为非精英进化策略(Non-Elitist ES)



目前最常用是引入multi-recombination构成 $(\mu/\mu_I, \lambda)$-ES，这样的好处是多个优解之间的组合会提取其共有的特征而舍去某个解所独有的特征，能避免陷入某个区域持续搜索。

## 进化策略的例子

根据定义，解是从一个正态分布中采样产生的，$x \sim \mathcal{N}\left(\mu, \sigma^{2} C\right)$ ，但在实际使用时，我们常常将它拆分为
$$
x = \mu + \sigma y, y \sim \mathcal{N}\left(0, C\right)
$$
其中 $y$ 可以被理解为是一个搜索方向

> 回顾高斯分布（多维高斯分布）



### 简单高斯进化策略

我们假设 $C = I$ ，也即各向同性。

**迭代过程如下**



1. 采样产生 $\Lambda$ 个子代：
   $$
   D^{t+1} = \{x_i^{t+1} = \mu^t + \sigma ^t y_i^t, \text{where} \ y_i^t \sim \mathcal{N}\left(0, I\right), i = 1, \dots, \Lambda \}
   $$

2. 对 $\Lambda$ 个子代精英排序：
   $$
   f(x_{1:\Lambda}^{t+1}) \le f(x_{2:\Lambda}^{t+1}) \le \cdots \le f(x_{\lambda:\Lambda}^{t+1})
   $$

3. 选择前 $\lambda$ 个精英子代，$x_{i:\lambda}$ 表示在 $\lambda$ 中排第 $i$ 个：
   $$
   D_{elite}^{t+1} = \{x_{i:\lambda}^{t+1}, i = 1, \dots, \lambda\}
   $$

5. 更新参数，需要注意的是更新 $\sigma$ 时用的还是上一代的 $\mu$ ：
   $$
   \mu^{t+1} = avg(D_{elite}^{t+1}) = \frac{1}{\lambda} \sum_{i=1}^{\lambda}x_i^{t+1}
   \\
   {\sigma^2}^{t+1} = var(D_{elite}^{t+1}) =  \frac{1}{\lambda} \sum_{i=1}^{\lambda}(x_i^{t+1} - \mu^t)^2
   $$

**代码**

**结果**





### CMA-ES

迭代框架是一致的，不同点在于需要更新C



首先补充点知识

1. 矩阵的特征分解
   $$
   C = BD^2B^T
   \\
   C^{\frac{1}{2}} = BDB^T
   \\
   C^{-\frac{1}{2}} = BD^{-1}B^T
   $$
   



#### 问题铺垫

我们考虑一个**黑箱优化**场景，想要最小化一个目标函数
$$
\begin{aligned}
f: & \mathbb{R}^{n} \rightarrow \mathbb{R} \\
& \boldsymbol{x} \mapsto f(\boldsymbol{x})
\end{aligned}
$$
目标是找到一个或多个搜索点（候选解），$x \in \mathbb{R}^{n}$，他们有尽可能小的函数值 $f(x)$。



#### 更新 $\mu, \sigma$

前面我们说过y是一个搜索方向，那么一次迭代中所有个体的搜索方向的和可以被表示为一条进化路径

将 式(2) 代入 式(5)，得：
$$
\begin{aligned}
\mu^{t+1} 
&= \frac{1}{\lambda} \sum_{i=1}^{\lambda}x_i^{t+1}\\
&= \frac{1}{\lambda} \sum_{i=1}^{\lambda}(\mu^t + \sigma ^t y_i^t)\\
&= \mu^t + \frac{\sigma^t}{\lambda}\sum_{i=1}^{\lambda}y_i^t
\end{aligned}
$$

因此可以得到：
$$
\mu^{t+1} - \mu^t = \sigma^t \frac{1}{\lambda}\sum_{i=1}^{\lambda}y_i^t
$$

这个式子描述了分布均值的移动，一次迭代中每个个体移动的叠加，构成了均值的一步移动，$\sigma$ 是步长。

又因为 $y_i^t \sim \mathcal{N}\left(0, C^t\right)$ ，所以

$$
\frac{\mu^{t+1} - \mu^t}{\sigma^t} \sim \frac{1}{\lambda}\mathcal{N}\left(0, \lambda C^t\right) = \frac{1}{\sqrt{\lambda}}\sqrt{C^t}\mathcal{N}\left(0, I\right)
$$

$$
\sqrt{\frac{\lambda}{C^t}} \frac{\mu^{t+1} - \mu^t}{\sigma^t} \sim \mathcal{N}\left(0, I\right)
$$

我们定义一条进化路径 $p_{\sigma}$ 
$$
\begin{align}
p_{\sigma}^{t+1} 
&= (1 - \alpha_{\sigma})p_{\sigma}^t + \sqrt{1-(1 - \alpha_{\sigma})^2}\sqrt{\frac{\lambda}{C^t}} \frac{\mu^{t+1} - \mu^t}{\sigma^t}\\
&= (1 - \alpha_{\sigma})p_{\sigma}^t + \sqrt{\frac{\alpha_{\sigma}(2 - \alpha_{\sigma})\lambda}{C^t}} \frac{\mu^{t+1} - \mu^t}{\sigma^t}
\end{align}
$$

> 为什么式子的系数是平方和为1？

我们希望 $p_{\sigma}$ 的长度是 $\mathbb{E}\|\mathcal{N}(0, I)\|$，当长度大于它的时候，我们就应该增加 $\sigma$ ，反之亦然。
$$
\sigma^{t+1} = \sigma^{t} \exp\left(\frac{\alpha_{\sigma}}{d_{\sigma}}\left(\frac{\left\|p_{\sigma}^{(t+1)}\right\|}{\mathbb{E}\|\mathcal{N}(0, I)\|}-1\right)\right)
$$
其中 $d_{\sigma} \approx 1$ 是阻尼参数，限制 $\sigma$ 的更新尺度



#### 更新C

前面写过 $y \sim \mathcal{N}\left(0, C\right)$，因此 $y$ 和 $C$ 存在关系 
$$
C^{t+1} = \frac{1}{\lambda} \sum_{i=1}^{\lambda} y_i^{t+1} {y_i^{t+1}}^T
$$
但要想通过 $y_i$ 来估计 $C$ 需要 $\lambda$ 足够大，我们试图找到一个方法能够在有限种群数量的前提下更新 $C$. 我们考虑两项：

- rank-min($\lambda, n$) update

  更新 $C^{t+1}$ 的时候还考虑 $C^{t}$ 这一历史信息的影响，于是
  $$
  C^{t+1} = (1 - \alpha_{c \lambda}) C^t + \alpha_{c \lambda} \frac{1}{\lambda} \sum_{i=1}^{\lambda} y_i^{t+1} {y_i^{t+1}}^T
  $$
   其中 $\alpha_{c \lambda} \approx \min \left(1, \lambda / n^{2}\right)$ ，$n$ 是解空间的维度

- rank-1 update

  因为 $(-y)(-y)^T = yy^T$ 导致平方后失去了符号信息，所以我们需要补偿这一符号信息，
  $$
  \begin{aligned}
  p_{c}^{t+1} 
  &= (1 - \alpha_{cp})p_{c}^t + \sqrt{1-(1 - \alpha_{cp})^2}\sqrt{\lambda} \frac{\mu^{t+1} - \mu^t}{\sigma^t}\\
  &= (1 - \alpha_{cp})p_{c}^t + \sqrt{\alpha_{cp}(2 - \alpha_{cp})\lambda} \frac{\mu^{t+1} - \mu^t}{\sigma^t}
  \end{aligned}
  $$

综合这两项，我们得到了 $C$ 的更新公式
$$
C^{t+1} = (1 - \alpha_{c\lambda} - \alpha_{c1}) C^t + \alpha_{c1}p_{c}^{t+1}{p_{c}^{t+1}}^T + \alpha_{c \lambda} \frac{1}{\lambda} \sum_{i=1}^{\lambda} y_i^{t+1} {y_i^{t+1}}^T
$$



## 参考

**文字部分整理自**

- [知乎-五楼whearer-进化策略-简介1](https://zhuanlan.zhihu.com/p/31028329)
- [博客-Lilian-Evolution Strategies](https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies)



**推荐文献**

原版ES

- 

引入ES到强化学习

- Evolution Strategies as a Scalable Alternative to Reinforcement Learning [2017] [OpenAI]