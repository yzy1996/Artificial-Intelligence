# GMM高斯混合模型

> Gaussian Mixed Model



## 问题引入

看图1，我们很容易用一个高斯分布函数就能拟合，但是看图2，我们就很难用一个高斯分布函数去拟合了，但又发现数据可以用两个高斯分布函数分开去拟合，所以这里就引入了**GMM高斯混合模型---多个高斯模型的线性组合**。

不同于单高斯模型的参数估计，是通过观测数据 $\boldsymbol{x}$ 估计参数 $(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。高斯混合模型不仅要估计每一个模型分量的参数，还要估计观测数据属于哪一个分量。所以我们的目的是要通过观测数据 $\boldsymbol{x}$ 估计参数 $(\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。



## 模型构建

（已知）有 $n$ 个观测数据：$\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n$

（已知）有 $K$ 个同分布不同参数的模型：$k=1,2,\dots,K$ 

（未知）属于第 $k$ 个模型分量的概率：$\pi_k$，满足条件：$0 \leq \pi_{k} \leq 1$ , $\sum_{k=1}^{K} \pi_{k}=1$ 

（未知）第 $k$ 个模型分量的高斯分布参数：$\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}$



前面两个未知参数都是针对全局而言的，是**先验信息**，而我们现在所拥有的是局部个体数据，是**后验信息**，因此需要额外引入一些符号变量来表示：

（未知）观测数据 $\boldsymbol{x}$ 属于第 $k$ 个模型分量：记作 $\boldsymbol{z} \in R^K$，采用“1-of-K”表示方法，其中只有一个分量 $z_k=1$ ，其余分量都等于0，满足条件：$z_{k} \in\{0,1\}$，$\sum_{k=1}^{K} z_{k}=1$ 。

> 其实每一个数据 $\boldsymbol{x}_i$ 都对应一个 $\boldsymbol{z}_i$ ，而 $\boldsymbol{z}_i$ 是 $K$ 维的，所以分量的完整写法应该是 $z_{ik}$ ，最终构成的是一个 $n×K$ 维的！！！通常我们会将 $z_{ik}$ 简写成 $z_k$ ，专指了与之对应的 $\boldsymbol{x}_i$ 。

（未知）第 $i$ 个观测数据里属于第 $k$ 个模型分量的后验概率：$\gamma(z_{ik})$ 

（未知）所有观测数据里属于第 $k$ 个模型分量的后验概率：$\gamma(z_k) = \sum^N_{i=1}\gamma(z_{ik})$ 



## 公式演化

属于第 $k$ 个模型分量的概率：
$$
p(z_{k}=1)=\pi_k
$$

（完整形式，与上式等价）：
$$
p(\boldsymbol{z})=p(z_{1})p(z_{2}) \dots p(z_{K})=\prod_{k=1}^{K} \pi_{k}^{z_k}
$$

属于第 $k$ 个模型分量的数据服从对应的高斯分布：
$$
p\left(\boldsymbol{x} | z_{k}=1\right)=\mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
$$

（完整形式，与上式等价）：
$$
p(\boldsymbol{x} | \boldsymbol{z})=\prod_{k=1}^{K} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)^{z_k}
$$

数据与模型的联合概率：
$$
p(\boldsymbol{x},z_k=1)=\pi_k\mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
$$

（完整形式，与上式等价）：
$$
p(\boldsymbol{x},\boldsymbol{z})=p(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})=\prod_{k=1}^{K} \left[\pi_k\mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right) \right]^{z_k}
$$

数据与模型的边缘概率：
$$
p(\boldsymbol{x})=\sum_{\boldsymbol{z}} p(\boldsymbol{z}) p(\boldsymbol{x} | \boldsymbol{z})=\sum_{k=1}^{K}p(\boldsymbol{x},z_k=1)=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \mathbf{\Sigma}_{k}\right)
$$

这样我们就得到了**GMM高斯混合模型**的概率密度函数：
$$
p(\boldsymbol{x} | \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \mathbf{\Sigma}_{k}\right)
$$

还有一个变量别忘记了：
$$
\begin{align} 
\gamma\left(z_{k}\right) \equiv p\left(z_{k}=1 | \boldsymbol{x}\right) 
&= \frac{p(\boldsymbol{x} , z_k=1)}{p(\boldsymbol{x})}\\
&= \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{j}, \mathbf{\Sigma}_{j}\right)} 
\end{align}
$$

$$
\gamma(z_{ik}) = p(z_{k}=1 | \boldsymbol{x}_i)=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_i | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_i | \boldsymbol{\mu}_{j}, \mathbf{\Sigma}_{j}\right)}
$$

> 注意：$\sum^K_{k=1} \gamma(z_k)= 1$ 以及 $\sum^K_{k=1} \gamma(z_{ik})= 1$， $\gamma(z_{ik}) \in \{0, 1\}$



## EM求解

> 还是使用最大似然估计



根据公式(8)，所有数据连乘得到似然函数，则对数似然函数为：
$$
l(\theta) = \sum_{i=1}^{N} \ln  p(\boldsymbol{x}_i | \boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{\Sigma})=\sum_{i=1}^{N} \ln \left[\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)\right]
$$

通过求导法去求解这个最大似然函数，可以得到 $(\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ 的最优值。

补充一点多元高斯函数的变换和求导：
$$
\mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\mathbf{\Sigma}_k|^{1 / 2}} \exp \left[-\frac{1}{2}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)^{T} \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right]
$$

$$
\ln {\mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)} = 
-\frac{D}{2} \ln (2 \pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_k| 
- \frac{1}{2} \left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_k\right)^{T} \boldsymbol{\Sigma}_k^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_k\right)
$$

$$
\begin{align}
\frac{\partial \mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\partial \boldsymbol{\mu}_k} 
&= \frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\mathbf{\Sigma}_k|^{1 / 2}} \exp \left[-\frac{1}{2}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)^{T} \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right]\left[-\boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right] \\
&= \mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\left[-\boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right]
\end{align}
$$

$$
\begin{align}
\frac{\partial \mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\partial \boldsymbol{\Sigma}_k} 
&= \frac{1}{(2 \pi)^{D / 2}} (-\frac{1}{2})\frac{1}{|\boldsymbol{\Sigma}_k|^{3/2}} \exp \left[-\frac{1}{2}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)^{T} \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right] + \frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\mathbf{\Sigma}_k|^{1 / 2}} \exp \left[-\frac{1}{2}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)^{T} \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i-\boldsymbol{\mu}_k)\right]\left[\frac{1}{2}\boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)(\boldsymbol{x}_i - \boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}\right] \\
&= \mathcal{N}(\boldsymbol{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\left(-\frac{1}{2}\boldsymbol{\Sigma}_k^{-1} + \frac{1}{2}\boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)(\boldsymbol{x}_i - \boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}\right)
\end{align}
$$



1. **均值 $\boldsymbol{\mu}_k$：**

$$
\frac{\partial l(\theta)}{\partial \boldsymbol{\mu}_k} 
= -\sum_{i=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\underbrace{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}_{\gamma(z_{ik})}}\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)
$$

> 类似于式(10)，为避免混淆，分母使用 $j$ 表示

$$
\begin{align}
\sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x}_i - \boldsymbol{\mu}_k) &= 0 \\
\sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{\Sigma}_k^{-1} \boldsymbol{\mu}_k &= \sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{\Sigma}_k^{-1} \boldsymbol{x}_i \\
\sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{\mu}_k &= \sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{x}_i \\
\boldsymbol{\mu}_k &= \frac{1}{N_k} \sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{x}_i
\end{align}
$$

> 其中 $N_k = \sum_{i=1}^{N} \gamma\left(z_{i k}\right)$ 表示属于第 $k$ 个模型分量的数据个数



2. **方差 $\boldsymbol{\Sigma}_k$：**

$$
\frac{\partial l(\theta)}{\partial \boldsymbol{\Sigma}_{k}} 
= \sum_{i=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\underbrace{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}_{\gamma(z_{ik})}}\left(-\frac{1}{2}\boldsymbol{\Sigma}_k^{-1} + \frac{1}{2}\boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)(\boldsymbol{x}_i - \boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}\right)
$$

$$
\begin{align}
\sum^N_{i=1} \gamma(z_{ik}) \left[-\frac{1}{2}\boldsymbol{\Sigma}_k^{-1} +\frac{1}{2}\boldsymbol{\Sigma}_k^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T}\boldsymbol{\Sigma}_k^{-1}\right] &= 0 \\
\sum^N_{i=1} \gamma(z_{ik})\boldsymbol{\Sigma}_k^{-1}  &= \sum_{i=1}^{N} \gamma\left(z_{i k}\right)\boldsymbol{\Sigma}_k^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T}\boldsymbol{\Sigma}_k^{-1} \\
\sum^N_{i=1} \gamma(z_{ik})  &= \sum_{i=1}^{N} \gamma\left(z_{i k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T}\boldsymbol{\Sigma}_k^{-1} \\
\sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{\Sigma}_k &= \sum_{i=1}^{N} \gamma\left(z_{i k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T} \\
\boldsymbol{\Sigma}_k &= \frac{1}{N_{k}} \sum_{i=1}^{N} \gamma\left(z_{i k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T}
\end{align}
$$



3. **分量概率 $\pi_k$：**

需要借助拉格朗日算子，加入 $\sum^K_{k=1} \pi_k = 1$：
$$
l(\theta) + \lambda \left(\sum^K_{k=1} \pi_k - 1\right)
$$
然后对 $\pi_k$ 求导：
$$
\frac{\partial}{\partial \pi_k}\left[l(\theta) + \lambda \left(\sum^K_{k=1} \pi_k - 1\right)\right] = \sum_{i=1}^{N} \frac{ \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)} + \lambda = 0
$$
两边同乘以 $\pi_k$：
$$
\begin{align}
\sum_{i=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\underbrace{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{i} | \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}_{\gamma(z_{ik})}} + \lambda \pi_k &= 0 \\
N_k + \lambda \pi_k &= 0
\end{align}
$$
上式对 $k$ 求和：
$$
\begin{align}
\sum^K_{k=1} N_k + \lambda \sum^K_{k=1}\pi_k &= 0 \\
N + \lambda &= 0 \\
\lambda &= -N
\end{align}
$$
代入式(33)，得：
$$
\pi_k = \frac{N_k}{N}
$$





## EM总结

算法步骤可归结为

1. 设定模型分量数 $K$，初始化 $\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$

2. E-Step：

   计算后验概率

   $$
   \gamma(z_{ik}) = \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_i | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_i | \boldsymbol{\mu}_{j}, \mathbf{\Sigma}_{j}\right)}
   $$

3. M-Step：

   更新参数
   $$
   \begin{align}
   N_k &= \sum_{i=1}^{N} \gamma\left(z_{i k}\right) \\
   \boldsymbol{\mu}_k &= \frac{1}{N_k} \sum^N_{i=1} \gamma(z_{ik}) \boldsymbol{x}_i \\
   \boldsymbol{\Sigma}_k &= \frac{1}{N_{k}} \sum_{i=1}^{N} \gamma\left(z_{i k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}_{k}\right)^{T} \\
   \pi_k &= \frac{N_k}{N}
   \end{align}
   $$

4. 判断参数是否收敛







## 参考

https://blog.csdn.net/jinping_shi/article/details/59613054

https://blog.csdn.net/v_july_v/article/details/81708386

https://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html

http://www.csuldw.com/2015/12/02/2015-12-02-EM-algorithms/

https://zhuanlan.zhihu.com/p/30483076

https://github.com/PRML/PRMLT

https://github.com/ctgk/PRML