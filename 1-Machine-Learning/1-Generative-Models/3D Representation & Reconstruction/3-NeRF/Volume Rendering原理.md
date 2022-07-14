Kajiya, J.T., Herzen, B.P.V.: Ray tracing volume densities. Computer Graphics (SIGGRAPH) (1984)

Max, N.: Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics (1995)





NeRF 原文中将 volume density $\sigma(\boldsymbol{x})$ 解释为：

> The volume density $\sigma(\boldsymbol{x})$ can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location x. 体积密度$\sigma(\boldsymbol{x})$ 可以解释为在位置x处无穷小粒子终止的射线的微分概率。


$$
C(\mathbf{r})=\int_{t_{n}}^{t_{f}} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) d t, \text { where } T(t)=\exp \left(-\int_{t_{n}}^{t} \sigma(\mathbf{r}(s)) d s\right)
$$
离散化后就成了
$$
\hat{C}(\mathbf{r})=\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) \mathbf{c}_{i}, \text { where } T_{i}=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)
$$
其中 $\delta_{i}=t_{i+1}-t_{i}$ 是两个相邻点的距离 



traditional alpha compositing $\alpha_{i}=1-\exp \left(-\sigma_{i} \delta_{i}\right)$





解释 T 累积的透光率

越远越不容易透过 

内部透射率是指吸收造成的能量损失

能量还剩多少，出发点是1，最后是0


$$
T = e^{-x} x 是深度
$$




The function T(t) denotes the accumulated transmittance along the ray fromtn to t, i.e., the probability that the ray travels from tn to t without hittingany other particle.



![image-20220713160847927](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/image-20220713160847927.png)



一个3D shape在空间中，我们将它假想成是由一系列粒子构成的，就像点云一样吧

相机的光线是射向这些粒子的

每穿过一个粒子，这个光线的能量就要衰减一次



透明度-100%是全透明，0%是不透明


$$
dd
$$




https://www.youtube.com/watch?v=otly9jcZ0Jg&ab_channel=NeuralRendering



首先我们要表示的场景，可以看成是一团由无限小的彩色粒子组成的云 (a cloud of tiny colored particles)。

相机会发射光束 $\mathbf{r}(t)=\mathbf{O}+t \mathbf{d}$

当光束打到位于$t$位置的粒子时，会返回这个粒子的颜色$\mathbf{c}(t)$

但实际上光束是否打到了这个粒子是不确定的，我们用一个概率密度$\sigma$来描述，并称之为体密度(volume density)

P(hit at t) = $\sigma(t)dt$

为了判断t是不是第一次被打到，就需要知道在t之前一次都没有被打到的概率，称为透射率(transmittance)，

P(no hits before t) = $T(t)$

在这里有一个关系是

P(no hits before t+dt) = P(no hits before t) * P(no hit at t)


$$
\begin{align}
T(t+dt) &= T(t) \times (1 - \sigma(t)dt) \\
\text{Split up differential} \quad T(t) + T^\prime(t)dt &= T(t) - T(t)\sigma(t)dt \\
\text{Rearange} \quad \frac{T^\prime(t)}{T(t)} &= - \sigma(t) \\
\text{Intergrate} \quad \log T(t) &= \int_{t_0}^t - \sigma(s)ds \\
\text{Exponential} \quad T(t) &= \exp(\int_{t_0}^t - \sigma(s)ds)
\end{align}
$$
