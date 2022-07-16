Kajiya, J.T., Herzen, B.P.V.: Ray tracing volume densities. Computer Graphics (SIGGRAPH) (1984)

Max, N.: Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics (1995)



![image-20220715163557865](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/image-20220715163557865.png)

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

> 为什么会有dt，是看成在t周围有一个小区间。这样才从概率密度到了概率，因为概率是概率密度的积分，一小点上的概率

为了判断t是不是第一次被打到，就需要知道在t之前一次都没有被打到的概率，称为透射率(transmittance)，

P(no hits before t) = $T(t)$

在这里有一个关系是

P(no hits before t+dt) = P(no hits before t) * P(no hit at t)


$$
\begin{align}
T(t+dt) &= T(t) \times (1 - \sigma(t)dt) \\
\text{Split up differential} \quad T(t) + T^\prime(t)dt &= T(t) - T(t)\sigma(t)dt \\
\text{Rearange} \quad \frac{T^\prime(t)}{T(t)} &= - \sigma(t) \\
\text{Intergrate} \quad \log T(t) &= \int_{t_0}^t - \sigma(t)dt \\
\text{Exponential} \quad T(t) &= \exp(\int_{t_0}^t - \sigma(t)dt) 
\end{align}
$$
P(first hit at t) = P(no hits before t) * P(hit at t) = $T(t)\sigma(t)dt$

点 t 处是有颜色的，加上去就是 $\int_{t_{0}}^{t_{1}} T(t) \sigma(t) \mathbf{c}(t) d t$



前面的会遮挡后面的。




$$
\begin{align}
\int T(t) \sigma(t) \mathrm{c}(t) d t &\approx \sum_{i=1}^{n} \int_{t_{i}}^{t_{i+1}} T(t) \sigma_{i} \mathrm{c}_{i} d t \\
\text{Substitute} &= \sum_{i=1}^{n} T_{i} \sigma_{i} \mathbf{c}_{i} \int_{t_{i}}^{t_{i+1}} \exp \left(-\sigma_{i}\left(t-t_{i}\right)\right) d t \\
&=\sum_{i=1}^{n} T_{i} \sigma_{i} \mathbf{c}_{i} \frac{\exp \left(-\sigma_{i}\left(t_{i+1}-t_{i}\right)\right)-1}{-\sigma_{i}} \\
&=\sum_{i=1}^{n} T_{i} \mathbf{c}_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right)
\end{align}
$$

$$
\text { For } t \in\left[t_{i}, t_{i+1}\right], T(t)=\exp \left(-\int_{t_{1}}^{t_{i}} \sigma_{i} d s\right) \exp \left(-\int_{t_{i}}^{t} \sigma_{i} d s\right)
$$

$$
\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)=T_{i} \quad \exp \left(-\sigma_{i}\left(t-t_{i}\right)\right)
$$

衡量被前面部分遮挡的有多少 | 又有多少被现在这块遮挡了





可以单独拎出来看 $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$

这样我们前面的表达式，substitute that back into the equations we jus tderived 可以写成
$$
\text{color} =\sum_{i=1}^{n} T_{i} \alpha_{i} \mathbf{c}_{i}=\sum_{i=1}^{n} T_{i} \mathbf{c}_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right)
\\
T_{i}=\prod_{j=1}^{i-1}\left(1-\alpha_{j}\right)=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)
$$
这样是可以和很多其他渲染技术统一的







本来是要对光束上每一个点进行积分，
