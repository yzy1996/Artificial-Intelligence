# <p align=center>`Generative Adversarial Networks (GANs)`</p>

A collection of resources on Generative Adversarial Networks (GANs).



## Table of Contents

- [Introduction](#Introduction)

- [Basic GAN](#Basic-GAN)

- [Applicational GAN](#Applicational-GAN)

  - [Semantic image synthesis](#Semantic-image-synthesis)

    the goal is to generate multi-modal photorealistic images in alignment with a given semantic label map

- [Multi-Objective GAN](#Multi-Objective-GAN)





## Introduction

> Introduce the principle of GAN, for more details, see the subfile

阶段1：能够

- mode collapse
- gradient vanishing

阶段2：够好 High-quality

- high-resolution
- controllable (representation disentanglement)
- multi-view consistent

阶段3：新追求

- 3D 



可以改进的地方：

- loss function
- regularization and normalization
- architecture



按应用场景分：

- image
- text
- audio
- video



Some review to help you know this field

[Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy]()

[A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications]() 

[Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications]()

[Generative adversarial network in medical imaging: A review]()



**Introduction of GAN**

> 

If you want to know more about more details of the derivation or the difficult of GAN’s training, you can see the part of [Traditional GAN](#Traditional-GAN)



**Why Are GANs So Popular?**

GANs are popular partly because they tackle the important unsolved challenge of unsupervised learning.

If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. We know how to make the icing and the cherry, but we don’t know how to make the cake. – Yann LeCun, 2016.



**Why is there an “s” after GANs?**

> It means GAN and its variants



**Commonly used datasets**

> Mnist, CelebA, LSUN, and ImageNet



**Facing problem**

> - mode collapse: diversity the generator can only learn some limited patterns from the large-scale target datasets, or assigns all of its probability mass to a small region in the space.
> - vanishing gradient: 



**Evaluation metrics of GAN**

> paper: https://arxiv.org/pdf/1806.07755.pdf
>
> code: https://github.com/xuqiantong/GAN-Metrics
>
> blog: https://zhuanlan.zhihu.com/p/99375611



## Traditional GAN

The development of some famous GAN models including <u>Vanilla GAN</u>, <u>DCGAN</u>, <u>WGAN</u>

## Applicational GAN

Some applications of GAN including the use of defense



### Semantic image synthesis

[You Only Need Adversarial Supervision for Semantic Image Synthesis](https://arxiv.org/pdf/2012.04781.pdf)  
**[`ICLR 2021`] ()**  
*Vadim Sushko, Edgar Schönfeld, Dan Zhang, Juergen Gall, Bernt Schiele, Anna Khoreva*



## Multi-Objective GAN

Add multi-objective and evolutionary algorithm into GAN





### objective functions of GANs



**vanilla GAN**
$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

$$
\min J^G = \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

where $J^G$ is the cost of for generator, $\log D(x)$ is the cross-entropy between $[D(x) \quad 1-D(x)]^T$ and $[1 \quad 0]^T$. Likewise,  $\log (1-D(G(z)))$ is the cross-entropy between $[1-D(G(z)) \quad D(G(z))]^T$ and $[1 \quad 0]^T$. It’s because that the cross-entropy 

For a fixed generator $G$, the optimal discriminator $D$ is:
$$
D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)},
$$
For this optimal $D^*$, the optimal $G$ satisfies:
$$
p_g(x) =p_{data}(x).
$$

> Problem: 
>
> The cross-entropy of $G$ can be expressed as follow:
> $$
> H^G = 1 * \log(1-D(G(z))) + 0*log(D(G(z))) = \log(1-D(G(z)))
> $$
> In early training progress, D can easily distinguish fake samples from real samples ($D(G(z)) \rightarrow 0$). This results in G not having sufficient gradient to improve, which is called **training instability**. Rather than training G in the way of Equation (2), another way of Equation (6) could provides larger gradients in early training.
> $$
> \min J^G = \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[-\log (D(G(\boldsymbol{z})))]
> $$
> So a new cross-entropy of $G$ can be expressed as:
> $$
> H = 1 * \log(-D(G(z))) + 0*log(1 + D(G(z)))
> $$



blog 

https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/

https://wiki.pathmind.com/generative-adversarial-network-gan





some new work

https://github.com/hankhank10/fakeface

styleGAN

styleGAN2

https://thispersondoesnotexist.com/





A GAN consists of a generator $G$ and a discriminator $D$, both are conducted by a neural network. $G$ takes a latent variable $z \sim p(z)$ sampled from a prior distribution and maps it to the observation space $\mathcal{X}$. $D$ takes an observation $x \in \mathcal{X}$ and produces a decision output over possible observation sources (either from $G$ or from the empirical data distribution). 



The generator and the discriminator in the standard GAN training procedure are trained by minimizing the following objectives:
$$
\begin{align}
&L_{D}=-\mathbb{E}_{x \sim p_{\text {data }}}[\log D(x)]-\mathbb{E}_{z \sim p(z)}[1-\log D(G(z))], \\
&L_{G}=-\mathbb{E}_{z \sim p(z)}[\log D(G(z))].
\end{align}
$$
This formulation is originally proposed by Goodfellow et al. (2014) as non-saturating (NS) GAN. A significant amount of research has been done on modifying this formulation in order to improve the training process. A notable example is the **hinge-loss** version of the adversarial loss:
$$
\begin{align}
&L_{D}=-\mathbb{E}_{x \sim p_{\text {data }}}[\min (0,-1+D(x))]-\mathbb{E}_{z \sim p(z)}[\min (0,-1-D(G(z)))], \\
&L_{G}=-\mathbb{E}_{z \sim p(z)}[D(G(z))].
\end{align}
$$
Another commonly adopted GAN formulation is the **Wassertein** GAN (WGAN), where the authors propose clipping the weights to enforce the continuous of Wassertein distance. The loss function of WGAN is:
$$
\begin{align}
&L_{D}=-\mathbb{E}_{x \sim p_{\text {data }}}[D(x)]+\mathbb{E}_{z \sim p(z)}[D(G(z))], \\
&L_{G}=-\mathbb{E}_{z \sim p(z)}[D(G(z))].
\end{align}
$$



[How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)





GAN models share two common aspects: solving a challenging saddle point optimization problem, interpreted as an adversarial game between a generator and a discriminator functions.



a popular paradigm to learn the distribution of the observed data



### Basic



### Applications

