

[GANSpace](#GANSpace)

[Unsupervised](#Unsupervised) 

[InterFaceGAN](#InterFaceGAN)

[GANalyze](#GANalyze)

[Factors of Variations](#Factors-of-Variations)

[GAN_Steerability](#GAN_Steerability)

---

## GANSpace

[GANSpace: Discovering Interpretable GAN Controls](https://arxiv.org/abs/2004.02546)

**`[NeurIPS 2020]`**	**`(Adobe&NVIDIA)`**	**`[Erik Härkönen, Aaron Hertzmann]`**	**([:memo:]())**	**[[:octocat:](https://github.com/harskish/ganspace)]**

<details><summary>Click to expand</summary><p>


<div align=center><img width="700" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201121154059.png"/></div>

> **Keywords**

PCA



> **Goal**

find useful directions in $z$ space



> **Pipeline**

sample $N$ random vector $z_{1:N}$, then compute the corresponding $w_i = M(z_i)$ value

compute PCA of these $w_{1:N}$ values, with a bias $V$ for $W$

given a new image defined by $w$, edit it by varying PCA coordinates $x$
$$
w^{\prime} = w + Vx
$$


</p></details>

---

## Unsupervised 

[Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](https://arxiv.org/abs/2002.03754)

**`[ICML 2020]`**	**`(Russia)`**	**`[Andrey Voynov, Artem Babenko]`**	**([:memo:]())**	**[[:octocat:](https://github.com/anvoynov/GANLatentDiscovery)]**

<details><summary>Click to expand</summary><p>


![A9Rlu0i5j_139dt6w_ea4](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201101155344.png)


Features: **unsupervised, background removal**

> **Framework**

via jointly learning **a set of directions** and a **model** to distinguish the corresponding image transformations



based on InfoGAN



有一个解耦开的矩阵 $A \in \mathbb{R}^{d \times K}$

一个网络R，用来判断是哪个解耦出来的分量

Self-supervised learning

![mylatex20201030_110850](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201030110908.svg)



</p></details>

---

## InterFaceGAN

[Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/abs/1907.10786)

**`[CVPR 2020]`**	**`(CUHK)`**	**`[Yujun Shen, Bolei Zhou]`**	**([:memo:]())**	**[[:octocat:](https://github.com/genforce/interfacegan)]**

<details><summary>Click to expand</summary><p>


<div align=center><img width="300" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201119220419.png"/></div>

> **Assumption**

For any binary semantic (e.g., male v.s. female), there exists a **hyperplane** in the latent space serving as the **separation boundary**. Semantic remains the same when the latent code walks within the same side of the hyperplane yet turns into the opposite when across the boundary.

> **Formulation**

$$
\mathrm{d}(\mathbf{n}, \mathbf{z})=\mathbf{n}^{T} \mathbf{z}
$$

$$
f(g(\mathbf{z}))=\lambda \mathrm{d}(\mathbf{n}, \mathbf{z})
$$

$G$: use the Generator of [PGGAN]() and [StyleGAN]() which are pretrained on [CelebA-HQ]()

> **Framework**

latent code z -> image x -> label

latent code z -> label

then train five independent linear SVMs on pose, smile, age, gender, eyeglasses

finally find n and edit the latent code z with $z_{edit} = z + \alpha n$



</p></details>

---

## GANalyze

[GANalyze: Toward visual definitions of cognitive image properties](https://arxiv.org/abs/1906.10112)

**`[CVPR 2019]`**	**`(MIT)`**	**`[Lore Goetschalckx, Alex Andonian]`**	**([:memo:]())**	**[[:octocat:](https://github.com/LoreGoetschalckx/GANalyze)]**

<details><summary>Click to expand</summary><p>


![image-20201119164856956](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201119164859.png)

> **Formulation**

$$
\operatorname{argmin}_{\theta} \mathcal{L}(\theta)=\mathbb{E}_{\mathbf{z}, \mathbf{y}, \alpha}\left[\left(A\left(G\left(T_{\theta}(\mathbf{z}, \alpha), \mathbf{y}\right)\right)-(A(G(\mathbf{z}, \mathbf{y}))+\alpha)\right)^{2}\right]
$$

$$
T_{\theta}(\mathbf{z}, \alpha)=\mathbf{z}+\alpha \theta
$$

$G$: use the Generator of [BigGAN]() which is pretrained on ImageNet

$A$: use a CNN of [MemNet]() to assesses an image property of memorability

$T$: moves the input $\mathbf{z}$ along a certain direction $\theta$ 

learn to increase (or decrease) the memorability with a certain amount $\alpha$



</p></details>

---

## Factors of Variations

[Controlling generative models with continuous factors of variations](https://arxiv.org/abs/2001.10238)

**`[ICLR 2020]`**	**`(France)`**	**`[Antoine Plumerault, Hervé Le Borgne]`**	**([:memo:]())**	**[[:octocat:](https://github.com/AntoinePlumerault/Controlling-generative-models-with-continuous-factors-of-variations)]**

<details><summary>Click to expand</summary><p>


>**Framework**

for an original generation: $I = G(z_0)$

want a transformation: $I \rightarrow \mathcal{T}_{t}(I)$ (e.g. $\mathcal{T}$ is a rotation, then $t$ is the angle)

approximate $z_T$ by $G(z_T) \approx \mathcal{T}_{t}(I)$ -> [invert the generator]()

then estimate the direction encoding the factor of variation described by $\mathcal{T}$ with the difference between $z_0$ and $z_T$ 

**given $\mathcal{T}$ to get $z_T$** 

> **Difficulty**

- reconstruction error
  $$
  \hat{z}=\underset{z \in \mathcal{Z}}{\arg \min } \mathcal{L}(I, G(\boldsymbol{z}))
  $$
  choose the error of the MSE on images in the frequency domain

- recursive estimation of the trajectory

  decomposing the transformation

> **Dataset**

[dSprites]() and [ILSVRC]()

> **GAN model**

[BigGAN](): two vector input (a latent vector **z** and a one-hot vector **c** to generate conditional categories)



</p></details>

---

## GAN_Steerability

[On the "steerability" of generative adversarial networks](https://arxiv.org/abs/1907.07171)

**`[ICLR 2020]`**	**`(MIT)`**	**`[Ali Jahanian, Lucy Chai, Phillip Isola]`**	**([:memo:]())**	**[[:octocat:](https://ali-design.github.io/gan_steerability/)]**

<details><summary>Click to expand</summary><p>


<div align=center><img width="800" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201121120437.png"/></div>

> **Formulation**

$$
w^{*}=\underset{w}{\arg \min } \mathbb{E}_{z, \alpha}[\mathcal{L}(G(z+\alpha w), \operatorname{edit}(G(z), \alpha))]
$$

objective $\mathcal{L}$ could be [$L2$ loss]() or [LPIPS perceptual image similarity metric]()



> **Pipeline**

GAN model: BigGAN and StyleGAN




</p></details>

---

