**`Unsupervised Discovery of Interpretable Directions in the GAN Latent Space`**

**`[ICML 2020]`** **[[:octocat:](https://github.com/anvoynov/GANLatentDiscovery)]** 

<details><summary>Click to expand</summary><p>

![A9Rlu0i5j_139dt6w_ea4](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201101155344.png)


Features: **unsupervised, background removal**

Method: via jointly learning **a set of directions** and a **model** to distinguish the corresponding image transformations



based on InfoGAN



有一个解耦开的矩阵 $A \in \mathbb{R}^{d \times K}$

一个网络R，用来判断是哪个解耦出来的分量

Self-supervised learning

![mylatex20201030_110850](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201030110908.svg)



</p></details>

---





**`Controlling generative models with continuos factors of variations`**

**`ICLR 2020`**

<details><summary>Click to expand</summary><p>

Features: 

main contribution: solves the optimization problem in the latent space that maximizes the score of the pretrained model, predicting image memorability. Want to increase of memorability

</p></details>

---

## InterFaceGAN

[Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/abs/1907.10786)

**`[CVPR 2020]`**	**`(CUHK)`**	**`[Yujun Shen, Bolei Zhou]`**	**([:memo:]())**	**[[:octocat:](https://github.com/genforce/interfacegan)]**

<details><summary>Click to expand</summary><p>


![image-20201119164856956](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201119164859.png)

**Formulation**
$$
\operatorname{argmin}_{\theta} \mathcal{L}(\theta)=\mathbb{E}_{\mathbf{z}, \mathbf{y}, \alpha}\left[\left(A\left(G\left(T_{\theta}(\mathbf{z}, \alpha), \mathbf{y}\right)\right)-(A(G(\mathbf{z}, \mathbf{y}))+\alpha)\right)^{2}\right]
$$

$$
T_{\theta}(\mathbf{z}, \alpha)=\mathbf{z}+\alpha \theta
$$

$G$: use the Generator of [BigGAN]() which is pretrained on ImageNet

$A$: use a CNN of [MemNet]() to assesses an image property of memorability

$T$: moves the input $\mathbf{z}$ along a certain direction $\theta$ 

: learn to increase (or decrease) the memorability with a certain amount $\alpha$





</p></details>



## GANalyze

[GANalyze: Toward visual definitions of cognitive image properties](https://arxiv.org/abs/1906.10112)

**`[CVPR 2019]`**	**`(MIT)`**	**`[Lore Goetschalckx, Alex Andonian]`**	**([:memo:]())**	**[[:octocat:](https://github.com/LoreGoetschalckx/GANalyze)]**

<details><summary>Click to expand</summary><p>


![image-20201119164856956](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201119164859.png)

**Formulation**
$$
\operatorname{argmin}_{\theta} \mathcal{L}(\theta)=\mathbb{E}_{\mathbf{z}, \mathbf{y}, \alpha}\left[\left(A\left(G\left(T_{\theta}(\mathbf{z}, \alpha), \mathbf{y}\right)\right)-(A(G(\mathbf{z}, \mathbf{y}))+\alpha)\right)^{2}\right]
$$

$$
T_{\theta}(\mathbf{z}, \alpha)=\mathbf{z}+\alpha \theta
$$

$G$: use the Generator of [BigGAN]() which is pretrained on ImageNet

$A$: use a CNN of [MemNet]() to assesses an image property of memorability

$T$: moves the input $\mathbf{z}$ along a certain direction $\theta$ 

: learn to increase (or decrease) the memorability with a certain amount $\alpha$





</p></details>

---

