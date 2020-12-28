# Training Instability



## Field Introduction

To tackle the instability of the training procedure

> Why it's a problem? 

GAN need to find a Nash equilibrium of a non-convex game in a continuous and high dimensional parameter space.



These methods can be divided into two categories:

- **Normalization**
  
  - **Spectral normalization** (weight matrices in the discriminator are divided by an approximation of their largest singular value)
- **Regularization**
  
  - **Wasserstein** (penalize the gradient norm of straight lines between real data and generated data)
  
  - [^Roth2017] (directly regularize the squared gradient norm for both the training data and the generated data.) 
  
  - **[DRAGAN](#DRAGAN)** (penalize the gradients at Gaussian perturbations of training data) 
  
  - Consistency regularization ()
  
    > pros & cons: simple to implement, not particularly computationally burdensome, and relatively insensitive to hyper-parameters





Will simultaneous regularization and normalization improve GANs performance?

> Won't. Both regularization and normalization are motivated by controlling Lipschitz constant of the discriminator
>
> A large-scale study on regularization and normalization in GANs



## Bibliography

[CR-GAN](#CR-GAN)

[ICR-GAN](#ICR-GAN)

---

### ICR-GAN

[Improved Consistency Regularization for GANs](https://arxiv.org/pdf/2002.04724.pdf)

**`[AAAI 2020]`**	**`(Google)`**	**`[Zhengli Zhao, Han Zhang]`**	**([:memo:]())**	**[[:octocat:](https://github.com/google/compare_gan)]**

<details><summary>Click to expand</summary><p>


![image-20201219215131885](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201219215132.png)

> **Summary**

They improve [CR-GAN](#CR-GAN) in two ways (apply forms of consistency regularization to the generated images, the latent vector space, and the generator):

- Balanced Consistency Regularization, in which generator samples are also augmented along with training data.
- Latent Consistency Regularization, in which draws from the prior are perturbed, and the sensitivity to those perturbations is discouraged and encouraged for the discriminator and the generator, respectively.

> **Details**

balanced consistency regularization (bCR)

</p></details>

---




### CR-GAN

[Consistency regularization for generative adversarial networks](https://arxiv.org/pdf/1910.12027.pdf)

**`[ICLR 2020]`**	**`(Google)`**	**`[Han Zhang, Honglak Lee]`**	**([Code]())**

<details><summary>Click to expand</summary><p>


> **Summary**

They propose a training stabilizer based on **consistency regularization**. In particular, they **augment data** passing into the GAN discriminator and **penalize the sensitivity** of the discriminator to these augmentations.

**Consistency regularization** is widely used in semi-supervised learning to ensure that the classifier output remains unaffected for an unlabeled example even it is augmented in semantic-preserving ways.

The pipeline is to first augment images with semantic-preserving augmentations before they are fed into the discriminator and penalize the sensitivity of the discriminator to these augmentations.

> **Details**

$T(x)$ donates a stochastic data augmentation function. $D(x)$ donates the last layer before the activation function. The proposed regularization is given by
$$
\min_{D} L_{c r} = \min_{D} \|D(x)-D(T(x))\|^{2}
$$
The overall consistency regularized GAN (CR-GAN) objective is written as
$$
L_{D}^{c r}=L_{D}+\lambda L_{c r}, \quad L_{G}^{c r}=L_{G}.
$$

> **Augmentation type**

1 Gaussian Noise; 2 **Random shift & flip**; 3 Cutout; 4 Random shift & flip with cutout

The experiment shows that No.2 performs best.



</p></details>

---

[A large-scale study on regularization and normalization in GANs](https://arxiv.org/pdf/1807.04720.pdf)

**`[ICML 2019]`**	**`(Google)`**	**`[Karol Kurach, Sylvain Gelly]`**	**([:memo:]())**	**[[:octocat:](https://github.com/google/compare_gan)]**

<details><summary>Click to expand</summary><p>


**Summary**

> 

</p></details>

---

### DRAGAN

[On Convergence and Stability of GANs](https://arxiv.org/pdf/1705.07215.pdf)

**`[None 2017]`**	**`(Gatech)`**	**`[Naveen Kodali, James Hays]`**	**([:memo:]())**	**[[:octocat:](https://github.com/kodalinaveen3/DRAGAN)]**

<details><summary>Click to expand</summary><p>


**Summary**

> 


</p></details>

---





[^Roth2017]: Stabilizing training of generative adversarial networks through regularization

