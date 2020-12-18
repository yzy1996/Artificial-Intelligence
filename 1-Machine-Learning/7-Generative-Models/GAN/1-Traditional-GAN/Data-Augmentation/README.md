To tackle the instability of the training procedure

> Why it's a problem? GAN need to find a Nash equilibrium of a non-convex game in a continuous and high dimensional parameter space.



These methods can be divided into two categories:

- **Normalization**
  - **Spectral normalization** (weight matrices in the discriminator are divided by an approximation of their largest singular value)
- **Regularization**
  - **Wasserstein** (penalize the gradient norm of straight lines between real data and generated data)
  - [^Roth 2017] (directly regularize the squared gradient norm for both the training data and the generated data.) 
  - **[DRAGAN](#DRAGAN)** (penalize the gradients at Gaussian perturbations of training data) 



Will simultaneous regularization and normalization improve GANs performance?

> Won't. Both regularization and normalization are motivated by controlling Lipschitz constant of the discriminator
>
> A large-scale study on regularization and normalization in GANs





[CR-GAN](#CR-GAN)









### CR-GAN

[Consistency regularization for generative adversarial networks](https://arxiv.org/pdf/1910.12027.pdf)

**`[ICLR 2020]`**	**`(Google)`**	**`[Han Zhang, Honglak Lee]`**	**([:memo:]())**	**[[:octocat:]()]**

<details><summary>Click to expand</summary><p>


**Summary**

> They propose a training stabilizer based on **consistency regularization**. In particular, they **augment data** passing into the GAN discriminator and **penalize the sensitivity** of the discriminator to these augmentations.
>
> **Consistency regularization** is widely used in semi-supervised learning to ensure that the classifier output remains unaffected for an unlabeled example even it is augmented in semantic-preserving ways.


</p></details>

---





z



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





[^Roth 2017]: Stabilizing training of generative adversarial networks through regularization
