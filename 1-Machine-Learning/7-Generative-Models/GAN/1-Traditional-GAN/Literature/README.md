[TOC]

### GAN: Vanilla GAN

> NeurIPS 2014 
>
> 

---

### CGAN: Conditional GAN

> CoRR abs/1411.1784 2014

Conditional GAN [18, 26] extends the GAN by feeding the labels to both G and D to generate images conditioned on the label, which can be the class label, modality information, or even partial data for inpainting. It has been used to generate MNIST digits conditioned on the class label and to learn multi-modal models. In conditional GAN, D is trained to classify a real image with mismatched conditions to a fake class. In DR-GAN, D classifies a real image to the corresponding class based on the labels.



The conditional GAN[10] concatenates condition vector into the input of the generator and the discriminator. Variants of this method was successfully applied in [7, 11, 14]. [7] obtained visually discriminative vector representation of text descriptions and then concatenated that vector into every layer of the discriminator and the noise vector of the generator. [11] used a similar method to generate face images from binary attribute vectors such as hair styles, face shapes, etc. In [14], Structure-GAN generates surface normal maps and then they are concatenated into noise vector of Style-GAN to put styles in those maps.







---

### WGAN-GP: Improved Training of Wasserstein GANs

**[`2017`]** **[`NIPS`]** **[[:memo:]()]** **[[:octocat:](https://github.com/igul222/improved_wgan_training)]**

<details><summary>Click to expand</summary><p>


**The main work:**

> To solve the problem of classification which is vulnerable to adversarial perturbations: carefully crafted small perturbations can cause misclassification of legitimate images. I can archive it into the field of **Machine deception**. (small perturbations do not affect human recognition but machine classifier)
>
> I can summarize their work as follows: given a picture with deception, GAN is used to generate the picture without deception, and finally classifier is used to classify.
>
> They use the GD of reconstruction error ($ \|G(\mathbf{z})-\mathbf{x}\|_{2}^{2} $) to find optimal $ G(z) $ 

**The methods it used:** 

- [ ] Several ways of attack: Fast Gradient Sign Method (FGSM), Randomized Fast Gradient Sign Method (RAND+FGSM), The Carlini-Wagner (CW) attack
- [ ] Lebesgue-measure

**Its contribution:**

> They proposed a novel defense strategy utilizing GANs to enhance the
> robustness of classification models against black-box and white-box adversarial attacks

**My Comments:**

> This work can be referred to using AE (Auto Encoder) for noise reduction. It’s just an easy application of GANs.
>

</p></details>

---





---





---





**[`CG-GAN: An Interactive Evolutionary GAN-based Approach for Facial Composite Generation`]**

**[`2020`]** **[`AAAI`]** 

<details><summary>Click to expand</summary><p>


**The main work:**

> Facial Composite is to synthesize two target pictures into one pictures 

**The methods it used:** 

> - [ ] using **pg-GAN** to create high-resolution human faces
> - [x] using Latent Variable Evolution (**LVE**) to guide the search through a process of interactive evolution 

**Its contribution:**

> It extends LVE with the ability to freeze certain features discovered during the search, and enables a more controlled user-recreation of target images.

**My Comments:**

> It’s a new 

</p></details>

---

**[`$R^2GAN$ Recipe Retrieval Generative Adversarial Network`]**

**[`2019`]** **[`CVPR`]**

<details><summary>Click to expand</summary><p>


**The main work:**

> Aim at exploring the feasibility of generating image from procedure text for retrieval problem. The specific content of the text is food recipe

It belongs to **NLP**, to solve a problem of information retrieval

The simplest way is linear scan

index the document-boolean retrieval model 

**The methods it used:** 

This paper studies food-to-recipe and recipe-to-food retrieval

>They specially use a GAN with one generator and dual discriminators

two-level ranking loss



**My Comments:**

> It’s a new 

</p></details>

---

**[`Self-Attention Generative Adversarial Networks`]**

**[`2019`]** **[`PMLR`]** **[[:octocat:](https://github.com/heykeetae/Self-Attention-GAN)]**

<details><summary>Click to expand</summary><p>


**The main work:**

> It firstly introduced **Attention** into GAN, mainly apply on high-resolution detail generation.
>
> [ref_blog](https://zhuanlan.zhihu.com/p/55741364)



**The methods it used:** 

![img](https://media.arxiv-vanity.com/render-output/2954637/fig/framework.png)



**My Comments:**

> It’s a new 

</p></details>



$$
\begin{aligned}
{\omega}_{k+1}^{i}
&={\omega}_{k}-\alpha \bar{\nabla} f_{i}\left({\omega}_{k}, D_{i n}^{i}\right)\qquad (4)\\
&={\omega}_{k}-\beta_{k} \frac{1}{B} \sum_{i \in B_{k}}\left(I-\alpha \tilde{\nabla}^{2} f_{i}\left({\omega}_{k}, \mathcal{D}_{h}^{i}\right)\right) \tilde{\nabla} f_{i}\left({\omega}_{k+1}^{i}, \mathcal{D}_{o}^{i}\right)\qquad (5)
\end{aligned}
$$

### AC-GAN(2017)

**[`2017`]** **[`ICLR`]** **[[:memo:](./Defense-GAN.pdf)]** **[[:octocat:](https://github.com/kabkabm/defensegan)]**

<details><summary>Click to expand</summary><p>


**The main work:**

> 

**The methods it used:** 

- [ ] 

**Its contribution:**

> They proposed a novel defense strategy utilizing GANs to enhance the
> robustness of classification models against black-box and white-box adversarial attacks

**My Comments:**

> 
>

</p></details>

---



### SRGAN

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

[CVPR 2017]

[[:octocat:](https://github.com/JustinhoCHN/SRGAN_Wasserstein)]

---

### PGGAN

[[:octocat:](https://github.com/ptrblck/prog_gans_pytorch_inference)] [[:octocat:](https://github.com/nashory/pggan-pytorch)]





---

## StyleGAN

[A Style-Based Generator Architecture for Generative Adversarial Networks]()

**`(2019)`**	**`[CVPR]`**	**[[:memo:]()]**	**[[:octocat:](https://github.com/heykeetae/Self-Attention-GAN)]**

