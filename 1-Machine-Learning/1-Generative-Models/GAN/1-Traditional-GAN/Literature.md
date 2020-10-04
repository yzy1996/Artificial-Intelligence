[TOC]



**[`Defense-GAN: protecting classifiers against adversarial attacks using generative models`]**

**[`2018`]** **[`ICLR`]** **[[:memo:](./Defense-GAN.pdf)]** **[[:octocat:](https://github.com/kabkabm/defensegan)]**

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



**[`Evolutionary Multi-Objective Optimization Driven by Generative Adversarial Networks`]**

**[`2019`]** **[``]** **[[:memo:](./EMOO-Driven-by-GAN.pdf)]** **[[:octocat:]()]** 



---



**[`Evolutionary Generative Adversarial Networks`]**

**[`2019`]** **[`TEVC`]** **[[:memo:](./E-GAN.pdf)]** **[[:octocat:](https://github.com/WANG-Chaoyue/EvolutionaryGAN)]** **[[:octocat:](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch)]**

<details><summary>Click to expand</summary><p>


**The main work:**

> 

**The methods it used:** 

**Its contribution:**

**My Comments:**

</p></details>

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

# AC-GAN(2017)

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



Style-GAN



