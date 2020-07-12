English | [简体中文](./README.zh-CN.md)


#  Generative Adversarial Networks(GAN)


![country](https://img.shields.io/badge/country-China-red)

This is my  research summary on Generative Adversarial Networks and I sort them into:

- Traditional GAN
- Applicational GAN
- Multi-Objective GAN



**Introduction of GAN**

> 

If you want to know more about more details of the derivation or the difficult of GAN’s training, you can see the part of [Traditional GAN](#Traditional-GAN)



**Why is there an “s” after GANs?**

> It means GAN and its variants



**Commonly used datasets**

> Mnist, CelebA, LSUN, and ImageNet



**Facing problem**

> - mode collapse: the generator can only learn some limited patterns from the large-scale target datasets, or assigns all of its probability mass to a small region in the space.
> - vanishing gradient: 



**Evaluation metrics of GAN**

> paper: https://arxiv.org/pdf/1806.07755.pdf
>
> code: https://github.com/xuqiantong/GAN-Metrics
>
> blog: https://zhuanlan.zhihu.com/p/99375611



## [Traditional GAN](1-Traditional-GAN)

The development of some famous GAN models including <u>Vanilla GAN</u>, <u>DCGAN</u>, <u>WGAN</u>

## [Applicational GAN](2-Applicational-GAN)

Some applications of GAN including the use of defense

## [Multi-Objective GAN](3-Multi-Objective-GAN)

Add multi-objective and evolutionary algorithm into GAN