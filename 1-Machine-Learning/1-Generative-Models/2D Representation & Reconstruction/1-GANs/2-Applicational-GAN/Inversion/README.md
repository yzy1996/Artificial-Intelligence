# GAN Inversion

which means embedding/mapping a given image into latent space.



## Introduction

GAN inversion aims to invert a given image back into the latent space of a pretrained GAN model.





前提假设是：When z1; z2 2 Z are close in the Z space, the corresponding images x1; x2 2 X are visually similar.

given a real image x, find a  latent representation z*, which could generate an image x’ and is close to x
$$
\mathbf{z}^{*}=\underset{\mathbf{z}}{\arg \min} \ \ell(G(\mathbf{z}), x)
$$


- 



inverse problem:

deblurring, image inpainting, phase retrieval,



**为什么要做 GAN inversion**

给一张真实的原图，找到对应生成模型的 z*，保证生成的假图能和真实原图一致。在实现了这样的前提下，就可以对这张图做更多的操作，这是过去inversion的意义。

resemble real data， be applicable to real image editing without requiring ad-hoc supervision or expensive optimization.

**是怎么找 z* 的**



There are two main approaches to embed instances from the image space to the latent space

- learning based: learn an encoder (AE) 
  $$
  \theta_{E}^{*}=\underset{\theta_{E}}{\arg \min } \sum_{n} \mathcal{L}\left(G\left(E\left(x_{n} ; \theta_{E}\right)\right), x_{n}\right)
  $$
  

- optimization based: select a random initial latent code and optimize it using gradient descent







还可以怎么来做呢？

GAN在这个问题里面，



## Literature

**GAN Inversion A Survey**



原理上的 2013 Signal recovery from pooling representations

All of these inverting efforts are instances of the pre-image problem, [The pre-image problem in kernel methods]()



Compressed Sensing using Generative Models



2017 Precise Recovery of Latent Vectors from Generative Adversarial Networks

2016 ECCV Generative visual manipulation on the natural image manifold.



[Inverting the generator of a generative adversarial network](https://arxiv.org/pdf/1611.05644.pdf)  
**[`NeurIPSW 2016`] (`Imperial College London`)**  
*Antonia Creswell, Anil Anthony Bharath*

[Generative visual manipulation on the natural image manifold](https://arxiv.org/pdf/1609.03552.pdf)  
**[`ECCV 2016`]**  
*Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros*

[Inverting The Generator of A Generative Adversarial Network](https://arxiv.org/pdf/1611.05644.pdf)  
*Antonia Creswell, Anil Anthony Bharath*  
**[`NIPS 2016`] (`ICL`)** 

[How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/pdf/1904.03189.pdf)  
*Rameen Abdal, Yipeng Qin, Peter Wonka*  
**[`ICCV 2019`] (`KAUST`)**	[[Code](https://github.com/NVlabs/stylegan)]



[Image Processing Using Multi-Code GAN Prior](https://arxiv.org/pdf/1912.07116.pdf)  
*Jinjin Gu, Yujun Shen, Bolei Zhou*  
**[CVPR 2020] (CUHK)**






<span id="Pixel2Style2Pixel"></span>
[Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/pdf/2008.00951.pdf)  
**[`CVPR 2021`]**  
*Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, Daniel Cohen-Or*

