# GAN Inversion

which means embedding/mapping a given image into latent space.



## Introduction

There are two main approaches to embed instances from the image space to the latent space

- learning based: learn an encoder (VAE) 
- optimization based: select a random initial latent code and optimize it using gradient descent



inverse problem:

deblurring, image inpainting, phase retrieval,







## Literature

**GAN Inversion A Survey**



原理上的 2013 Signal recovery from pooling representations

All of these inverting efforts are instances of the pre-image problem, [The pre-image problem in kernel methods]()



Compressed Sensing using Generative Models



2017 Precise Recovery of Latent Vectors from Generative Adversarial Networks

2016 ECCV Generative visual manipulation on the natural image manifold.

2016 Inverting The Generator Of A Generative Adversarial Network

---

### Image2StyleGAN

[How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/pdf/1904.03189.pdf)  
**[`ICCV 2019`] (`KAUST`)**	[[Code](https://github.com/NVlabs/stylegan)]
*Rameen Abdal, Yipeng Qin, Peter Wonka*

<details><summary>Click to expand</summary><p>

> **Summary**

They propose an embedding algorithm to map a given image into the latent space of StyleGAN pre-trained on the FFHQ dataset. This embedding enables semantic image editing operations that can be applied to existing photographs. They show results for *image morphing*, *style transfer*, and *expression transfer*.

> **Details**

<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210110163352.png" alt="image-20210110163352567" style="zoom:50%;" />

</p></details>

---

[Inverting the generator of a generative adversarial network](https://arxiv.org/pdf/1611.05644.pdf)  
**[`NeurIPSW 2016`] (`Imperial College London`)**  
*Antonia Creswell, Anil Anthony Bharath*

[Generative visual manipulation on the natural image manifold](https://arxiv.org/pdf/1609.03552.pdf)  
**[`ECCV 2016`]**  
*Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros*



Image processing using multi-code gan prior






<span id="Pixel2Style2Pixel"></span>
[Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/pdf/2008.00951.pdf)  
**[`CVPR 2021`]**  
*Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, Daniel Cohen-Or*

