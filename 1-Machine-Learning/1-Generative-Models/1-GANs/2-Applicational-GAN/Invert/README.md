# Latent Space Embedding



## Introduction



embed/map a given image into latent space



There are two main approaches to embed instances from the image space to the latent space

- learn an encoder (VAE)
- select a random initial latent code and optimize it using gradient descent



## Bibliography

[2019-Image2StyleGAN](Image2StyleGAN)

---

### Image2StyleGAN

[How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/pdf/1904.03189.pdf)

**[`ICCV 2019`]**	**(`KAUST`)**	**[[Code](https://github.com/NVlabs/stylegan)]**

**[`Rameen Abdal`, `Yipeng Qin`, `Peter Wonka`]**

<details><summary>Click to expand</summary><p>


> **Summary**

They propose an embedding algorithm to map a given image into the latent space of StyleGAN pre-trained on the FFHQ dataset. This embedding enables semantic image editing operations that can be applied to existing photographs. They show results for *image morphing*, *style transfer*, and *expression transfer*.



> **Details**

<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210110163352.png" alt="image-20210110163352567" style="zoom:50%;" />

</p></details>

---



[Inverting the generator of a generative adversarial network](https://arxiv.org/pdf/1611.05644.pdf)

NIPS 2016 Workshop Imperial College London

Antonia Creswell, Anil Anthony Bharath



---

[Generative visual manipulation on the natural image manifold](https://arxiv.org/pdf/1609.03552.pdf)

ECCV 2016

Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros