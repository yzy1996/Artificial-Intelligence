# Neural Radiance Fields Notes



## Introduction

the larger field of *Neural rendering* is defined by the [excellent review paper by Tewari et al.](https://arxiv.org/abs/2004.03805) as

> “deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.”



**Neural volume rendering** refers to methods that generate <u>images</u> or <u>video</u> by tracing a ray into the scene and taking an integral of some sort over the length of the ray. Typically a neural network like a multi-layer perceptron (MLP) encodes a function from the 3D coordinates on the ray to quantities like density and color, which are integrated to yield an image.



## Literature

[NeRF](#NeRF)

[NeRF++](#NeRF++)



**Generalization**

- GRAF
- GRF
- PixelNeRF
- Pi-GAN



**Lighting**

- NeRV

---

### NeRF

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)

**[`ECCV 2020`]**	**(`UCB, UCSD`)**	**[[Code-Tensorflow](https://github.com/bmild/nerf)]**	**[[Code-PyTorch](https://github.com/yenchenlin/nerf-pytorch)]**	**[[Code-PyTorch](https://github.com/krrish94/nerf-pytorch)]**	**([Page](https://www.matthewtancik.com/nerf))**

**[`Ben Mildenhall`, `Pratul P. Srinivasan`, `Matthew Tancik`, `Jonathan T. Barron`, `Ravi Ramamoorthi`, `Ren Ng`]**

<details><summary>Click to expand</summary>


![image-20201204115352659](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201204115352.png)

> **First You should know**

The NeRF training procedure relies on the fact that given a 3D scene, two intersecting rays from two different cameras should yield the same color.

> **Summary**

Synthesize novel views of complex scenes from a sparse set of input views. Optimize an underlying continuous volumetric scene function. We aim to model geometry and appearance of complex real scenes from multi-view unstructured flash images. Neural Reflectance Fields are a continuous function neural representation that **implicitly models both scene geometry and reflectance**. represent by a deep multi-layer perceptron (MLP)

> **Pipeline**

Input a single continuous 5D coordinate - spatial location ($x, y, z$) and viewing direction ($\theta, \phi$)

</details>

---