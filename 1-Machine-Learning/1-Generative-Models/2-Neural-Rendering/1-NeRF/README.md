# Neural Radiance Fields Notes



## Introduction



**Computer Graphics** (CG) is a branch of computer science that deals with **generating images** with the aid of computers. 



3D reconstruction from multiple images: this tech is to predict the ①**depth** from ②**length** and ③**breadth**.

We try to predict a function for depth determination at various points in the image against the object itself.

Here comes the Neural Radiance Fields.



the larger field of *Neural rendering* is defined by the [excellent review paper by Tewari et al.](https://arxiv.org/abs/2004.03805) as

> “deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.”



**Neural volume rendering** refers to methods that generate <u>images</u> or <u>video</u> by tracing a ray into the scene and taking an integral of some sort over the length of the ray. Typically a neural network like a multi-layer perceptron (MLP) encodes a function from the 3D coordinates on the ray to quantities like density and color, which are integrated to yield an image.



**A radiance fields**  is a continuous function $f$ which maps a 3D point $\mathbf{x} \in \mathbb{R}^3$ and a viewing direction $\mathbf{d} \in \mathbb{S}^2$ to a volume density $\sigma \in \mathbb{R}^+$ and an RGB color value $\mathbf{c} \in \mathbb{R}^3$. 



NeRF uses a neural network to map a 3D location $\mathbf{x} \in \mathbb{R}^3$ and a viewing direction $\mathbf{d} \in \mathbb{R}^3$ to a volume density $\sigma_\theta(\mathbf{x}) \in \mathbb{R}^+$ and a color value $c_\theta(\mathbf{x}, \mathbf{d}) \in \mathbb{R}^3$.

这样写的好处是清晰了sigma和 c 是有什么决定的，配上那个图就清晰了



NeRF带来的好处是什么呢？

Conditioning on the viewing direction $\mathbf{d}$ allows for modeling view-dependent effects such as specular reflections and improves reconstruction quality in case the Lambertian assumption is violated.



不需要目标的mask，

While NeRF does not require object masks for training due to its volumetric radiance representation, extracting the scene geometry from the volume density requires careful tuning of the density threshold and leads to artifacts due to the ambiguity present in the density field,

再怎么渲染呢







## Trick

**(1) positional encoding**

Low dimensional input needs to be mapped to higher-dimensional features to be able to represent complex signals when $f$ is parameterized with a neural network. Specifically, we element-wise apply a pre-defined **positional encoding** to each component of $\mathbf{x}$ and $\mathbf{d}$.
$$
\gamma(t, L) = \left(\sin(2^0t\pi), \cos(2^0t\pi), \dots, \sin(2^{L}t\pi), \cos(2^{L}t\pi)\right),
$$
where $t$ is a scalar input, and $L$ the number of frequency octaves.



**(2) SIREN**



## QA

> Why not use a convolutional layer?

They are linear relation.



## Dataset

commonly-used single object datasets, Photoshape and image collections

- Chairs
- Cats
- CelebA
- CelebA-HQ
- 

more challenging single-object

CompCars

LSUN Churches

FFHQ



## Literature

[NeRF](#NeRF)

[NeRF++](#NeRF++)

[UNISURF](#UNISURF)

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
**[`ECCV 2020`]** **(`UCB, UCSD`)** **[[Code-Tensorflow](https://github.com/bmild/nerf)]** **[[Code-PyTorch](https://github.com/yenchenlin/nerf-pytorch)]** **[[Code-PyTorch](https://github.com/krrish94/nerf-pytorch)]** **([Page](https://www.matthewtancik.com/nerf))**  
*[`Ben Mildenhall`, `Pratul P. Srinivasan`, `Matthew Tancik`, `Jonathan T. Barron`, `Ravi Ramamoorthi`, `Ren Ng`]*

<details><summary>Click to expand</summary>


![image-20201204115352659](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201204115352.png)

> **First You should know**

The NeRF training procedure relies on the fact that given a 3D scene, two intersecting rays from two different cameras should yield the same color.

> **Summary**

Synthesize novel views of complex scenes from a sparse set of input views. Optimize an underlying continuous volumetric scene function. We aim to model geometry and appearance of complex real scenes from multi-view unstructured flash images. Neural Reflectance Fields are a continuous function neural representation that **implicitly models both scene geometry and reflectance**. represent by a deep multi-layer perceptron (MLP)

> **Pipeline**

Input a single continuous 5D coordinate - spatial location $$ (x, y, z) $$ and viewing direction $(\theta, \phi)$





> **Details**

learn this Neural Radiance Fields by parameterizing $$f$$ with a multi-layer perceptron (MLP):
$$
\begin{aligned}
f_{\theta}: \mathbb{R}^{L_{x}} \times \mathbb{R}^{L_{\mathrm{d}}} & \rightarrow \mathbb{R}^{+} \times \mathbb{R}^{3} \\
(\gamma(\mathbf{x}), \gamma(\mathbf{d})) & \mapsto(\sigma, \mathbf{c})
\end{aligned}
$$




rendering



</details>

---


### GRAF

[Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2007.02442.pdf)  
**[`NeurIPS 2020`]** **(`MPI`)** **[[Code](https://github.com/autonomousvision/graf)]**  
*[`Katja Schwarz`, `Yiyi Liao`, `Michael Niemeyer`, `Andreas Geiger`]*

<details><summary>Click to expand</summary>


![image-20210108153435365](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210108153442.png)

> **Summary**



> **Details**

camera matrix $$\mathbf{K}$$

camera pose $$\mathbf{\xi}$$

2D sampling pattern $$\nu$$



shape code $$\mathbf{z}_s \in \mathbb{R}^m$$

appearance code $$\mathbf{z}_a \in \mathbb{R}^n$$


$$
\begin{aligned}
g_{\theta}: \mathbb{R}^{L_{\mathbf{x}}} \times \mathbb{R}^{L_{\mathbf{d}}} \times \mathbb{R}^{M_{s}} \times \mathbb{R}^{M_{a}} & \rightarrow \mathbb{R}^{+} \times \mathbb{R}^{3} \\
\left(\gamma(\mathbf{x}), \gamma(\mathbf{d}), \mathbf{z}_{s}, \mathbf{z}_{a}\right) & \mapsto(\sigma, \mathbf{c})
\end{aligned}
$$




</details>

---

### GIRAFFE

[Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/pdf/2011.12100.pdf)

**[`arxiv 2020`]**	**(`MPI`)**	

**[`Michael Niemeyer`, `Andreas Geiger`]**

<details><summary>Click to expand</summary>


![image-20210109152339076](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210109152339.png)

> **Summary**

disentangle individual objects and allows for translating and rotating them in the scene as well as changing the camera pose.

controllable images synthesis without additional supervision

Our key hypothesis is that incorporating a compositional 3D scene representation into the generative model leads to more controllable image synthesis

> **Details**

$$
\begin{aligned}
h_{\theta}: \mathbb{R}^{L_{\mathbf{x}}} \times \mathbb{R}^{L_{\mathbf{d}}} \times \mathbb{R}^{M_{s}} \times \mathbb{R}^{M_{a}} & \rightarrow \mathbb{R}^{+} \times \mathbb{R}^{M_{f}} \\
\left(\gamma(\mathbf{x}), \gamma(\mathbf{d}), \mathbf{z}_{s}, \mathbf{z}_{a}\right) & \mapsto(\sigma, \mathbf{f})
\end{aligned}
$$



</details>

---

### pi-GAN

[Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2012.00926.pdf)

**[`arxiv 2020`]**	**(`Stanford`)**	

**[`Eric R. Chan`, `Marco Monteiro`, `Petr Kellnhofer`, `Jiajun Wu`, `Gordon Wetzstein`]**

<details><summary>Click to expand</summary>


> **Summary**

Synthesize high-quality view consistent images a SIREN-based 3D representation 

Using a method of combining sinusoidal representation networks and neural radiance fields.



multi-view consistency



> **Details**

First represent 3D object 



Density and color are defined as:
$$
\begin{align}
\sigma(\mathbf{x}) &=\mathbf{W}_{\sigma} \Phi(\mathbf{x})+\mathbf{b}_{\sigma}, \\
\mathbf{c}(\mathbf{x}, \mathbf{d}) &=\mathbf{W}_{c} \phi_{c}\left([\Phi(\mathbf{x}), \mathbf{d}]^{T}\right)+\mathbf{b}_{c},
\end{align}
$$


</details>

---

### GRF

[GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering](https://arxiv.org/pdf/2010.04595.pdf)

**[`ICLR 2021`]**	**(`Stanford`)**	**[[Code](https://github.com/alextrevithick/GRF)]**

**[`Alex Trevithick`, `Bo Yang`]**

<details><summary>Click to expand</summary>


> **Summary**



</details>

---

### Non-Rigid Neural Radiance Fields

Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video

**[[Code](https://github.com/facebookresearch/nonrigid_nerf)]**

<details><summary>Click to expand</summary>


![Pipeline figure](https://github.com/facebookresearch/nonrigid_nerf/raw/master/misc/pipeline.png)



</details>

---

### DietNeRF

[Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis](https://arxiv.org/pdf/2104.00677.pdf)  
**[`Arxiv`] (`Berkeley`)**  
*Ajay Jain, Matthew Tancik, Pieter Abbeel*

<details><summary>Click to expand</summary>


> Why the name?





> Key point

can be estimated from only a few photos and can generate views with unobserved regions



> 如何做到的呢？

在传统NeRF的基础上，加了一个**sematic consistency loss**。后者来自于**CLIP's Vision Transformer**，用来衡量真样本，和新生成的不同pose下的是不是同一个object。



> 为什么这样可以做到？

没有prior knowledge的话，是很难学到没有见过的物体的。所以考虑借助一个pre-trained image encoder来guide



> Details

$$
\mathcal{L}_{\mathrm{SC}, \ell_{2}}(I, \hat{I})=\frac{\lambda}{2}\|\phi(I)-\phi(\hat{I})\|_{2}^{2}
$$



</details>

---

### NeRF--



</details>

---

<span id="UNISURF"></span>
[UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction](https://arxiv.org/pdf/2104.10078.pdf)  
**[`arxiv`] (`MPI`)**  
*Michael Oechsle, Songyou Peng, Andreas Geiger*

<details><summary>Click to expand</summary>

> **Summary**

Combine **volumetric radiance** (powerful for "unsupervised" coarse scene) and **surface rendering** (accurate reconstruction).

<div align="center">
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210428214156.png" alt="image-20210428214154651" width="500">
</div>

> **Details**



</details>

---

