# Introduction

> Problem

Modeling a real scene from captured images and reproducing its appearance under novel conditions.

recover scene geometry and reflectance

surface reconstruction

> Related technology

3D reconstruction & inverse rendering

> new break

combines neural scene representations with classical ray marching - a volume rendering approach that is naturally differentiable



- implicit model
- explicit model





The problem of learning discriminative 3D models from 2D images

3D properties such as camera viewpoint or object pose

最终要的还是2D照片，但学到的是3D表征，最后用一个相机固定2D视角，因此可以生成多个角度的图像

implicit or explicit

learn model for **single** or **multiple** objects.



Firstly, you need to know 

- [ ] [marching cubes]()

NV 

SRN

**NeRF (Neural Radiance Fields)**



## Explanation

marching cubes



## Literature



### NeRF

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)

**[`ECCV 2020`]**	**(`UCB, UCSD`)**	**[[Code-Tensorflow](https://github.com/bmild/nerf)]**	**[[Code-PyTorch](https://github.com/yenchenlin/nerf-pytorch)]**	**[[Code-PyTorch](https://github.com/krrish94/nerf-pytorch)]**	**([Page](https://www.matthewtancik.com/nerf))**

**[`Ben Mildenhall`, `Pratul P. Srinivasan`, `Matthew Tancik`, `Jonathan T. Barron`, `Ravi Ramamoorthi`, `Ren Ng`]**

<details><summary>Click to expand</summary>
dddd

dddddd

dddd

</details>

---

### Meshlet

[Meshlet priors for 3D mesh reconstruction](https://arxiv.org/pdf/2006.03997.pdf)

**`[NeurIPS 2020]`**	**`(EPFL)`**	**`[Edoardo Remelli, Pascal Fua]`**	**[[Code](https://github.com/cvlab-epfl/MeshSDF)]**

<details><summary>Click to expand</summary><p>


</p></details>

---


### PatchNets

[PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations](https://arxiv.org/pdf/2008.01639.pdf)

**`[ECCV 2020]`**	**`(MPI, Facebook)`**	**`[Edgar Tretschk, Christian Theobalt]`**	**[[Code]()]**	**[[Slides](http://gvv.mpi-inf.mpg.de/projects/PatchNets/data/patchnets_slides.pdf)]**

<details><summary>Click to expand</summary><p>



</p></details>

---

### DeepSDF

[Deepsdf: Learning continuous signed distance functions for shape representation](https://arxiv.org/pdf/1901.05103.pdf)

**`[CVPR 2019]`**	**`(UW, MIT, Facebook)`**	**`[Jeong Joon Park, Peter Florence]`**	**[[Code](https://github.com/facebookresearch/DeepSDF)]**

<details><summary>Click to expand</summary><p>


</p></details>

---

[Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis](https://arxiv.org/abs/1912.05237)

**`[CVPR 2020]`**	**`(Max Planck Institute)`**	**`[Yiyi Liao, Katja Schwarz]`**	**[[Code](https://github.com/autonomousvision/controllable_image_synthesis)]**

<details><summary>Click to expand</summary><p>


![image-20201214211146939](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201214211210.png)



> **Summary**



> **Method**






</p></details>

---




