# GAN for 3D Understanding







## Introduction





generative 3D face model

- FLAME
- 





















The problem of learning discriminative 3D models from 2D images

3D properties such as camera viewpoint or object pose

最终要的还是2D照片，但学到的是3D表征，最后用一个相机固定2D视角，因此可以生成多个角度的图像

implicit or explicit

learn model for **single** or **multiple** objects.



## Literature



- [RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis](https://arxiv.org/pdf/1909.12573.pdf)  
  **[`ICLR 2020`] (`U Tokyo, RIKEN`)**  
  *Atsuhiro Noguchi, Tatsuya Harada*

- [Do 2D GANs Know 3D Shape? Unsupervised 3D shape reconstruction from 2D Image GANs](https://arxiv.org/pdf/2011.00844.pdf)  
  **[`ICLR 2021`] (`CUHK, NTU`)**  
  *Xingang Pan, Bo Dai, Ziwei Liu, Chen Change Loy, Ping Luo*

- [Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering](https://arxiv.org/pdf/2010.09125.pdf)  
  **[`ICLR 2021`] (NVIDIA, Toronto)**  
  *Yuxuan Zhang, Wenzheng Chen, Huan Ling, Jun Gao, Yinan Zhang, Antonio Torralba, Sanja Fidler*









<details><summary>Click to expand</summary>
<p>A keyboard. </p>
<p>A keyboard. </p>
</p>> summary </p>
dfdf <br>
sss


sss 
</details>























### Meshlet

[Meshlet priors for 3D mesh reconstruction](https://arxiv.org/pdf/2006.03997.pdf)

**`[NeurIPS 2020]`**	**`(EPFL)`**	**`[Edoardo Remelli, Pascal Fua]`**	**[[Paper](https://github.com/cvlab-epfl/MeshSDF)]**

<details><summary>Click to expand</summary><p>



</p></details>

---


### PatchNets

[PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations](https://arxiv.org/pdf/2008.01639.pdf)

**`[ECCV 2020]`**	**`(MPI, Facebook)`**	**`[Edgar Tretschk, Christian Theobalt]`**	**[[Paper]()]**	**[[Slides](http://gvv.mpi-inf.mpg.de/projects/PatchNets/data/patchnets_slides.pdf)]**

<details><summary>Click to expand</summary><p>



</p></details>

---

### DeepSDF

[Deepsdf: Learning continuous signed distance functions for shape representation](https://arxiv.org/pdf/1901.05103.pdf)

**`[CVPR 2019]`**	**`(UW, MIT, Facebook)`**	**`[Jeong Joon Park, Peter Florence]`**	**[[Paper](https://github.com/facebookresearch/DeepSDF)]**

<details><summary>Click to expand</summary><p>



</p></details>

---

### BlockGAN

[BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images](https://arxiv.org/abs/2002.08988)

**`[NeurIPS 2020]`**	**`(Adobe)`**	**`[Thu Nguyen-Phuoc, Christian Richardt]`**	**[[Paper](https://github.com/thunguyenphuoc/BlockGAN)]**

<details><summary>Click to expand</summary>

<div align=center><img width="600" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201214151442.png"/></div>

> **Summary**

learns 3D object-oriented scene representations directly from unlabeled 2D images

> **Method**

divide an 3D feature into background and foreground

a noise vector $\mathbb{z}_i$ and the object's 3D pose $\theta_i = (s_i, \mathbf{R}_i, \mathbf{t}_i)$

3D feature $O_i = g_i(\mathbb{z}_i, \theta_i)$
$$
\mathbf{x}=p\left(f(\underbrace{O_{0},}_{\text {background }} \underbrace{O_{1}, \ldots, O_{K}}_{\text {foreground }})\right)
$$

</details>

---

[Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis](https://arxiv.org/abs/1912.05237)

**`[CVPR 2020]`**	**`(Max Planck Institute)`**	**`[Yiyi Liao, Katja Schwarz]`**	**[[Paper](https://github.com/autonomousvision/controllable_image_synthesis)]**

<details><summary>Click to expand</summary><p>


![image-20201214211146939](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201214211210.png)



> **Summary**



> **Method**






</p></details>

---



### GIF

[Generative Interpretable Faces](https://arxiv.org/pdf/2009.00149.pdf)

**[`3DV 2020`]**	**(`MPI`)**	**[[Code](https://github.com/ParthaEth/GIF)]**

**[`Partha Ghosh`, `Pravir Singh Gupta`, `Roy Uziel`, `Anurag Ranjan`, `Michael Black`, `Timo Bolkart`]**

<details><summary>Click to expand</summary><p>


> **Summary**





> **Details**



</p></details>

---



$\sub$

