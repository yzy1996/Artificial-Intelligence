# <p align=center>`Neural Radiance Fields` </p>

> 这一个笔记主要围绕NeRF相关展开，通过相关文献，整理从为什么NeRF会诞生，到NeRF还存在的问题。
>
> related link yenchen's [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF)



[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)  
**[`ECCV 2020`] (`UCB, UCSD`)**  
*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*

<div align="center">
<img width="700" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201204115352.png"/>
    <p>Figure 1</p>
</div>
[Code-Tensorflow](https://github.com/bmild/nerf)

[Code-PyTorch](https://github.com/yenchenlin/nerf-pytorch)

[Code-PyTorch](https://github.com/krrish94/nerf-pytorch)

spatial location $$ (x, y, z) $$ and viewing direction $(\theta, \phi)$

## Introduction



The NeRF training procedure relies on the fact that given a 3D scene, two intersecting rays from two different cameras should yield the same color.





**Computer Graphics** (CG) is a branch of computer science that deals with **generating images** with the aid of computers. 



3D reconstruction from multiple images: this tech is to predict the ①**depth** from ②**length** and ③**breadth**.

We try to predict a function for depth determination at various points in the image against the object itself.

Here comes the Neural Radiance Fields.



**A radiance fields**  is a continuous function $f$ which maps a 3D point $\mathbf{x} \in \mathbb{R}^3$ and a viewing direction $\mathbf{d} \in \mathbb{S}^2$ to a volume density $\sigma(\mathbf{x}) \in \mathbb{R}^+$ and an RGB color value $\mathbf{c}(\mathbf{x}, \mathbf{d}) \in \mathbb{R}^3$. 

![图片1](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210811231246.svg)

这样写的好处是清晰了sigma和 c 是有什么决定的，配上那个图就清晰了



NeRF带来的好处是什么呢？

view-independent, 因为x

Conditioning on the viewing direction $\mathbf{d}$ allows for modeling view-dependent effects such as specular reflections and improves reconstruction quality in case the Lambertian assumption is violated.



不需要目标的mask，

While NeRF does not require object masks for training due to its volumetric radiance representation, extracting the scene geometry from the volume density requires careful tuning of the density threshold and leads to artifacts due to the ambiguity present in the density field,

再怎么渲染呢





NeRF represents the 3D geometry and appearance of a scene as a continuous 5D to 2D mapping function and uses volume rendering to synthesize novel views. The training process relies on multiple images with given camera poses.



volume density does not admit accurate surface reconstruction



NeRF use volume rendering by learning alpha-compositing of a radiance field along rays.

带来的另一个好处是可解释性



high fidelity





我们的训练总是需要去regress一个目标的，DeepSDF regress a signed distance function, while NeRF regress 



A NeRF model stores a volumetric scene representation as the weights of an MLP, trained on many
images with known pose.





如何渲染的呢

integrating the density and color at regular intervals along each viewing ray.



<div align="center"><img width="500" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210722155353.png" ></div>


$$
\begin{aligned}
\hat{C}(\mathbf{r}) &=\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{\theta}\left(\mathbf{x}_{i}\right) \delta_{i}\right)\right) c_{\theta}\left(\mathbf{x}_{i}, \mathbf{d}\right) \\
T_{i} &=\exp \left(-\sum_{j<i} \sigma_{\theta}\left(\mathbf{x}_{j}\right) \delta_{j}\right)
\end{aligned}
$$



采样方式







我们可以作如下概括，

- 为何NeRF惊艳到了所有人
  - brutal simplicity (不讲道理地简单)，just an MLP taking in a 5D coordinate and outputting density and color

- 效果好的原因：
  - 神经网络强大，以及借助下面这些 tricks
  - periodic activation functions
  - positional encoding
  - stratified sampling scheme
- 存在的问题：
  - 训练和之后的渲染都很**慢**
  - 只能表征**静态**的场景
  - 依赖已知的**摄像头位置** (heavily rely on known camera pose)
  - 无法**泛化**到其他场景或目标 
  - 无法做到**少样本**训练 (fail to represent or synthesize with few instances)

（后续的文献工作也主要是为了解决这些问题）



## Table of Contents

- [1. Improve Performance](#Improve-Performance)
- [2. Shape Encode](Shape-Encode)
- [3. Dynamic & Deformable](#Dynamic-&-Deformable)
- [4. Composition](Composition)
- [5. Pose Estimation](#Pose-Estimation)



### 1. Improve Performance

**data structures**

- [PlenOctrees for real-time rendering of neural radiance fields](https://arxiv.org/pdf/2103.14024.pdf)  
  **[`Arxiv 2021`] (`UCB`)**  
  *Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa*
- [Baking neural radiance fields for real-time view synthesis](https://arxiv.org/pdf/2103.14645.pdf)  
  **[`Arxiv 2021`] (`Google`)**  
  *Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, Paul Debevec*

**pruning**

- [Neural sparse voxel fields](https://arxiv.org/pdf/2007.11571.pdf)  
  **[`NeurIPS 2020`] (`MPI`)**  
  *Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt*

**importance sampling**

- [DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks](https://arxiv.org/pdf/2103.03231.pdf)  
  **[`EGSR 2021`] (`Graz University of Technology`)**  
  *Thomas Neff, Pascal Stadlbauer, Mathias Parger, Andreas Kurz, Joerg H. Mueller, Chakravarty R. Alla Chaitanya, Anton Kaplanyan, Markus Steinberger*

**fast integration**

- [Autoint: Automatic integration for fast neural volume rendering](https://arxiv.org/pdf/2012.01714.pdf)  
  **[`CVPR 2021`] (`Stanford`)**  
  *David B. Lindell, Julien N. P. Martel, Gordon Wetzstein*



### 2. Shape Encode

- [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2007.02442.pdf)  
  **[`NeurIPS 2020`] (`MPI`)** [[Code](https://github.com/autonomousvision/graf)]  
  *Katja Schwarz, Yiyi Liao, Michael Niemeyer, Andreas Geiger*
- [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2012.00926.pdf)  
  **[`CVPR 2021`] (`Stanford`)**  
  *Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, Gordon Wetzstein*
- [pixelNeRF: Neural Radiance Fields from One or Few Images](https://arxiv.org/pdf/2012.02190.pdf)  
  **[`CVPR 2021`] (`UCB`)**  
  *Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa*
- [GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering](https://arxiv.org/pdf/2010.04595.pdf)  
  **[`ICCV 2021`] (`Williams, Oxford, PolyU`)**  
  *Alex Trevithick, Bo Yang*



### 3. Dynamic & Deformable

- [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/pdf/2011.12948.pdf)  
  **[`Arxiv 2020`] (`Washington, Google`)** [[Code](https://github.com/google/nerfies)]  
  *Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, Ricardo Martin-Brualla*
- [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/pdf/2011.13961.pdf)  
  **[`CVPR 2021`] (`CSIC-UPC, MPI`)**  
  *Albert Pumarola, Enric Corona, Gerard Pons-Moll, Francesc Moreno-Noguer*
- [Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction](https://arxiv.org/pdf/2012.03065.pdf)  
  **[`CVPR 2021`] (`Technical University of Munich, Facebook`)**  
  *Guy Gafni, Justus Thies, Michael Zollhöfer, Matthias Nießner*
- [Neural Scene Graphs for Dynamic Scenes](https://arxiv.org/pdf/2011.10379.pdf)  
  **[`CVPR 2021`] (`Algolux, Technical University of Munich`)**  
  *Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, Felix Heide*
- [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video](https://arxiv.org/abs/2012.12247)  
  **[`ICCV 2021`] (`MPI, Facebook`)** [[Code](https://github.com/facebookresearch/nonrigid_nerf)]  
  *Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, Christian Theobalt*



### 4. Composition

- [NeRF++: Analyzing and Improving Neural Radiance Fields](https://arxiv.org/pdf/2010.07492.pdf)  
  **[`Arxiv 2020`] (`Cornell Tech, Intel`)** [[Code](https://github.com/Kai-46/nerfplusplus)]  
  *Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun*

- [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/pdf/2011.12100.pdf)  
  **[`CVPR 2021`] (`MPI`)**   
  *Michael Niemeyer, Andreas Geiger*




### 5. Pose Estimation

Existing NeRF-based methods assume that the camera parameters are known. So it's better to train NeRF model without known camera poses. Even though there are some existing approaches (e.g. SfM) to pre-compute camera parameters.

也可以说解决的问题是，novel view synthesis from 2D images **without known camera poses**.

做到的效果包括：`一是有多准确`，`二是可变化范围有多大`

iNeRF and NeRF-- optimize camera pose along with other parameters when training NeRF.



- [iNeRF: Inverting Neural Radiance Fields for Pose Estimation](https://arxiv.org/pdf/2012.05877.pdf)  
  **[`IROS 2021`] (`MIT, Google`)**  
  *Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, Tsung-Yi Lin*

- [NeRF--: Neural Radiance Fields Without Known Camera Parameters](https://arxiv.org/pdf/2102.07064.pdf)  
  **[`Arxiv 2021`] (`Oxford`)** [[Code](https://github.com/ActiveVisionLab/nerfmm)]  
  *Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, Victor Adrian Prisacariu*

- [GNeRF: GAN-based Neural Radiance Field without Posed Camera](https://arxiv.org/pdf/2103.15606.pdf)  
  **[`Arxiv 2021`] (`ShanghaiTech`)**  
  *Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, Jingyi Yu*



补充：其实谈到相机位置估计，不可避免会和 Structure-from-Motion (SfM) 去比较，他们的开源包叫COLMAP：

- [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf)  
  **[`CVPR 2016`] (`UNC, ETH`)** [[Code](https://github.com/colmap/colmap)]  
  *Johannes L. Schonberger, Jan-Michael Frahm*



## Trick

**(1) positional encoding**

Low dimensional input needs to be mapped to higher-dimensional features to be able to represent complex signals when $f$ is parameterized with a neural network. Specifically, we element-wise apply a pre-defined **positional encoding** to each component of $\mathbf{x}$ and $\mathbf{d}$.
$$
\gamma(t, L) = \left(\sin(2^0t\pi), \cos(2^0t\pi), \dots, \sin(2^{L}t\pi), \cos(2^{L}t\pi)\right),
$$
where $t$ is a scalar input, and $L$ the number of frequency octaves.



**(2) SIREN**



## Q&A

> Why not use a convolutional layer?

They are linear relation.



## Dataset

commonly-used single object datasets, Photoshape and image collections

- Chairs
- Cats
- CelebA
- CelebA-HQ

more challenging single-object

CompCars

LSUN Churches

FFHQ





