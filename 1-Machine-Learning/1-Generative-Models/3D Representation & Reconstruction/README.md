# <p align=center>`3D representation & reconstruction` </p>

A collection of resources on 3D representation and reconstruction from multi-view images.

> 表征是为了干什么，那肯定是为了重建，所以两者不分家。其中表征就是一个Encoder，重建就是一个Decoder。



## Table of Contents

- [3D Representation](#3D-Representation) (一些3D表征的方法)
  - [Point Cloud](./1-3D-Representation/Point-Cloud), [Meshes](./1-3D-Representation/Meshes), [Voxels](./1-3D-Representation/Voxels), Neural Implicit Functions (except NeRF)
- [Rendering](#Rendering) (有了表征当然就要有渲染)
  - [Surface Rendering](./2-Rendering/Surface-Rendering), [Volume Rendering](./2-Rendering/Volume-Rendering)

- [NeRF](#NeRF) (因为太火爆，所以给 NeRF related 开个专门的话题)
- [Neural Rendering](#Neural-Rendering) (这是一个很大的话题，因此把不好划分到上述的归类到这里)
  - [Inverse Rendering](./4-Neural-Rendering/Inverse-Rendering), [Relight](./4-Neural-Rendering/Relight), [Single View Reconstruction](./4-Neural-Rendering/Single-View-Reconstruction)
- [Tricks](#Tricks) 
  - SIREN
- [Datasets](#Datasets)



Survey



## Introduction

Our goal is to **reconstruct 3D objects or scenes** (geometry and appearance) from **single or multiple view 2D images**, by the means of one of the **3D representation methods** (e.g., point cloud, neural implicit function, surface),  with this comes **novel views synthesis** by **rendering**.

> 其实除了从 2D image 里学，输入数据还可以是PointCloud，或者RGB-D Images。



因为真实世界是3D的，所以我们希望能够表征和重建3D模型，另一方面3D表征带来的优势是 和视角无关的，这为机器人环境探索，行人重识别带来了便利。

这里我主要关注的是 neural implicit functions 这一类方法，它们是 coordinate-based neural models，因为建立了空间中的点到某一指标的映射关系；Occupancy Field 和 SDF 是映射到 Surface 值，NeRF 是映射到 不透明度和颜色。这一类方法具有的特点是：全空间连续可导 (可以用DL，分辨率可以无限大)，表征能力强大，占用内存小。

这些神经隐式表征方法还需要额外的渲染技术，渲染可以简单理解为“对3D模型拍个照得到2D图像”，复杂一点讲需要涵盖 cameras pose, lights, surface geometry and material 这么多因素。

通常学习的过程中很难做到单张图训练，学习到足够的先验信息后再通过逆向渲染做到对单张图的推断。在做的过程中，为了简化，我们也会使用canonical view / model。



## 3D Representation

首先划分为 是表征**单个物体** 还是表征**一个类别的物体**

其次划分为 不同的表征方法

但似乎是某些特定的方法专门研究出来表征一个类别物体的



当我们在谈论一个2D object 的时候，一幅图像的载体就是像素值 pixel，先不去想彩色图，甚至不是灰度图，而就是黑白图，也就是经过二值化后的灰度图。是不是就能知道这个object的形状。

而3D object，载体可以是体素值（类比像素值）voxel grids；为什么是可以是，因为还可以是point clouds，meshes。为什么呢，因为看到他们我们也可以知道这个object的形状呀。这些表征的特点是：**离散**，对complex geometry 的 fidelity **（保真度）高**。



### PointCloud

[A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://arxiv.org/pdf/1612.00603.pdf)  
**[`CVPR 2017`] (`Tsinghua, Stanford`)**  
*Haoqiang Fan, Hao Su, Leonidas Guibas*

[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)  
**[`CVPR 2017`] (`Stanford`)**  
*Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas*

[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)  
**[`NeurIPS 2017`] (`Stanford`)**  
Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas

[Large-scale point cloud semantic segmentation with superpoint graphs](https://arxiv.org/pdf/1711.09869.pdf)  
**[`CVPR 2018`] (`Universite Paris-Est`)**  
*Loic Landrieu, Martin Simonovsky*



### Mesh

[Pixel2Mesh Generating 3D Mesh Models from Single RGB Images](https://arxiv.org/pdf/1804.01654.pdf)  
**[`ECCV 2018`] (`Fudan, Princeton`)**  
*Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang*

[Meshlet Priors for 3D Mesh Reconstruction](https://arxiv.org/pdf/2001.01744.pdf)  
**[`CVPR 2020`] (`NVIDIA, UCSB`)**  
*Abhishek Badki, Orazio Gallo, Jan Kautz, Pradeep Sen*



### Voxels

优点：由像素直接上升到体素，很多2D的方法可以直接迁移过来

缺点：curse of dimensionality

[3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/pdf/1604.00449.pdf)  
**[`ECCV 2016`] (`Stanford`)**  
*Christopher B. Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, Silvio Savarese*

[Voxnet: A 3d convolutional neural network for real-time object recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)  
**[`IROS 2015`] (CMU)**  
*Daniel Maturana, Sebastian Scherer*

[Octnet: Learning deep 3d representations at high resolutions](https://arxiv.org/pdf/1611.05009.pdf)  
**[`CVPR 2017`] (`Graz University of Technology, MPI, ETH`)**  
*Gernot Riegler, Ali Osman Ulusoy, Andreas Geiger*

[Octnetfusion: Learning depth fusion from data](https://arxiv.org/pdf/1704.01047.pdf)  
**[`3DV 2017`] (`Graz University of Technology, MPI, ETH`)**  
*Gernot Riegler, Ali Osman Ulusoy, Horst Bischof, Andreas Geiger*



### Neural Implicit Function

先回顾一下在还没引入 Deep Learning 之前的历史，属于 classic multi-view stereo (**MVS**) methods. They mainly focus on either matching features across views or representing shapes with a voxel grid. The former  approaches need a complex pipeline rquiring additional steps like fusing depth information and meshing. The latter ones are limited to low resolution due to cubic memory requirements.

鉴于此，引入了 neural implicit representations, 完全连续，用起来简单，占用内存小。

虽然下面分类单独拎出来了，但其实也是属于这一大类下的。surface-based, volume-based



### Signed Distance Function

缺点：bad on sharp areas

<span id="IM-NET"></span>
[Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/pdf/1812.02822.pdf)  
**[`CVPR 2019`] (`Simon Fraser University`)**  
*Zhiqin Chen, Hao Zhang*

[Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/pdf/1812.03828.pdf)  
**[`CVPR 2019`] (`MPI, Google`)**  
*Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, Andreas Geiger*

[DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/pdf/1901.05103.pdf)  
**[`CVPR 2019`] (UW, MIT)**  
*Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove*



### Surface Reconstruction

优点：

缺点：

Regard the object surface as a 2-dimensional manifold embedded in the 3-dimensional space.

- [Analytic Marching: An Analytic Meshing Solution from Deep Implicit Surface Networks]()  
  **[`ICML 2020`] (`SCUT`)**  
  *Jiabao Lei, Kui Jia*

- [AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://arxiv.org/pdf/1802.05384.pdf)  
  **[`CVPR 2018`] (`LIGM, Adobe`)**  
  *Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, Mathieu Aubry*



> 这里补充一点是，可以用 Marching cubes 从SDF 得到Mesh



## Rendering

> for more details see [folder](./2-Rendering)

这里关心的是神经渲染这一类，因为要和神经表征去搭配。主要的方法是 rasterization and raytracing

- volume rendering

  integrate densities by drawing samples along the viewing rays

- surface rendering

  Differentiable volumetric rendering (DVR)

  Multiview neural surface reconstruction (IDR)



fast rendering

- [Fast Training of Neural Lumigraph Representations using Meta Learning](https://arxiv.org/pdf/2106.14942.pdf)  
  **[`Arxiv 2021`] (`Stanford`)**  
  Alexander W. Bergman, Petr Kellnhofer, Gordon Wetzstein



### Marching Cubes

[Deep marching cubes: Learning explicit surface representations](http://www.cvlibs.net/publications/Liao2018CVPR.pdf)  
**[`CVPR 2018`] (MPI, Zhejiang U)**  
*Yiyi Liao, Simon Donné, Andreas Geiger*

[Marching cubes: A high resolution 3D surface construction algorithm](https://people.eecs.berkeley.edu/~jrs/meshpapers/LorensenCline.pdf)  
**[`SIGGRAPH 1987`] (`General Electric Company`)**  
*William E. Lorensen, Harvey E. Cline*



## NeRF

> for more details see [folder](./3-NeRF)

### Enhance NeRF

data structures

- [PlenOctrees for real-time rendering of neural radiance fields](https://arxiv.org/pdf/2103.14024.pdf)  
  **[`Arxiv 2021`] (`UCB`)**  
  *Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa*
- [Baking neural radiance fields for real-time view synthesis](https://arxiv.org/pdf/2103.14645.pdf)  
  **[`Arxiv 2021`] (`Google`)**  
  *Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, Paul Debevec*

pruning

- [Neural sparse voxel fields](https://arxiv.org/pdf/2007.11571.pdf)  
  **[`NeurIPS 2020`] (`MPI`)**  
  *Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt*

importance sampling

- [DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks](https://arxiv.org/pdf/2103.03231.pdf)  
  **[`EGSR 2021`] (`Graz University of Technology`)**  
  *Thomas Neff, Pascal Stadlbauer, Mathias Parger, Andreas Kurz, Joerg H. Mueller, Chakravarty R. Alla Chaitanya, Anton Kaplanyan, Markus Steinberger*

fast integration

- [Autoint: Automatic integration for fast neural volume rendering](https://arxiv.org/pdf/2012.01714.pdf)  
  **[`CVPR 2021`] (`Stanford`)**  
  *David B. Lindell, Julien N. P. Martel, Gordon Wetzstein*



### NeRF + Surface

- [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction](https://arxiv.org/pdf/2104.10078.pdf)  
  **[`Arxiv 2021`] (`MPI`)**  
  *Michael Oechsle, Songyou Peng, Andreas Geiger*



## Neural Rendering

> for more details see [folder](./4-Neural-Rendering)
>
> neural rendering 包含了 NeRF，指的是用 神经表征和渲染 的一类方法，它是一类方法或者技术的总称。
>
> The goal of neural rendering is to project a 3D neural scene representation into one or multiple 2D images.
>
> 简单说neural rendering可以涵盖所有新的带neural的技术

### Single View Reconstruction (SVR)

> 从单张图像重建整个3D场景是很重要的一个话题，more details see [file](./Single-View-Reconstruction)
>
> 也可以叫 single image reconstruction，对一个 3D 目标，单个图像就是单个视角

- [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf)  
  **[`CVPR 2020`]  (`Oxford`)**  
  *Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi*
- [Learning Shape Priors for Single-View 3D Completion and Reconstruction](https://arxiv.org/pdf/1809.05068.pdf)  
  **[`ECCV 2018`] (`MIT`)**  
  *Jiajun Wu, Chengkai Zhang, Xiuming Zhang, Zhoutong Zhang, William T. Freeman, Joshua B. Tenenbaum*



## Tricks

> for more details see [folder](./5-Tricks) 

**Issue**:

standard ReLU MLPs fail to adequately represent fine details in these complex low-dimensional signals due to a spectral bias

**Solution**:

- replace the ReLU activations with sine functions
- lift the input coordinates into a Fourier feature space 



## Datasets

> for more details see [folder](./6-Datasets) 

- [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/pdf/1406.5670.pdf)  
  **[`CVPR 2015`] (`Princeton, CUH, MIT`)**  
  *Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, Jianxiong Xiao*
