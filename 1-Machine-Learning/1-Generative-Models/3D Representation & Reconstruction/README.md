# <p align=center>`3D representation & reconstruction` </p>

## Contents



输入端的数据可以是：

- RGB Images
- RGB-D Images
- PointCloud
- PointCloud (Voxelized)



表征的方式一般有：

- Volumetric (OctNet)
- PointClouds (PointSetGen)
- Surfaces (AtlasNet)
- Signed Distance Function
- Geometeric Primitives



一个通过的流程框架是一个 Encoder Decoder



A collection of resources on 3D reconstruction.

3D 重建和 3D 表征是否是一个东西？

> 表征是为了干什么，那肯定是为了重建
>
> 表征有多种方式，传统较老的点云，体素，最新的nerf；有表面的，也有整个实体的
>
> 表征可以是表征单个物体，也可以是表征多个物体



通常为了简化，我们会使用canonical view / model



表征的方法有很多，

RGB-D 也是一种表征，



分类为显式的和隐式的



## Introduction

To tackle the instability of the training procedure...



These methods can be divided into two categories:

- ...



到了 space mapping



## Literature



### Survey



### 3D representation



PointCloud



#### Voxels

由像素直接上升到体素，很多2D的方法可以直接迁移过来

优点：

缺点：curse of dimensionality

[3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/pdf/1604.00449.pdf)  
**[`ECCV 2016`] (`Stanford`)**  
*Christopher B. Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, Silvio Savarese*



[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)  
**[] **  
*Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas*



#### Signed Distance Function

缺点：bad on sharp areas





#### Surface Reconstruction

Regard the object surface as a 2-dimensional manifold embedded in the 3-dimensional space.

- [Analytic Marching: An Analytic Meshing Solution from Deep Implicit Surface Networks]()  
  **[`ICML 2020`] (`SCUT`)**  
  *Jiabao Lei, Kui Jia*

- [AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://arxiv.org/pdf/1802.05384.pdf)  
  **[`CVPR 2018`] (`LIGM, Adobe`)**  
  *Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, Mathieu Aubry*



> 这里补充一点是，可以用 Marching cubes 从SDF 得到Mesh



### Single View Reconstruction (SVR)

> more details ref [file](./Single-View-Reconstruction)
>
> 也可以叫 single image reconstruction，对一个 3D 目标，单个图像就是单个视角

- [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf)  
  **[`CVPR 2020`]  (`Oxford`)**  
  *Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi*

- [Learning Shape Priors for Single-View 3D Completion and Reconstruction](https://arxiv.org/pdf/1809.05068.pdf)  
  **[`ECCV 2018`] (`MIT`)**  
  *Jiajun Wu, Chengkai Zhang, Xiuming Zhang, Zhoutong Zhang, William T. Freeman, Joshua B. Tenenbaum*

## Main Research Group

