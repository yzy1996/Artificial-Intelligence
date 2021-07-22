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



Our goal is to **reconstruct 3D objects or scenes** from **single or multiple view 2D images**, by the means of one of the **3D representation methods** (e.g., point cloud, neural implicit function, surface),  with this comes **novel views synthesis**.





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



### Datasets

[3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/pdf/1406.5670.pdf)  
**[`CVPR 2015`] (`Princeton, CUH, MIT`)**  
*Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, Jianxiong Xiao*



### Neural Rendering

这是目前一个最新的3D表征的方向。see details in 




### Single View Reconstruction (SVR)

> 从单张图像重建整个3D场景是很重要的一个话题，more details ref [file](./Single-View-Reconstruction)
>
> 也可以叫 single image reconstruction，对一个 3D 目标，单个图像就是单个视角

- [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf)  
  **[`CVPR 2020`]  (`Oxford`)**  
  *Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi*

- [Learning Shape Priors for Single-View 3D Completion and Reconstruction](https://arxiv.org/pdf/1809.05068.pdf)  
  **[`ECCV 2018`] (`MIT`)**  
  *Jiajun Wu, Chengkai Zhang, Xiuming Zhang, Zhoutong Zhang, William T. Freeman, Joshua B. Tenenbaum*





### 3D representation



#### PointCloud

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



#### Mesh

[Pixel2Mesh Generating 3D Mesh Models from Single RGB Images](https://arxiv.org/pdf/1804.01654.pdf)  
**[`ECCV 2018`] (`Fudan, Princeton`)**  
*Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang*

[Meshlet Priors for 3D Mesh Reconstruction](https://arxiv.org/pdf/2001.01744.pdf)  
**[`CVPR 2020`] (`NVIDIA, UCSB`)**  
*Abhishek Badki, Orazio Gallo, Jan Kautz, Pradeep Sen*



#### Voxels

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



#### Signed Distance Function

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



#### Surface Reconstruction

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

#### Neural implicit function

neural radiance fields



### Marching Cubes

[Deep marching cubes: Learning explicit surface representations](http://www.cvlibs.net/publications/Liao2018CVPR.pdf)  
**[`CVPR 2018`] (MPI, Zhejiang U)**  
*Yiyi Liao, Simon Donné, Andreas Geiger*

[Marching cubes: A high resolution 3D surface construction algorithm](https://people.eecs.berkeley.edu/~jrs/meshpapers/LorensenCline.pdf)  
**[`SIGGRAPH 1987`] (`General Electric Company`)**  
*William E. Lorensen, Harvey E. Cline*







