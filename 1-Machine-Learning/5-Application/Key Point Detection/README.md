# <p align=center>`KeyPoints/Landmarks` </p>

Both 2D and 3D keypoint detection are long-standing problems in computer vision. 

> A set of keypoints representing any object (**shape/structure**) is important for **geometric reasoning**, due to their simplicity and ease of handling. [^ intro2]

> Keypoints-based methods have been crucial to the success of many vision applications. Examples include: 3D reconstruction, registration, human body pose, recognition, and generation. [^ intro2]

Conventional works define keypoints manually or learn from supervised examples, automatically discovering them from unlabeled data (**unsupervised**) is what we need.

The keypoints should be **geometrically** and **semantically** consistent across viewing angles and instances of an object category.

The model we learn often covers a collection of objects of **a specific category**.

## Introduction

**name**: landmark/keypoint detection/discovery/estimation

> 思考这个和 pose estimation 的关系？pose也是由一个个关键点构成的，



**先笼统地介绍：**

- 关键点很重要：因为可以看成是物体的一种最简洁形状表征，就可以用来形状编辑，重建，识别等；所以如何找关键点是一个很重要的研究问题。同时分类和识别工作同时伴随着的是特征提取，那么在geometric vision 领域，比如 3D reconstruction and shape alignment 是不是也伴随着有一个 keypoint detection module ，然后再是 geometric reasoning。

- 关键点的特点 - 不随视角，光线，形状变化，姿态 而变化

  **Equivariance**: equivariant to image transformation, including object and camera motions.

- 关键点检测的拓展：姿态估计



**现在可以做到：**

- 2D/3D数据输入
- 监督和无监督，这里的监督指的是特征点标记
- 一个模型涵盖同一类物体



**再详细根据分类介绍**

- 根据对象domain划分

> **Facial keypoints** (facial landmarks): often defined manually with coordinates (x, y). These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc.
>
> 不过跟我们关系不大
>
> <div align=center>
> 	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511113228.jpg" width="100" />
> </div>



**概述目前主要的方法有哪些**：

[todo]



直接利用/借用keypoint的工作：

**Non-Rigid Structure-from-Motion (NRSfM)** methods ref:

- Multiview aggregation for learning category-specic shape reconstruction
- Symmetric non-rigid structure from motion for category-specific object structure estimation

> The key idea is that a large number of object deformations can be explained by linearly combining a smaller K number of basis shapes at some pose. 对刚体而言，只有一个基础形状，秩为3。

里面用来对**形状变形建模**的主要方法有：

- low-rank shape prior 
  - A simple prior-free method for non-rigid structure-from-motion factorization
  - Recovering non-rigid 3D shape from image streams (鼻祖)
  - Nonrigid structure-from-motion: Estimating shape and motion with hierarchical priors
  - Nonrigid structure from motion in trajectory space
- isometric prior 
  - Non-rigid structure from locally-rigid motion
  - Isometric non-rigid shape-from-motion in linear time



**Related keywords**:

landmark, parts, skeletons, category-specific

keypoint heatmap: 关键点热力图，图中数值越大的位置，越有可能是关键点



**未来可以做的**

现在都是对单个物体进行关键点检测，当场景中同时有多个物体呢？一种是先进行单个物体的标记，再套用但物体的检测；一种是对所有的part加标注，最后再判断是否属于同一个物体。



**评价指标**

可以手动标然后做回归



**Something related**

the structure of objects can be also described as constituent parts.



## Data

annotated keypoints for:

- face [^ face]

- hands [^ hand]

- human bodies [^ body1] [^ body2]



## Literature

有物体上的，也有人体上的，人体上又分为脸部和躯体

有针对2D数据的，也有针对3D数据的

因此我根据以上进行分类划分

---

值得还没读完的

- OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

  提到了一个叫 Part Affinity Fields (PAFs) 的方法
  
- [CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints](https://arxiv.org/pdf/2105.02343.pdf)

  有一个这样的实验图

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511175347.png" width="400" />
</div>

- 6-dof object pose from semantic keypoints

- 3d landmark model discovery from a registered set of organic shapes



## Supervised

[Simultaneous facial landmark detection, pose and deformation estimation under facial occlusion](https://arxiv.org/pdf/1709.08130.pdf)  
**[`CVPR 2017`] (`Rensselaer Polytechnic Institute`)**  
*Yue Wu, Chao Gou, Qiang Ji*

[Deep Deformation Network for Object Landmark Localization](https://arxiv.org/pdf/1605.01014.pdf)  
**[`ECCV 2016`] (`NEC`)**  
*Xiang Yu, Feng Zhou, Manmohan Chandraker*

[Facial landmark detection by deep multi-task learning](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)  
**[`ECCV 2014`] (`CUHK`)**  
*Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang*

[Deep Convolutional Network Cascade for Facial Point Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf)  
**[`CVPR 2013`] (`CUHK`)**  
*Yi Sun, Xiaogang Wang, Xiaoou Tang*

## Unsupervised

> [input dimension] to [output dimension]

### 2D to 2D 

[Self-supervised learning of interpretable keypoints from unlabelled videos](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.pdf)  
**[`CVPR_2020`] (`Oxford`)**  
*Tomas Jakab, Ankush Gupta, Hakan Bilen, Andrea Vedaldi*

[Unsupervised learning of landmarks by descriptor vector exchange](https://arxiv.org/pdf/1908.06427.pdf)  
**[`ICCV 2019`] (`Oxford`)**  
*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi*

[Teacher supervises students how to learn from partially labeled images for facial landmark detection](https://arxiv.org/pdf/1908.02116.pdf)  
**[`ICCV 2019`] (`SUST`)**  
*Xuanyi Dong, Yi Yang*

[Self-supervised learning of a facial attribute embedding from video](https://arxiv.org/pdf/1808.06882.pdf)  
**[`BMVC 2018`] (`Oxford`)**  
*Olivia Wiles, A. Sophia Koepke, Andrew Zisserman*

[Unsupervised learning of object landmarks through conditional image generation](https://arxiv.org/pdf/1806.07823.pdf)  
**[`NeurIPS 2018`] (`Oxford`)**  
*Tomas Jakab, Ankush Gupta, Hakan Bilen, Andrea Vedaldi*

[Unsupervised discovery of object landmarks as structural representations](https://arxiv.org/pdf/1804.04412.pdf)  
**[`CVPR 2018`] (`Michigan`)**  
*Yuting Zhang, Yijie Guo, Yixin Jin, Yijun Luo, Zhiyuan He, Honglak Lee*

[Unsupervised learning of object landmarks by factorized spatial embeddings](https://arxiv.org/pdf/1705.02193.pdf)  
**[`ICCV 2017`] (`Oxford`)**  
*James Thewlis, Hakan Bilen, Andrea Vedaldi*

### 2D to 3D

[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://arxiv.org/pdf/1807.03146.pdf)  
**[`NeurIPS 2018`] (`Google`)**  
*Supasorn Suwajanakorn, Noah Snavely, Jonathan Tompson, Mohammad Norouzi*

[Implicit 3D Orientation Learning for 6D Object Detection from RGB Images](https://arxiv.org/pdf/1902.01275.pdf)  
**[`ECCV 2018`] (`German Aerospace Center, TUM`)**  
*Martin Sundermeyer, Zoltan-Csaba Marton, Maximilian Durner, Manuel Brucker, Rudolph Triebel*

### 3D to 3D

[KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://arxiv.org/pdf/2104.11224.pdf)  
**[`CVPR 2021`] (`Oxford, UCB, Stanford`)**  
*Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa*

[Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets](https://arxiv.org/pdf/2003.07619.pdf)  
**[`ECCV 2020`] (`ETH`)**  
*Clara Fernandez-Labrador, Ajad Chhatkuli, Danda Pani Paudel, Jose J. Guerrero, Cédric Demonceaux, Luc Van Gool*

[Unsupervised learning of intrinsic structural representation points](https://arxiv.org/pdf/2003.01661.pdf)  
**[`CVPR 2020`] (`HKU, MPI`)**  
*Nenglun Chen, Lingjie Liu, Zhiming Cui, Runnan Chen, Duygu Ceylan, Changhe Tu, Wenping Wang*

[Convolutional experts constrained local model for 3d facial landmark detection](https://arxiv.org/pdf/1611.08657.pdf)
**[`CVPR-W 2017`] (`CMU`)**  
*Amir Zadeh, Tadas Baltrušaitis, Louis-Philippe Morency*

[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://arxiv.org/pdf/1904.00229.pdf)  
**[`ICCV 2019`] (`NUS`)**  
*Jiaxin Li, Gim Hee Lee*

## For different domain

### human bodies

[Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/pdf/1704.07809.pdf)  
**[`CVPR 2017`] (`CMU`)**  
*Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh*

[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf)  
**[`ECCV 2016`] (`Michigan`)**  
*Alejandro Newell, Kaiyu Yang, Jia Deng*

[Cascaded hand pose regression](https://openaccess.thecvf.com/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)  
**[`CVPR 2015`] (`CUHK`)**  
*Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang, Jian Sun*

[DeepPose: Human pose estimation via deep neural networks](https://arxiv.org/pdf/1312.4659.pdf)  
**[`CVPR 2014`] (`Google`)**  
*Alexander Toshev, Christian Szegedy*

[Articulated pose estimation with flexible mixtures-of-parts](https://www.cs.cmu.edu/~deva/papers/pose2011.pdf)  
**[`CVPR 2011`] (`UCI`)**  
*Yi Yang, Deva Ramanan*

[Cascaded pose regression](https://authors.library.caltech.edu/23201/1/Dollar2010p133332008_Ieee_Conference_On_Computer_Vision_And_Pattern_Recognition_Vols_1-12.pdf)  
**[`CVPR 2010`] (`CIT`)**  
*Piotr Dollár, Peter Welinder, Pietro Perona*

### Bird

Deep Deformation Network for Object Landmark Localization

Part Localization using Multi-Proposal Consensus for Fine-Grained Categorization

Bird part localization using exemplar-based models with enforced pose and subcategory consistency

### Furniture

[Single Image 3D Interpreter Network](https://arxiv.org/pdf/1604.08685.pdf)  
**[`ECCV 2016`] (`MIT`)**  
*Jiajun Wu, Tianfan Xue, Joseph J. Lim, Yuandong Tian, Joshua B. Tenenbaum, Antonio Torralba, William T. Freeman*



## 价值意义

robotics applications need 3D keypoints for control 

- 2019 Keypoint affordances for category-level robotic manipulation
- 2019 kpam-sc: Generalizable manipulation planning using keypoint affordance and shape completion



[^ intro2]: Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets
[^KeypointDeformer]: KeypointDeformer
[^ face]: 300 faces in-the-wild challenge: Database and results
[^ hand]: Real-time continuous pose recovery of human hands using convolutional networks
[^ body1]: 2D human pose estimation: New benchmark and state of the art analysis
[^ body2]: Microsoft COCO: Common objects in context

