# <p align=center>`KeyPoints/Landmarks` </p>

Both 2D and 3D keypoint detection are long-standing problems in computer vision. 

> A set of keypoints representing any object (**shape/structure**) is important for **geometric reasoning**, due to their simplicity and ease of handling. [^ intro2]

> Keypoints-based methods have been crucial to the success of many vision applications. Examples include: 3D reconstruction, registration, human body pose, recognition, and generation. [^ intro2]

Conventional works define keypoints manually or learn from supervised examples, automatically discovering them from unlabeled data (**unsupervised**) is what we need.

The keypoints should be **geometrically** and **semantically** consistent across viewing angles and instances of an object category.

The model we learn often covers a collection of objects of **a specific category**.

## Introduction

**name**: landmark/keypoint detector/discovery/estimation



内容：找到同一类别里 不受视角影响的 几何结构，语义一致 的关键点，

The meaning of predicting keyPoints/landmarks?

for shape edit？

invariant to pose, shape, and illumination

a good proxy for shape editing [^KeypointDeformer]



伴随着这个的做法，同时还做了什么呢

pose estimation



分类和识别工作同时伴随着的是特征提取，那么在geometric vision 领域，比如 3D reconstruction and shape alignment 是不是也伴随着又一个 keypoint detection module ，然后是 geometric reasoning,



希望达到的效果：

- **Equivariance**: equivariant to image transformation, including object and camera motions.



现在可以做到

- 2D/3D数据输入
- 监督和无监督，这里的监督指的是特征点标记

- 一个模型涵盖同一类物体



概述目前主要的方法有哪些：



直接利用，借用keypoint的工作：

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



$$

> **Related words**:

landmark, parts, skeletons, category-specific

## Data

annotated keypoints for:

- face [^ face]

- hands [^ hand]

- human bodies [^ body1] [^ body2]



评价指标

可以手动标然后做回归





related works:

the structure of objects can be also described as constituent parts.

---


用的是 discovery ，是不是也可以用 detection

2D Unsupervised keypoint discovery 

2017 Unsupervised learning of object landmarks by factorized spatial embeddings

2018 Unsupervised discovery of object landmarks as structural representations

2018 Self-supervised learning of a facial attribute embedding from video

2018 Unsupervised learning of object landmarks through conditional image generation

2019 Unsupervised learning of landmarks by descriptor vector exchange

2020 Self-supervised learning of interpretable keypoints from unlabelled videos



3D keypoints on 2D image

- 2018 Discovery of latent 3d keypoints via end-to-end geometric reasoning

> 用的 3D pose 的信息



3D keypoints on 3D shapes.

- 2021 KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control
- 2020 Unsupervised learning of intrinsic structural representation points
- 2020 Unsupervised learning of category-specific symmetric 3d keypoints from point sets





价值意义

robotics applications need 3D keypoints for control 

- 2019 Keypoint affordances for category-level robotic manipulation
- 2019 kpam-sc: Generalizable manipulation planning using keypoint affordance and shape completion



根据这些特征点，可以做识别



## Literature

6-dof object pose from semantic keypoints

3d landmark model discovery from a registered set of organic shapes

## Supervised

2013 Deep convolutional network cascade for facial point detection

2014 Facial landmark detection by deep multi-task learning

2016 Deep deformation network for object landmark localization

2017 Simultaneous facial landmark detection, pose and deformation estimation under facial occlusion

## Unsupervised



[input dimension] to [output dimension]

### 2D to 2D 

[KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://arxiv.org/pdf/2104.11224.pdf)  
**[`CVPR 2021`] (`Oxford, UCB, Stanford`)**  
*Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa*

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

[Convolutional experts constrained local model for 3d facial landmark detection](https://arxiv.org/pdf/1611.08657.pdf)
**[`CVPR-W 2017`] (`CMU`)**  
*Amir Zadeh, Tadas Baltrušaitis, Louis-Philippe Morency*

Keypoint recognition using randomized trees

Deep Deformation Network for Object Landmark Localization

### 2D to 3D

[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://arxiv.org/pdf/1807.03146.pdf)  
**[`NeurIPS 2018`] (`Google`)**  
*Supasorn Suwajanakorn, Noah Snavely, Jonathan Tompson, Mohammad Norouzi*



### 3D to 3D

[Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets](https://arxiv.org/pdf/2003.07619.pdf)  
**[`ECCV 2020`] (`ETH`)**  
*Clara Fernandez-Labrador, Ajad Chhatkuli, Danda Pani Paudel, Jose J. Guerrero, Cédric Demonceaux, Luc Van Gool*

Usip: Unsupervised stable interest point detection from 3d point clouds



## For different domain

human bodies

DeepPose: Human pose estimation via deep neural networks

Stacked hourglass networks for human pose estimation

Articulated pose estimation with flexible mixtures-of-parts

Cascaded pose regression



bird

Deep deformation network for object landmark localization

Part Localization using Multi-Proposal Consensus for Fine-Grained Categorization

Bird part localization using exemplar-based models with enforced pose and subcategory consistency



furniture 

Single Image 3D Interpreter Network.



[^ intro2]: Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets
[^KeypointDeformer]: KeypointDeformer
[^ face]: 300 faces in-the-wild challenge: Database and results
[^ hand]: Real-time continuous pose recovery of human hands using convolutional networks
[^ body1]: 2D human pose estimation: New benchmark and state of the art analysis
[^ body2]: Microsoft COCO: Common objects in context

