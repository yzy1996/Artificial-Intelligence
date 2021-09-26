# <p align=center>`Mediator | Midium Template`</p>





task of Canonical Surface Mapping (CSM)  also pixel to 3D 

> Specifically, given an image, we learn to map pixels on the object to their corresponding locations on an abstract 3D model of the category.



how to utilize the right data

one way is to collect large-scale labeled data

we can label hundreds or thousands of keypoints per image for thousands of images.

collecting such labeled data requires enormous manual labeling effort, making it difficult to scale to generic categories.



有一种是提取特征的方法，受限于同一个物体 no change in the visible content



关于数据，有

2D 图像 带 pose

known pose for real images 

相同的物体照片

他们需要看的是 



Learning invariant (to viewpoint transformation or motion) representations. pixel-wise embedding invariant to certain transforms



% Category-Specific 3D Reconstruction
% - Learning category-specific mesh reconstruction from image collections
% - Category-specific object reconstruction from a single image



- [Canonical Surface Mapping via Geometric Cycle Consistency](https://arxiv.org/pdf/1907.10043.pdf)  
  **[`ICCV 2019`] (`CMU, Facebook`)**  
  *Nilesh Kulkarni, Abhinav Gupta, Shubham Tulsiani*

- [Articulation-aware Canonical Surface Mapping](https://arxiv.org/pdf/2004.00614.pdf)  
  **[`CVPR 2020`] (`UM, CMU, Facebook`)**  
  *Nilesh Kulkarni, Abhinav Gupta, David F. Fouhey, Shubham Tulsiani*





### Consistency as Meta-Supervision

**volumetric reconstruction**

NIPS 2016 Unsupervised learning of 3d structure from images.

CVPR 2017 Multi-view supervision for single-view reconstruction via differentiable ray consistency

nips 2016 Perspective transformer nets: Learning single view 3d object reconstruction without 3d supervision

**depth prediction**

use geometric consistency between the predicted 3D and available views as supervision.



CVPR 2017 Unsupervised learning of depth and ego-motion from video

CVPR 2015 Flowweb: Joint image set alignment by weaving consistent,
pixel-wise correspondences