# 2D shape correspondences

> correspondences 统一指代 shape 的
>
> 这个笔记主要关注 (1) 别人做了什么，怎么做；(2) 我们和别人有什么不一样

---

首先这里和3D的区别在哪里：这里的2D，3D是指的输入是几维的，因此是说我们这里关心的是通过输入图片来学习形状相关性，至于是找2D图像之间的，还是还原到3D模型之间的都可以。而形状相关性主要有以下几个类别的应用：

- detection (keypoint and detection)
- segmentation
- ~~classification~~

> **为什么这个问题值得被研究?**

应用多 generic framework for: texture transfer \ pose and animation transfer \ statistical shape analysis \ 多视角识别



前面的谓语有用：Inference, 



### Overview

- 《A survey on shape correspondence》 2010

- (3D) 《Recent advances in shape correspondence》 2020

> Shape correspondence problem is stated as finding a set of corresponding points between given shapes

SIFT or SURF



To learn 2D shape correspondences, 

To We can assign correspondences with the same colors

correspondence betweeen two sets of points in space

different view direction with fixed correspondences



% 对比直接建立在2D层面的相关性以及建立在3D层面的相关性

Compared with directly learning correspondence maps from 2D images, learning from 3D structures as an intermediate medium is more powerful. 



很相关的一篇文章：Semantic Correspondence via 2D-3D-2D Cycle

They first predict 3D structures from a single image and then project 3D semantic labels





利用2D-2D

可以看成是一种matching

match local feature 提取像素点的特征，然后做匹配，既可以通过学习变形的function，也可以通过学习encoder压缩到一个低维共性点



learn parametric warping to related images

- Warpnet: Weakly supervised matching for singleview reconstruction 单视图重建
- Convolutional neural network architecture for geometric matching 通过扭曲变形，让两幅图关键点重合
- End-to-end weakly-supervised semantic alignment

learn equivariant embeddings for matching

- Unsupervised learning of object frames by dense equivariant image labelling 可以不是一个物体
- Self-supervised learning of a facial attribute embedding from video 
- Unsupervised learning of object landmarks by factorized spatial embeddings 可以不是一个物体
- Unsupervised learning of landmarks by descriptor vector exchange 可以不是一个物体
- Self-supervised learning of interpretable keypoints from unlabelled videos



they use multi-view or synthetic data to generate supervision

### Detection







### Segmentation

对于相关性而言，都已经知道相关性了，one-shot标注后直接就迁移到了新的object上了。传统方法主要是依靠手动标记，所以重点找一下不需要手动标记的方法。

FiG-NeRF: Figure-Ground Neural Radiance Fields for 3D Object Category Modelling

> 因为解耦了前景和背景，所以能很自然地做分割
>
> They disentangle foreground and background latent codes and compute a foreground segmentation mask by rendering the depth of the foreground and background models. (Specially) they tackle the difficulty where the background occludes the foreground by compute an amodal segmentation by thresholding the accumulated foreground density.











#### 查的过程中看到的3D相关的

经常提到：descriptor

> To learn dense correspondences between 3D 

Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence 2020

Dense Human Body Correspondences Using Convolutional Networks 2016

3d-coded: 3d correspondences by deep deformation 2018

Unsupervised Learning of Dense Shape Correspondence 2019

Deep Functional Maps: Structured Prediction for Dense Shape Correspondence

FAUST: Dataset and evaluation for 3D mesh registration

Unsupervised cycle-consistent deformation for shape matching





<img src="C:\Users\zhiyuyang4\AppData\Roaming\Typora\typora-user-images\image-20210526160159027.png" alt="image-20210526160159027" style="zoom:33%;" />