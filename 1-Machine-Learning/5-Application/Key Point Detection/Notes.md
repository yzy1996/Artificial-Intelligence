# <p align=center>`KeyPoints/Landmarks` </p>

Due to the lack of Github support for LaTeX math formulas, it is recommended that you can download it and view it locally with your own Markdown editor.

---

[KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://arxiv.org/pdf/2104.11224.pdf)  
**[`CVPR 2021`] (`Oxford, UCB, Stanford`)**  
*Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210506170726.png" width="800" />
</div>

> **Summary** 

The problem is to align a source 3D object to a target 3D object from the same object category. Deform the source shape and preserve the shape details.

The method is using the shape deformation algorithm to discover 3D keypoints and control the shape via comparing the latent representations.

目的是用户可交互式形状编辑，较为直接和容易的方式就是以关键点为参考，关键点也是带有语义信息的。

本文做到了：1. 完全无监督的方式，用户可以通过改变关键点的位置对图片进行带有语义的变形；2. 类别先验带来的对称性结构理解 (用的PCA)


> **Details**

The loss function is a similarity loss and a keypoint regularization loss (semantically consistent & well-distributed, preserve shape symmetries).

learn two parts:

- a keypoint predictor $\phi: \boldsymbol{x} \mapsto \boldsymbol{p}$
- a conditional deformation model on keypoints $\Psi:\left(\boldsymbol{x}, \boldsymbol{p}, \boldsymbol{p}^{\prime}\right) \mapsto \boldsymbol{x}^{\prime}$

**Dataset**: ShapeNet, KeypointNet, Google Scanned Objects dataset

> **Limitation**

does not allow object part rotation

</p></details>

---

[Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets](https://arxiv.org/pdf/2003.07619.pdf)  
**[`ECCV 2020`] (`ETH`)**  
*Clara Fernandez-Labrador, Ajad Chhatkuli, Danda Pani Paudel, Jose J. Guerrero, Cédric Demonceaux, Luc Van Gool*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210509174703.png" width="800" />
</div>

> **Summary**

(可以联想到2020CVPR另一篇利用对称性做生成的) reflective symmetry 反射对称

Using the symmetric liner basis shapes to learn keypoints directly from 3D point clouds.

They define the desired properties:

- generalizability over different shape instances and alignments in a category
- one-to-one ordered correspondences and semantic consistency
- representative of the shape as well as the category while preserving shape symmetry

做了两件事：1是对shape进行建模，2是在无序的点云中推断有序的关键点

> **Details**

Main method is the **low-rank symmetric shape basis**  
ref: A simple prior-free method for non-rigid structure-from-motion factorization, Recovering non-rigid 3D shape from image streams, Nonrigid structure-from-motion: Estimating shape and motion with hierarchical priors.

</p></details>

---

[Self-supervised learning of interpretable keypoints from unlabelled videos](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.pdf)  
**[`CVPR_2020`] (`Oxford`)**  
*Tomas Jakab, Ankush Gupta, Hakan Bilen, Andrea Vedaldi*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511173048.png" width="800" />
</div>

> **Summary**

Learn from only unlabeled videos and **a weak empirical prior** on the object poses. (为什么要用视频呢？因为视频帧之间的对象是同一个目标，只是有pose的差异，通过分析这些差异就可以对pose建模。)

The pose priors are obtained from unpaired data. (虽然强调跟训练pose的网络无关，但这也算是一种多余的先验输入辅助，还是有监督的，只不过是 little additional supervision。)

Introduce a **conditional generator** design combining **image translation**

> **Details**

用了 Adversarial loss，是通过新的 unpaired 数据 (看上图)。目标是希望 新生成的 和 真实的不匹配 一致 (比如都是轮廓样式)。

一个 AE loss，用的是 VGG 的 perceptual loss 
$$
\mathcal{L}_{\text {perc }}=\frac{1}{N} \sum_{i=1}^{N}\left\|\Gamma\left(\hat{x}_{i}\right)-\Gamma\left(\boldsymbol{x}_{i}\right)\right\|_{2}^{2},
$$
一个 difference adversarial loss：
$$
\mathcal{L}_{\mathrm{disc}}(D)=\frac{1}{M} \sum_{j=1}^{M} D\left(\overline{\boldsymbol{y}}_{j}\right)^{2}+\frac{1}{N} \sum_{i=1}^{N}\left(1-D\left(\boldsymbol{y}_{i}\right)\right)^{2}
$$

</p></details>

---

[Unsupervised learning of intrinsic structural representation points](https://arxiv.org/pdf/2003.01661.pdf)  
**[`CVPR 2020`] (`HKU, MPI`)**  
*Nenglun Chen, Lingjie Liu, Zhiming Cui, Runnan Chen, Duygu Ceylan, Changhe Tu, Wenping Wang*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511224355.png" width="800" />
</div>

> **Summary**

**Take 3D point cloud as input, output the structure points**



</p></details>

---

[Unsupervised learning of landmarks by descriptor vector exchange](https://arxiv.org/pdf/1908.06427.pdf)  
**[`ICCV 2019`] (`Oxford`)**  
*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511180002.png" width="400" />
</div>

> **Summary**

Develop a new perspective on **equivariance to random image transformations** method. Based on [previous work](Unsupervised learning of object landmarks by factorized spatial embeddings).

Introduce an **invariant descriptors** (例如[SIFT](Distinctive image features from scaleinvariant
keypoints)，就想是一个embedding) to establish correspondences between images which is the same as **landmark detectors**. In addition, landmarks are invariant to intra-class variations in addition to viewing effects. (看上图很好理解)

> **Details**

用了一个中间instance来增强变形的能力。

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511202103.png" width="800" />
</div>

</p></details>

---

[Teacher supervises students how to learn from partially labeled images for facial landmark detection](https://arxiv.org/pdf/1908.02116.pdf)  
**[`ICCV 2019`] (`SUST`)**  
*Xuanyi Dong, Yi Yang*

[Self-supervised learning of a facial attribute embedding from video](https://arxiv.org/pdf/1808.06882.pdf)  
**[`BMVC 2018`] (`Oxford`)**  
*Olivia Wiles, A. Sophia Koepke, Andrew Zisserman*

<details><summary>Click to expand</summary>

The aim is to train a network to learn an embedding that encodes facial attributes in a selfsupervised manner, without any labels.

</p></details>

---

[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://arxiv.org/pdf/1904.00229.pdf)  
**[`ICCV 2019`] (`NUS`)**  
*Jiaxin Li, Gim Hee Lee*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511225110.png" width="800" />
</div>

> **Summary**

ss

</p></details>

---

[Unsupervised learning of object landmarks through conditional image generation](https://arxiv.org/pdf/1806.07823.pdf)
**[`NeurIPS 2018`] (`Oxford`)**  
*Tomas Jakab, Ankush Gupta, Hakan Bilen, Andrea Vedaldi*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210508112737.png" width="800" />
</div>

> **Summary**

Generating image $\hat{\mathbf{x}}^\prime$ conditioned on ①the appearance of image $\mathbf{x}$ and ②the geometry of image $\mathbf{x}^\prime$. Just adopting a simple perceptual loss formulation.

Learn object landmarks from synthetic image deformations. Use image generation with the goal of learning landmarks.

Compared to other works, the advantage is the **simplicity** and **generality** of the formulation -> allow for more complex task e.g. highly-articulated human body.

> **Logic**

learn landmark-like representations  -> encode the geometry of the object 因为改变的就只有几何pose

> **Details**

$$
\min_{\Psi, \Phi} \mathbb{E}_{\mathrm{x}, \mathrm{x}^{\prime}} \left[\mathcal{L} \left(\mathrm{x}^{\prime}, \Psi\left(\mathrm{x}, \Phi\left(\mathrm{x}^{\prime}\right)\right)\right)\right]
$$

</p></details>

---

[Unsupervised discovery of object landmarks as structural representations](https://arxiv.org/pdf/1804.04412.pdf)  
**[`CVPR 2018`] (`Michigan`)**  
*Yuting Zhang, Yijie Guo, Yixin Jin, Yijun Luo, Zhiyuan He, Honglak Lee*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210508214918.png" width="800" />
</div>

> **Summary**

Using an **autoencoder** model to learn object structures. AE也就意味着有重建，会根据特征点去重建图像

The advantage is their performance is semantically meaningful and more predictive of manually annotated landmarks.

> **Details**

each landmark has a corresponding detector, outputs a detection score map with the detected
landmark located at the maximum.

</p></details>

---

[Unsupervised learning of object landmarks by factorized spatial embeddings](https://arxiv.org/pdf/1705.02193.pdf)  
**[`ICCV 2017`] (`Oxford`)**  
*James Thewlis, Hakan Bilen, Andrea Vedaldi*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210509151055.png" width="400" />
    <p>(point r in the reference space S, a map $Phi$ detects the location q)</p>
</div>

> **Summary**

Detect consistent landmarks with image deformations under a process of factorizing viewpoint.

They mainly learn viewpoint-independent representations of objects from images,  the structure of an object is expressed as a set of landmark points. The landmark can be seen as a representation of transformations. 

利用 变形前后关键点不变性 作为学习信号。

> **Details**

$S \subset \mathbb{R}^{3}$ is the surface of a physical object, independent of the particular image $\mathrm{x}$  
$\mathrm{x}: \Lambda \rightarrow \mathbb{R}$ is an image of the object  
$\Lambda \sub \mathbb{R}^2$  is the image domain

learn a function $q = \Phi_S(p;\mathrm{x})$, where $p \in S$ is the object points, $q \in \Lambda$ is the corresponding pixels.  
an image warp function $g: \mathbb{R}^2 \mapsto \mathbb{R}^2$.

The factorization is:
$$
\forall p \in S: \Phi_{S}(p ; \mathbf{x} \circ g)=g\left(\Phi_{S}(p ; \mathbf{x})\right).
$$
to deformable objects, they introduce a common reference space - object frame, using a reference points $r$, rewrite the function above:
$$
\forall r \in S_{0}: \Phi(r ; \mathbf{x} \circ g)=g(\Phi(r ; \mathbf{x})).
$$

</p></details>

---

[Convolutional experts constrained local model for 3d facial landmark detection](https://arxiv.org/pdf/1611.08657.pdf)
**[`CVPR-W 2017`] (`CMU`)**  
*Amir Zadeh, Tadas Baltrušaitis, Louis-Philippe Morency*

[Single Image 3D Interpreter Network](https://arxiv.org/pdf/1604.08685.pdf)  
**[`ECCV 2016`] (`MIT`)**  
*Jiajun Wu, Tianfan Xue, Joseph J. Lim, Yuandong Tian, Joshua B. Tenenbaum, Antonio Torralba, William T. Freeman*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511164624.png" width="800" />
</div>

> **Summary**

This work achieves state-of-the-art performance on both 2D keypoint estimation and 3D structure recovery. 有 2D annotations on real images.

</p></details>

---

[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://arxiv.org/pdf/1807.03146.pdf)  
**[`NeurIPS 2018`] (`Google`)**  
*Supasorn Suwajanakorn, Noah Snavely, Jonathan Tompson, Mohammad Norouzi*

<details><summary>Click to expand</summary>

<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210508160114.png" width="800" />
</div>

> **Summary**

**dubbed-"KeypointNet"**

learn category- specific 3D keypoints by solving an auxiliary task of rigid registration between multiple renders of the same shape and by considering the category instances to be pre-aligned.

from an end-to-end geometric reasoning framework, jointly optimize the keypoints.

also show these 3D keypoints can infer their depths without access to object geometry.

using aligned 3D and multiple 2D images with known pose.

</p></details>

---

