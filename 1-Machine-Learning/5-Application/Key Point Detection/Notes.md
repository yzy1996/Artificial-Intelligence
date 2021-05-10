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

### 3D 

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

