# <p align=center>`Embedding | Decoder`</p>

> Due to the lack of Github support for LaTeX math formulas, it is recommended that you can download it and view it locally with your own Markdown editor (like Typora, VSCode).



---

[Unsupervised learning of object landmarks by factorized spatial embeddings](https://arxiv.org/pdf/1705.02193.pdf)  
**[`ICCV 2017`] (`Oxford`)**  
*James Thewlis, Hakan Bilen, Andrea Vedaldi*

<details><summary>Click to expand</summary>


<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210509151055.png" width="400" />
    <p>(point r in the reference space S, a map $Phi$ detects the location q 两幅图是同一物体不同视角)</p>
</div>


> **Summary**

Detect consistent landmarks with image deformations under a process of factorizing viewpoint.

They mainly learn viewpoint-independent representations of objects from images,  the structure of an object is expressed as a set of landmark points. The landmark can be seen as a representation of transformations. 

利用 变形前后关键点不变性 作为学习信号。

？这个关键点是如何定义的呢，人还是自动的

> **Details**

$S \subset \mathbb{R}^{3}$ is the surface of a physical object, independent of the particular image $\mathrm{x}$  
$\mathrm{x}: \Lambda \rightarrow \mathbb{R}$ is an image of the object  
$\Lambda \sub \mathbb{R}^2$  is the image domain

learn a function $q = \Phi_S(p;\mathrm{x})$, where $p \in S$ is the object points, $q \in \Lambda$ is the corresponding pixels.  
an image warp function $g: \mathbb{R}^2 \mapsto \mathbb{R}^2$. 主要是依靠viewpoint的变换

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

[Self-supervised learning of a facial attribute embedding from video](https://arxiv.org/pdf/1808.06882.pdf)  
**[`BMVC 2018`] (`Oxford`)**  
*Olivia Wiles, A. Sophia Koepke, Andrew Zisserman*

<details><summary>Click to expand</summary>


The aim is to train a network to learn an embedding that encodes facial attributes in a self supervised manner, without any labels.

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



