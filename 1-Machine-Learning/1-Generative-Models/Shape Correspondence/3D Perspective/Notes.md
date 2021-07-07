> Due to the lack of Github support for LaTeX math formulas, it is recommended that you can download it and view it locally with your own Markdown editor (like Typora, VSCode).



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

[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://arxiv.org/pdf/1904.00229.pdf)  
**[`ICCV 2019`] (`NUS`)**  
*Jiaxin Li, Gim Hee Lee*

<details><summary>Click to expand</summary>


<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210511225110.png" width="800" />
</div>


> **Summary**



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

