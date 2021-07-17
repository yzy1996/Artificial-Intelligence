# <p align=center>`Single View Reconstruction`</p>



## Contents

[ShapeHD](#ShapeHD)

---

<span id="ShapeHD"></span>
[Learning Shape Priors for Single-View 3D Completion and Reconstruction](https://arxiv.org/pdf/1809.05068.pdf)  
**[`ECCV 2018`] (`MIT`)**  
*Jiajun Wu, Chengkai Zhang, Xiuming Zhang, Zhoutong Zhang, William T. Freeman, Joshua B. Tenenbaum*

<details><summary>Click to expand</summary><p>

<div align=center><img width="600" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210714105831.png"/></div>

> **Summary**

They just propose a penalty term to improve the quality of 3D reconstruction by integrating deep generative models with adversarially learned shape priors. After training the model, they can achieve single-view 3D shape completion and reconstruction.

> **Details**

Their model consists of three components:

- (encoder-decoder) 2.5D sketch estimator to predict the object's depth, surface normals, and silhouette from an RGB image.
- (encoder-decoder) 3D shape estimator to predict a 3D shape in the canonical view from 2.5D sketches.
- (discriminator from pre-trained GAN) deep naturalness model to penalize the shape estimator

</p></details>

---

