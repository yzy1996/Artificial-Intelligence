# Shape Correspondence



<span id="DensePose"></span>
[DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/pdf/1802.00434.pdf)  
**[`CVPR 2018`] (`INRIA, Facebook`)**  
*Rıza Alp Güler, Natalia Neverova, Iasonas Kokkinos*

<details><summary>Click to expand</summary>

> Summary

establish dense correspondences from 2D images to surface-based representations of human body



> Details

UV mapping: 



Two-stage annotation process 

> Evaluation Measures

- Pointwise evaluation: Ratio of Correct Point (RCP)

  correct if geodesic distance is below a threshold. a curve $f(t)$. evaluate the area under the AUC.

- geodesic point similarity (GPS). inspired by object keypoint similarity (OKS)



Predict dense correspondences between image pixels and surface points through a fully-convolutional network.



> Training process





</p></details>

---

<span id="SMPL"></span>
[SMPL: A Skinned Multi-Person Linear Model](https://dl.acm.org/doi/pdf/10.1145/2816795.2818013)  
**[`TOG 2015`] (`IMPI`)**  
*Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, Michael J. Black*

<details><summary>Click to expand</summary>

> Summary



vertex-based

> Details

**Blend Skinning**

- Linear Blend Skinning (LBS) 线性混合蒙皮，使用最广泛，但是在关节处会产生不真实的变形
- dual-quaternion blend skinning (DQBS) 双四元数混合蒙皮

**Rigging**

​	building the relation between vertex and skeletal point

**Blend shapes**

​	the deformation of the base shape



**SMPL model**



N = 6839 vertices

K = 23 joints

pose parameter $\theta$

shape parameter $\beta$



</p></details>

---