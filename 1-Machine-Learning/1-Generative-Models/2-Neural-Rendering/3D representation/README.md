# 3D representation 



## Introduction

there is no canonical representation which is both computationally and memory efficient for high-resolution 3D representation 



volume-based 

surface-based 

> Pixels, voxels, and views: A study of shape representations for single view 3D object shape prediction



> In contrast to traditional multi-view stereo algorithms, learned models are able to encode rich prior information about the space of 3D shapes which helps to resolve ambiguities in the input. [Occupancy](#Occupancy)



computationally expensive since it requires many forward passes through the network for every pixel.



表征的目的：



尽可能花费少地

- 重建 reconstruction, 尽可能少的角度输入图像，最理想的是一张图片即可。a handful of views or ideally just one view.
- 直接对3D数据开展识别这样的task
- 





From a data structure point of view, a point cloud is an unordered set of vectors. While most work in deep learning focus on regular input representations like sequences, images and volumes.









现有的表征模型有：

- voxel grids

  large memory cost, limiting the output resolution

- point cloud

  表征是稀疏的, lack the connectivity structure and hence require additional post processing steps to extra 3D geometry from the model.

- mesh

  based on forming a template mesh 

  

- implicit field

continues, arbitrary resolution

 





典型论文以及

探讨一下优缺点





point cloud



---

[PointNet](#PointNet)

[PointNet++](#PointNet++)

[Occupancy](#Occupancy)



## Literature



### PointNet

[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

**[CVPR 2017]**	**(Stanford)**	**[[Code](https://github.com/charlesq34/pointnet)]**

**[`Charles R. Qi`, `Hao Su`, `Kaichun Mo`, `Leonidas J. Guibas`]**

<details><summary>Click to expand</summary><p>


<div align=center><img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210223205556.png" alt="image-20210223153047231" style="zoom:50%;" /></div>

> **Summary**

We design a deep learning framework that directly consumes **unordered** **point sets** as inputs and provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing.

> **Details**

**Symmetry Function for Unordered Input.** In order to make a model invariant to input permutation. Choose a symmetric function to aggregate the information from each point.
$$
f\left(\left\{x_{1}, \ldots, x_{n}\right\}\right) \approx g\left(h\left(x_{1}\right), \ldots, h\left(x_{n}\right)\right)
$$
where $h$ is a multi-layer perception network and $g$ is a composition of a single variable function and a max pooling function.

**Local and Global Information Aggregation.** 

**Joint Alignment Network.** The semantic labeling of a point cloud has to be invariant if the point cloud undergoes certain geometric transformations. We therefore expect that the learnt representation by
our point set is invariant to these transformations.

</p></details>

---

### PointNet++

[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)

**[NeurIPS 2017]**	**(Stanford)**

**[`Charles R. Qi`, `Li Yi`, `Hao Su`, `Leonidas J. Guibas`]**

<details><summary>Click to expand</summary><p>


> **Summary**



</p></details>

---

### Occupancy

Occupancy Networks: Learning 3D Reconstruction in Function Space

**[`CVPR 2019`]**	**[`MPI`]**

**[`Lars Mescheder`，`Andreas Geiger`]**

<details><summary>Click to expand</summary><p>


<div align=center><img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210223153049.png" alt="image-20210223153047231" style="zoom:50%;" /></div>

> **Summary**

Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier.



The key insight is that we represent the 3D object with a neural network that assigns to every location $\boldsymbol{p} \in \mathbb{R}^3$ an occupancy probability between 0 and 1. (just like a binary classification)



> **Details**

input object $x \in \mathcal{X}$; a query $\boldsymbol{p} \in \mathbb{R}^3$; output $s \in \mathbb{R}$.
$$
f_\theta : (\boldsymbol{p}, x) \mapsto o, ~~~~~~~~\text{where}~~ s \in [0, 1]
$$

for the $i$-th sample in a training batch we sample $K$ points $p_{ij} \in \mathbb{R}^3, j=1, \dots, K$. 

The train loss is:
$$
\mathcal{L}(\theta) = \sum_{i=1}^N \sum_{j=1}^K \mathcal{L}(f_\theta(p_{ij}, x_i), o_{ij})
$$
After training, we can extra an approximate isosurface:
$$
\{\boldsymbol{p} \in \mathbb{R}^3 | f_\theta(\boldsymbol{p}, x) = \tau\}
$$
我的疑问是，o的真值是怎么来的呢，如果输入是一张图片的话

</p></details>

---

