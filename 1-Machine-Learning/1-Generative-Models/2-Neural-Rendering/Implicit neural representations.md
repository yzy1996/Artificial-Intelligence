# Implicit neural representations

引入2D表征，2D图像的表征里似乎只有像素一种，



为什么要表征3D





对象是什么：

3D object shape representations: surface-based, volume-based

but they are all coordinate-based 



Implicit neural representation helps learn a continuous function which can achieve reconstructions at any arbitrary resolution.



> neural rendering 和它的关系是什么呢？



become more and more popular in the 3D vision community due to their **compactness** and strong representation power. 



> 和Deep Implicit Function (DIFs) 有什么关系



## Definition







Neural representations is a compact representation for 3D shapes.



**concrete content**

implicit surface 

fully object reconstruction for incomplete 3D point cloud data or depth scans.



coordinate-based neural networks are used to represent other low-dimensional signals (2D images)



Issue:

standard ReLU MLPs fail to adequately represent fine details in these complex low-dimensional signals due to a spectral bias

Solution:

- replace the ReLU activations with sine functions
- lift the input coordinates into a Fourier feature space 



**prototype model**



signed distance functions (SDF)

occupancy networks



## Application









#### teamplates

polygon mesh-beased 

implicit templates



application in texture transfer, shape analysis and so on.

> 能很好地表征单个个体的形状，但无法建立不同物体之间的联系
>
> 前面有mesh templates可以做到
>
> 这样实现的好处是可以理解semantic relationship, 进而 help understanding and editing



怎么来做呢

第一部分：对每一类型有一个平均的形状

第二部分：一个变形场变形





为什么有价值，重要的

3D model reconstruction is for ...

matching, manipulation and understanding

可以用来干什么？

有哪些任务task吧？

human digitization [Pixel-aligned implicit function for high-resolution clothed human digitization]





## Literature









---

[Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://arxiv.org/pdf/2012.02189.pdf)

**[`Arxiv 2020`]**	**(`UCB`)**	

**[`Matthew Tancik`, `Ben Mildenhall`, `Terrance Wang`, `Divi Schmidt`, `Pratul P. Srinivasan`, `Jonathan T. Barron`, `Ren Ng`]**

<details><summary>Click to expand</summary><p>


> **Summary**





> **Details**

A given signal $T$ mapping from a set $C \in \mathbb{R}^d \rightarrow \mathbb{R}^n$

A coordinate-based neural representation $f_{\theta}$ for $T$ 

Known direct pointwise observations $\{\mathbf{x}_i, T(\mathbf{x}_i)\}$
$$
\begin{aligned}
L(\theta) = \sum_i\| f_{\theta}(\mathbf{x}_i)-T(\mathbf{x}_i) \|_2^2 \\
\theta_{i+1} = \theta_i - \alpha \nabla L(\theta_i)
\end{aligned}
$$
However, we usually can not access direct observations of T, only indirect observations are available.

For example, if $T$ is a 3D object, $M(T,\mathbf{p})$ could be a 2D image captured of the object from camera pose $\mathbf{p}$.
$$
L_M(\theta) = \sum_i\| M(f_{\theta},\mathbf{p}_i)-M(T, \mathbf{p}_i) \|_2^2
$$
given a fixed budget of $m$ optimization steps, different initial weight values $\theta_0$ will result in different final weights $\theta_m$ and signal approximation error $L(\theta_m)$.



given a dataset of observations of signals $T$ from a particular distribution $\mathcal{T}$ 
$$
\theta_{0}^{*}=\arg \min _{\theta_{0}} E_{T \sim \mathcal{T}}\left[L\left(\theta_{m}\left(\theta_{0}, T\right)\right)\right]
$$




</p></details>

---


