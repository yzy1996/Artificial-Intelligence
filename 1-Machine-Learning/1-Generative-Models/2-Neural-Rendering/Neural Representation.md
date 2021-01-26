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
