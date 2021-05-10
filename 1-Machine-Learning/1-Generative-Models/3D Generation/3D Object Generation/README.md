[liao2020unsupervised](#liao2020unsupervised)

<span id="liao2020unsupervised"></span>[Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis](https://arxiv.org/pdf/1912.05237.pdf)  
**[`CVPR 2020`]** **(`MPI`)**  
*Yiyi Liao, Katja Schwarz, Lars Mescheder, Andreas Geiger*

<details><summary>Click to expand</summary><p>


> **Summary**

In this process, 3D supervision is hard to acquire,  



> **Details**

![image-20210428110836239](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210428110844.png)


$$
g_{\theta}^{3D}: \mathbf{z} \mapsto \{\mathbf{o}_{bg}, \mathbf{o}_1, \dots, \mathbf{o}_N\}
$$

$$
g_{\theta}^{2 D}: \mathbf{X}_{i}, \mathbf{A}_{i}, \mathbf{D}_{i} \mapsto \mathbf{X}_{i}^{\prime}, \mathbf{A}_{i}^{\prime}, \mathbf{D}_{i}^{\prime}
$$



Differentiable projection:

features map $\mathbf{X}_i \in \mathbb{R}^{W \times H \times F}$, initial alpha map $\mathbf{A}_i \in \mathbb{R}^{W \times H}$, initial depth map $\mathbf{D}_i \in \mathbb{R}^{W \times H}$.

> **Loss**

`Adversarial Loss` + `Compactness Loss` + `Geometric Consistency Loss`



</p></details>

--

