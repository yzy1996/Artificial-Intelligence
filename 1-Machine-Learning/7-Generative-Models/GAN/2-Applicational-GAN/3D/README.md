# Introduction

The problem of learning discriminative 3D models from 2D images

3D properties such as camera viewpoint or object pose

最终要的还是2D照片，但学到的是3D表征，最后用一个相机固定2D视角，因此可以生成多个角度的图像





implicit or explicit





learn model for **single** or **multiple** objects.

# Literature

Contents



## BlockGAN

[BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images](https://arxiv.org/abs/2002.08988)

**`[NeurIPS 2020]`**	**`(Adobe)`**	**`[Thu Nguyen-Phuoc, Christian Richardt]`**	**([:memo:]())**	**[[:octocat:](https://github.com/thunguyenphuoc/BlockGAN)]**

<details><summary>Click to expand</summary><p>


![image-20201214151435632](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201214151442.png)



**Summary**

> learns 3D object-oriented scene representations directly from unlabeled 2D images



**Method**

> divide an 3D feature into background and foreground



> a noise vector $\mathbb{z}_i$ and the object's 3D pose $\theta_i = (s_i, \mathbf{R}_i, \mathbf{t}_i)$
>
> 
>
> 3D feature $O_i = g_i(\mathbb{z}_i, \theta_i)$

$$
\mathbf{x}=p\left(f(\underbrace{O_{0},}_{\text {background }} \underbrace{O_{1}, \ldots, O_{K}}_{\text {foreground }})\right)
$$



</p></details>

---

## 

[Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis](https://arxiv.org/abs/1912.05237)

**`[CVPR 2020]`**	**`(Max Planck Institute)`**	**`[Yiyi Liao, Katja Schwarz]`**	**([:memo:]())**	**[[:octocat:](https://github.com/autonomousvision/controllable_image_synthesis)]**

<details><summary>Click to expand</summary><p>


![image-20201214211146939](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201214211210.png)



**Summary**

> learns 3D object-oriented scene representations directly from unlabeled 2D images



**Method**

> divide an 3D feature into background and foreground
>
> a noise vector $\mathbb{z}_i$ and the object's 3D pose $\theta_i = (s_i, \mathbf{R}_i, \mathbf{t}_i)$
>
> 3D feature $O_i = g_i(\mathbb{z}_i, \theta_i)$

$$
\mathbf{x}=p\left(f(\underbrace{O_{0},}_{\text {background }} \underbrace{O_{1}, \ldots, O_{K}}_{\text {foreground }})\right)
$$



</p></details>

---





