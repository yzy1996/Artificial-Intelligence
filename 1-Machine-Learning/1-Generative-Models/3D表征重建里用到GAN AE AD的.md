本笔记重点关心 GAN AD AE 的点







很多都是和点云结合



Learning Representations and Generative Models for 3D Point Clouds  
**[`ICML 2018`] (`Stanford`)**  
*Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, Leonidas Guibas*





[Learning Progressive Point Embeddings for 3D Point Cloud Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Wen_Learning_Progressive_Point_Embeddings_for_3D_Point_Cloud_Generation_CVPR_2021_paper.pdf)  
*Cheng Wen, Baosheng Yu, Dacheng Tao*  
**[`CVPR 2021`] (`Sydney`)**  



They propose a dual-generators framework 







[DFR: Differentiable Function Rendering for Learning 3D Generation from Images](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14082)  
*Yunjie Wu, Zhengxing Sun*  
**[`Computer Graphics Forum 2020`] (`Nanjing University`)**

![image-20211009153751542](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20211009153759.png)

point-picking strategy which formulates each ray’s state by only picked points instead of all

3D generation network with only 2D images



Because of the implicit characteristics, it is hard to design an encoder for it. DeepSDF propose an auto-decoder model to replace the traditional auto-encoder and omit the encoder part.



employ an encoder-decoder network

The encoder is a resnet-18, which takes the RGB image as input.

The decoder is a conditional network to represent the 3D shape.



没有引NERF，在NERF之前或者同期



在训练好他的 DFR 模型后，他说一个应用是训练出一个不需要3D训练数据的3D GAN（但这他的DFR模型就是可以提取3D信息的，用来训练当然是可以的）。所以他只是当作一个应用，而不是他的论文核心。



判别器判断的是真假 silhouette，生成器的输出是 implicit function，然后通过DFR 渲染出一张某个角度的轮廓去比较。细节之处采用的是WGAN-GP，判别器就是简单的DCGAN。





---





Occupancy networks: Learning 3d reconstruction in function space

Learning implicit fields for generative shape modeling

**use encoder for voxel or point cloud to build up the auto-encoder**.





[Synthesizing 3D Shapes from Silhouette Image Collections using Multi-projection Generative Adversarial Networks](https://arxiv.org/pdf/1906.03841.pdf)  
*Xiao Li, Yue Dong, Pieter Peers, Xin Tong*  
**[`CVPR 2019`] (`USTC, Microsoft`)**

multiple discriminator



They have silhouette images of the class of objects that follow a distribution $\mathbf{Y}$

where $\mathbf{Y} = \mathbf{P} (\mathbf{X}, \mathbf{\Phi})$ and $\mathbf{\Phi}$ are the latent parameters of the projection (e.g., the intrinsic camera parameters)

Given a 3D voxel shape and viewpoint, we first compute a ray intersection probability for each voxel using ray-casting. Next, the silhouette is computed as the expected value of the intersection probability along the z axis.

---



