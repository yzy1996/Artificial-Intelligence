# <p align=center>`Single View Reconstruction`</p>





the type of reconstruction includes 





## Introduction





## Literature

[Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf)  
**[`CVPR 2020`]  (`Oxford`)**  
*Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi*










- [SMR](#smr)

### SMR

Self-Supervised 3D Mesh Reconstruction from Single Images  
**[`CVPR 2021`]**	**(`CUHK`)**  
*Tao Hu, Liwei Wang, Xiaogang Xu, Shu Liu, Jiaya Jia*  
<details><summary>Click to expand</summary><p>
3D attribute $A=[C, L, S, T]$, 3D object $O(S, T)$, where $C$ is Camera, $L$ is Light, $S$ is Shape, $T$ is Texture.

2D image $I$, its silhouette $M$

input $X=[I, M]$, 

their relations are:
$$
\text{rendering: } X = R(A)\\
\text{encoding: } A = E_\theta(X)
$$

<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210410172628.png" alt="image-20210410172613853" style="zoom:50%;" />



</p></details>

---







