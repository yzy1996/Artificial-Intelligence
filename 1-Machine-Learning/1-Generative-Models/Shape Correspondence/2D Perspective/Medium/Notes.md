# <p align=center>`Mediator | Midium Template`</p>

> Due to the lack of Github support for LaTeX math formulas, it is recommended that you can download it and view it locally with your own Markdown editor (like Typora, VSCode).







[Canonical Surface Mapping via Geometric Cycle Consistency](https://arxiv.org/pdf/1907.10043.pdf)  
**[`ICCV 2019`] (`CMU, Facebook`)**  
*Nilesh Kulkarni, Abhinav Gupta, Shubham Tulsiani*

<details><summary>Click to expand</summary>

<div align=center><img width="700" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210926111519.png"/></div>

> **Summary**

They explore the task of Canonical Surface Mapping (CSM) which means ''**given an image, learn to map pixels on the object to their corresponding locations on an abstract 3D model of the category**''. This brings possibility of inferring dense correspondence across images of a category.

They combine **pixel-to-3D** and **3D-to-pixel** to form a **cycle** (pixels -> 3D -> pixels) and then use a geometric cycle consistency loss to achieve unsupervised training. 

> **Details**

<div align=center><img width="400" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210926161819.png"/></div>

一个问题是，上图感觉是能和做到和背景解耦的，它是如何实现的呢：需要foreground mask labels的监督信号。

里面的多张图是同一物体吗，（是不同物体的数据集还是同一物体不同姿态的数据集）：不是同一物体，用的数据集是CUB-200-2011 PASCAL3D+



> **Shortcoming**

- need foreground mask labels
- 



</p></details>

---

[Articulation-aware Canonical Surface Mapping](https://arxiv.org/pdf/2004.00614.pdf)  
**[`CVPR 2020`] (`UM, CMU, Facebook`)**  
*Nilesh Kulkarni, Abhinav Gupta, David F. Fouhey, Shubham Tulsiani*

继续前作，因为前面是没有3D模板的角度信息，在这一篇里作者既解决了CSM问题，同时推断出了关节和角度；也可以说之前的工作是关注刚体，现在是非刚体了

<div align=center><img width="400" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210926174050.png"/></div>



---

[Semantic Correspondence via 2D-3D-2D Cycle](https://arxiv.org/pdf/2004.09061.pdf)  
**[`Arxiv 2020`] (`SJTU`)**  
*Yang You, Chengkun Li, Yujing Lou, Zhoujun Cheng, Lizhuang Ma, Cewu Lu, Weiming Wang*

<details><summary>Click to expand</summary>


<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210705222652.png" alt="image-20210705222643577" style="zoom: 33%;" />

> Summary



</p></details>

---

[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://arxiv.org/pdf/1807.03146.pdf)  
**[`NeurIPS 2018`] (`Google`)**  
*Supasorn Suwajanakorn, Noah Snavely, Jonathan Tompson, Mohammad Norouzi*

<details><summary>Click to expand</summary>



<div align=center>
	<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210508160114.png" width="800" />
</div>



> **Summary**

**dubbed-"KeypointNet"**

learn category- specific 3D keypoints by solving an auxiliary task of rigid registration between multiple renders of the same shape and by considering the category instances to be pre-aligned.

from an end-to-end geometric reasoning framework, jointly optimize the keypoints.

also show these 3D keypoints can infer their depths without access to object geometry.

using aligned 3D and multiple 2D images with known pose.

</p></details>





