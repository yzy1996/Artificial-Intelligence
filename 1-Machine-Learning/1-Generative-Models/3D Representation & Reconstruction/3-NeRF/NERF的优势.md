可以保证 view-condidtency，

因为体密度只和坐标x有关





存在的问题呢？



**NeRF overfits to training views**

mimicking the image-formation process at observed poses -- from dietnerf



the optimal scene representation is underdetermined



Degenerate solutions





关于nerf的描述

NeRF[28] proposed optimizing scene radiance fields, representedusing global MLPs, from RGB images to achieve photorealisticnovel view synthesis. However, NeRF cannot handlelarge-scale scenes well due to its limited MLP networkcapacity and impractical slow per-scene optimization.