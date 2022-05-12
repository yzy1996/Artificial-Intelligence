# Shape Encode

We hope to enable users to perform user-controlled shape deformation in the scene.

our control should be operated on the whole 3D shape, thus we can synthesizes the novel view images of the edited scene without re-training the network.



support user to edit the attributes of 3D shape, and resulting photo realistic rendering from novel views.



What we can control:

- color editing 
- object translation and rotation
- shape deformation 







核心是解耦出shape and appearance，用不同的 latent code来表示，这样就可以做到根据code生成。

Category-Level NeRF



里面也包含了 conditional nerf

- [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2007.02442.pdf)  
  *Katja Schwarz, Yiyi Liao, Michael Niemeyer, Andreas Geiger*  
  **[`NeurIPS 2020`] (`MPI`)** [[Code](https://github.com/autonomousvision/graf)]  

- [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/pdf/2012.00926.pdf)  
  *Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, Gordon Wetzstein*  
  **[`CVPR 2021`] (`Stanford`)**  

- [pixelNeRF: Neural Radiance Fields from One or Few Images](https://arxiv.org/pdf/2012.02190.pdf)  
  *Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa*  
  **[`CVPR 2021`] (`UCB`)**  

- [GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering](https://arxiv.org/pdf/2010.04595.pdf)  
  *Alex Trevithick, Bo Yang*  
  **[`ICCV 2021`] (`Williams, Oxford, PolyU`)**  

- [CodeNeRF: Disentangled Neural Radiance Fields for Object Categories](https://arxiv.org/pdf/2109.01750.pdf)  
  *Wonbong Jang, Lourdes Agapito*  
  **[`ICCV 2021`] (`UCL`)** 

- [Editing Conditional Radiance Fields](https://arxiv.org/pdf/2105.06466.pdf)  
  *Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard Zhang, Jun-Yan Zhu, Bryan Russell*  
  **[`ICCV 2021`] (`MIT, Adobe`)**

- [NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination](https://arxiv.org/pdf/2106.01970.pdf)  
  Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul Debevec, William T. Freeman, Jonathan T. Barron  
  **[`Arxiv 2021`] (`MIT`)**
  
- [NeRF-Editing: Geometry Editing of Neural Radiance Fields](https://arxiv.org/abs/2205.04978)  
  *Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, Lin Gao*  
  **[`CVPR 2022`] (`CAS`)**
  

