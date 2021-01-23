# Neural Rendering Notes

Related repositories:

- [collection of neural rendering papers](https://github.com/weihaox/awesome-neural-rendering) by 

- [collection of neural radiance fields papers](https://github.com/yenchenlin/awesome-NeRF) by 



## Introduction

Computer graphics + generative model two areas come together -> neural rendering 



The first publication that used the term "neural rendering" is *Generative Query Network* (GQN). A newest state-of-the-art report formally defines *Neural Rendering* as:

> Deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.



You need to know first

- [ ] Physical Image Formation

  light source, scene geometry, material properties, light transport, optics, sensor behavior

- [ ] 



2D-to-3D representation



3D-to-3D image rendering

 

3D-aware image synthesis

voxel-based representation



A coordinate-based MLP model. 

<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210123193611.png" alt="image-20210123193611687" style="zoom:50%;" />

the issue is: computing the network weights $\theta$ that 





## Research Teams

[Berkeley Artificial Intelligence Research Lab (BAIR)](https://bair.berkeley.edu/)

- AP. [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)
- AP. [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)
- PhD. [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)
- PhD. [Matthew Tancik](https://www.matthewtancik.com/)

Selected works: `NeRF` | `pixelNeRF`



[Max Planck Institute Graphics Vision & Video Group](http://gvv.mpi-inf.mpg.de/GVV_Team.html) 

- Prof. [Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/)
- Prof. [Andreas Geiger](http://www.cvlibs.net/)
- PhD. [Ayush Tewari](https://people.mpi-inf.mpg.de/~atewari/)
- Ph.D. [Michael Niemeyer](https://m-niemeyer.github.io/)

Selected works: `PatchNets` |



[Stanford Computational Image Lab](http://www.computationalimaging.org/team/)

- AP. [Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)
- Postdoc. [Vincent Sitzmann](https://vsitzmann.github.io/)

Selected works: `SRN` | `MetaSDF`



[University of Washington Graphics and Imaging Laboratory](https://grail.cs.washington.edu/)

- Prof. [Steven Seitz](https://homes.cs.washington.edu/~seitz/)
- PhD. [Keunhong Park](https://keunhong.com/)
- PhD. [Jeong Joon Park](https://jjparkcv.github.io/)

Selected works: `Nerfies` | `DeepSDF`



## Literature

`survey` State of the Art on Neural Rendering



[GQN](#GQN)

---

### GQN

[Neural scene representation and rendering](https://science.sciencemag.org/content/360/6394/1204/tab-pdf)

**[`Science 2018`]**	**(`DeepMind `)**	**[[Code](https://github.com/NVlabs/stylegan)]**

**[`Tero Karras`, `Samuli Laine`, `Timo Aila`]**

<details><summary>Click to expand</summary><p>


> **Summary**

It enables machines to learn to perceive their surroundings based on a representation and generation network. The authors argue that the network has an implicit notion of 3D due to the fact that it could take a varying number of images of the scene as input, and output arbitrary views with correct occlusion.

</p></details>

---