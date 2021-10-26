# <p align=center>`Neural Implicit Surfaces`</p>

A collection of resources on Neural Implicit Surfaces.

## Introduction

This line of research focuses on representing the scene’s geometry implicitly using a neural network, making the surface rendering process differentiable. 

The goal is to synthesize novel views of a scene from a sparse set of input images.

Previous works of neural volume rendering techniques which is modeled using a generic density function, 



The **geometry** is represented by the object surface.



> Chinese Description: 一个高保真的3D重建，应该保证物体表面是足够真实的，过去基于体密度的方法，是通过找体密度的一个等势面来确定表面的（这种方面会带来噪声，和低保真度）。因此更好的一种方式是





The main drawback of these methods is their requirement of **masks that separate objects from the background.** require foreground mask as supervision

struggle with severe self-occlusion or thin structures

Learning to render surfaces directly tends to grow extraneous parts due to optimization problems.



non-Lambertian surfaces and thin structures



These methods can be divided into two categories:

- ...



- SDF or occupancy

  

### Problem Definition

a set of posed images $\{\mathcal{I}_k\}$ of a 3D object, goal is to reconstruct the surface $\mathcal{S}$ of the object.



## Literature

### Survey



- <span id="VolSDF"></span>
  [Volume Rendering of Neural Implicit Surfaces](https://arxiv.org/pdf/2106.12052.pdf)  
  *Lior Yariv, Jiatao Gu, Yoni Kasten, Yaron Lipman*  
  **[`Arxiv 2021`] (`Weizmann Institute of Science, Facebook`)**  

- <span id="Neural-Splines"></span>
  [Neural Splines: Fitting 3D Surfaces with Infinitely-Wide Neural Networks](https://arxiv.org/pdf/2006.13782.pdf)  
  **[`CVPR 2021`] (`NYU, Amazon`)**  
  *Francis Williams, Matthew Trager, Joan Bruna, Denis Zorin*

- <span id="UNISURF"></span>
  [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction](https://arxiv.org/pdf/2104.10078.pdf)  
  **[`ICCV 2021`] (`MPI`)**  
  *Michael Oechsle, Songyou Peng, Andreas Geiger*

- [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/pdf/2106.10689.pdf)  
  **[`NeurIPS 2021`]  (`HKU, MIT`)**   
  *Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, Wenping Wang*



DVR

Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision



IDR

Multiview neural surface reconstruction by disentangling geometry and appearance









## Main Research Group

<h5 align="center"><i>If warmup is the answer, what is the question?</i></h5>



