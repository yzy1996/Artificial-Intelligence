# <p align=center>`Neural Implicit Surfaces`</p>

A collection of resources on Neural Implicit Surfaces.

## Introduction

This line of research focuses on representing the scene’s geometry implicitly using a neural network, making the surface rendering process differentiable. 



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



### Category

<span id="VolSDF"></span>
[Volume Rendering of Neural Implicit Surfaces](https://arxiv.org/pdf/2106.12052.pdf)  
**[`Arxiv 2021`] (`Weizmann Institute of Science, Facebook`)**  
*Lior Yariv, Jiatao Gu, Yoni Kasten, Yaron Lipman*



DVR

Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision



IDR

Multiview neural surface reconstruction by disentangling geometry and appearance



[NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/pdf/2106.10689.pdf)  
**[`Arxiv 2021`]  (`HKU, MIT`)**   
*Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, Wenping Wang*





## Main Research Group

<h5 align="center"><i>If warmup is the answer, what is the question?</i></h5>



