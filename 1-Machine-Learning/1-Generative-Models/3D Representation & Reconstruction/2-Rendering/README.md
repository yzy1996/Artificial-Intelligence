# Rendering



## Non-differentiable Rendering

a discrete operation called rasterization





## Differentiable Rendering

a fundamental tool in computer graphics that convert 3D models into 2D images.



neural networks bring "neural rendering" because natural differentiable





---

Neural 3d mesh renderer

CVPR 2018

Differentiable monte carlo ray tracing through edge sampling

SIGGRAPH 2018

Soft rasterizer: A differentiable renderer for image-based 3d reasoning

ICCV 2019

Opendr: An approximate differentiable renderer

ECCV 2014





Neural scene representation and rendering.

Science

Rendernet: A deep convolutional network for differentiable rendering from 3d shapes

NIPS 2018

Face-to-parameter translation for game character auto-creation.

ICCV 2019





Dist: Rendering deep implicit signed distance function with differentiable sphere tracing

iterative ray marching to render a DeepSDF decoder

stratified random sampling 





## Volume Rendering

a set of $\mathcal{S}$ of sampled points per ray

A ray $x$ emanating from a camera position c in direction v 

x = c + tv 

volume rendering is approximating the integrated light radiance along this ray reaching the camera.




$$
\text{transparency: } T(t)=\exp \left(-\int_{0}^{t} \sigma(\boldsymbol{x}(s)) d s\right)
\\
\text{opacity: } O(t)=1-T(t)
\\
\text{PDF: } \tau(t)=\frac{d O}{d t}(t)=\sigma(\boldsymbol{x}(t)) T(t)
$$
the volume rendering is the expected light along the ray
$$
I(\boldsymbol{c}, \boldsymbol{v})=\int_{0}^{\infty} L(\boldsymbol{x}(t), \boldsymbol{n}(t), \boldsymbol{v}) \tau(t) d t
$$






## Surface Rendering

