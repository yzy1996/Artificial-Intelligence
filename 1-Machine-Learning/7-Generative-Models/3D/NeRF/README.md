# NeRF: neural volume rendering



the larger field of *Neural rendering* is defined by the [excellent review paper by Tewari et al.](https://www.neuralrender.com/assets/downloads/TewariFriedThiesSitzmannEtAl_EG2020STAR.pdf) as

> “deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.”



**Neural volume rendering** refers to methods that generate <u>images</u> or <u>video</u> by tracing a ray into the scene and taking an integral of some sort over the length of the ray. Typically a neural network like a multi-layer perceptron (MLP) encodes a function from the 3D coordinates on the ray to quantities like density and color, which are integrated to yield an image.



- 
- 