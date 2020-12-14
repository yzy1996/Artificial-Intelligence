> Problem

Modeling a real scene from captured images and reproducing its appearance under novel conditions.

recover scene geometry and reflectance

surface reconstruction

> Related technology

3D reconstruction & inverse rendering

> new break

combines neural scene representations with classical ray marching - a volume rendering approach that is naturally differentiable



- implicit model
- explicit model





# Literature Contents

[Neural Reflectance Fields](#Neural Reflectance Fields)

[Analytic Meshing](#Analytic Meshing)



## Neural Reflectance Fields

[Neural Reflectance Fields for Appearance Acquisition]()

**`[]`**	**`(UC San Diego)`**	**`[SAI BI, ZEXIANG XU]`**	**([:memo:]())**	**[[:octocat:](https://github.com/NVlabs/stylegan)]**

<details><summary>Click to expand</summary><p>


**Field**

view synthesis works



**Noun explanation**

reflectance: 

light transmittance:



**Related work**

> 1. Neural scene representations
>
>    method including: volumes & point clouds & implicit functions & **neural reflectance field**
>
>    ray marching
>
> 2. Geometry and reflectance capture
>
>    Classically, modeling and rerendering a real scene requires full reconstruction of its geometry and reflectance. From captured images, scene geometry is usually reconstructed by structure-from-motion and multi-view stereo.
>
>    Now a practical device - modern cellphone that has a camera and a built-in flash light – and capture flash images to acquire spatially varying **BRDFs**. Such a device only acquires reflectance samples under collocated light and view.
>
> 3. Relighting and view synthesis
>
>    

**Aims**

> We aim to model geometry and appearance of complex real scenes from multi-view unstructured flash images

Neural Reflectance Fields are a continuous function neural representation that **implicitly models both scene geometry and reflectance**.

represent by a deep multi-layer perceptron (MLP)


</p></details>

---



https://whimsical.com/view-synthesis-2zfiLatcYU3nsNmAb2y74f





## Analytic Meshing

[Analytic Marching: An Analytic Meshing Solution from Deep Implicit Surface Networks](https://arxiv.org/abs/2002.06597)

**`[PMLR 2020]`**	**`(SUST)`**	**`[Jiabao Lei, Kui Jia]`**	**([:memo:]())**	**[[:octocat:](https://github.com/NVlabs/stylegan)]**

<details><summary>Click to expand</summary><p>


**Summary**

> This paper studies the problem of **surface reconstruction** through learning **surface mesh** via **implicit functions**, where implicit functions are implemented as multi-layer perceptrons (**MLPs**) with rectified linear units (**ReLU**).



**Related work**

- [ ] marching cubes (standard algorithm)
- [ ] a ReLU based MLP partitions its input space into a number of linear regions
- [ ] polygonal mesh 
- [ ] signed distance function or SDF



`isosurface extraction`: implicit representation -> explicit surface mesh



</p></details>

---