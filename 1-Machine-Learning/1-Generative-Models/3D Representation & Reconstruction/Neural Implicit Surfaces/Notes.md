# <p align=center>`Neural Implicit Surfaces`</p>



[NeuS](#NeuS)

[VolSDF](#VolSDF)

---

<span id="NeuS"></span>
[NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/pdf/2106.10689.pdf)  
**[`Arxiv 2021`]  (`HKU, MIT`)**   
*Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, Wenping Wang*

<details><summary>Click to expand</summary>

<div align=center><img width="700" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210916164746.png"/></div>

> **Summary**

A method to reconstruct surface with high fidelity from 2D images. NeuS uses the SDF for surface representation and uses a novel volume rendering scheme to learn. **Only focus on solid objects.**

> **Details**

Solve the problems:

- simply applying a standard volume rendering method to the density associated with SDF would lead to discernible bias. :arrow_right: propose a novel volume rendering scheme to ensure unbiased surface reconstruction in the first-order approximation of SDF.



NeuS consists of two networks:

- $f: \mathbb{R}^3 \rightarrow \mathbb{R}$ spatial point :arrow_right: its signed distance
- $c: \mathbb{R}^3 \times \mathbb{S}^2 \rightarrow \mathbb{R}^3$ spatial point and view direction :arrow_right: color

$$
\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x}) = 0 \}
$$

logistic density distribution $\phi_{s}(x)=s e^{-s x} /\left(1+e^{-s x}\right)^{2}$, which is the derivative of the Sigmoid function $\Phi_s(x) = (1 + e^{-sx})^{-1}$. Using S-density field $\phi_s(f(x))$ to render.

</details>

---

<span id="VolSDF"></span>
[Volume Rendering of Neural Implicit Surfaces](https://arxiv.org/pdf/2106.12052.pdf)  
**[`Arxiv 2021`] (`Weizmann Institute of Science, Facebook`)**  
*Lior Yariv, Jiatao Gu, Yoni Kasten, Yaron Lipman*

<details><summary>Click to expand</summary>

<div align=center><img width="700" src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210916202925.png"/></div>

> **Summary**

11

> **Details**

11

</details>

---

