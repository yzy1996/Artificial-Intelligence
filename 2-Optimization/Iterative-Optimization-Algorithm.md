# Iterative optimization algorithm



include **point-based** and **population-based**

they have the same generic update formula:
$$
x^{t+1}=x^{t}+\delta x^{t},
$$
where $x^t$ and $\delta x^t$ are the **sample** (or a single sample called “particle" in swarm algorithms) and the **update** (a.k.a. step vector) at iteration $t$, respectively. The update is often a function $g(\cdot)$ of historic sample values, objective values, and gradients,
$$
\delta \boldsymbol{x}^{t}=g\left(\left\{\boldsymbol{x}^{\tau}, f\left(\boldsymbol{x}^{\tau}\right), \nabla f\left(\boldsymbol{x}^{\tau}\right)\right\}_{\tau=1}^{t}\right)
$$




## point-based 



**gradient descent**
$$
\delta \boldsymbol{x}^{t} = -\alpha \nabla f\left(\boldsymbol{x}^{t}\right)
$$








## population based 

