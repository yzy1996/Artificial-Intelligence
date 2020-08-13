1. compute the gradient descent direction:
   $$
   p_k = (-x_{1(k)}, -\gamma x_{2(k)})^T
   $$

2. compute the stepsize:
   $$
   t_k  = \underset{\alpha \geq 0}{\arg\min} \frac{1}{2}\left[(1-\alpha)^{2} x_{1(k)}^{2}+\gamma(1-\alpha \gamma)^{2} x_{2(k)}^{2}\right]
   $$
   Using the collocation method, we get:
   $$
   t_k = \frac{x_{1(k)}^2 + \gamma^2 x_{2(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2}
   $$

3. apply descent method:
   $$
   x_{k+1} = x_k + \frac{x_{1(k)}^2 + \gamma^2 x_{2(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2} (-x_{1(k)}, -\gamma x_{2(k)})^T
   $$
   divide $x_1, x_2$ into two parts:
   $$
   x_{1(k+1)} = x_{1(k)} - \frac{x_{1(k)}^2 + \gamma^2 x_{2(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2} x_{1(k)} = \frac{(\gamma - 1)\gamma^2 x_{2(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2} x_{1(k)}
   $$

   $$
   x_{2(k+1)} = x_{2(k)} - \frac{x_{1(k)}^2 + \gamma^2 x_{2(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2} \gamma x_{2(k)} = \frac{(1 - \gamma) x_{1(k)}^2}{x_{1(k)}^2 + \gamma^3 x_{2(k)}^2} x_{2(k)}
   $$

4. select an initial point and start the iteration, here select a special point  $x_{1(0)} = \gamma$，$x_{2(0)} = 1$ 

   when $k = 0$, we have:
   $$
   x_{1(1)} = \gamma\left(\frac{\gamma-1}{\gamma+1}\right)
   $$

   $$
   x_{2(1)} = -\left(\frac{\gamma-1}{\gamma+1}\right)
   $$

   when $k =1$, we have：
   $$
   x_{1(2)} = \gamma\left(\frac{\gamma-1}{\gamma+1}\right)^2
   $$

   $$
   x_{2(2)} = -\left(\frac{\gamma-1}{\gamma+1}\right)^2
   $$

   by mathematical induction, we can get a general formula:

   $$
   x_{1(k)} = \gamma\left(\frac{\gamma-1}{\gamma+1}\right)^k
   $$

   $$
   x_{2(k)} = -\left(\frac{\gamma-1}{\gamma+1}\right)^k
   $$


- Because $\nabla f(x) = Qx$ , $\nabla^2 f(x) = Q$ and $Q$ is positive definite; so the function $f(x)$ is convex according to the definition.

- Firstly, we prove $f(x) = max\{x_i\}$ is convex:

    $$
    \begin{aligned}
    f(\theta x + (1-\theta) y) 
    &= max\{\theta x + (1-\theta) y\}  \\
    &\leq max\{\theta x\} + max\{(1-\theta)y\} \\
    &= \theta f(x) + (1-\theta) f(y)
    \end{aligned}
    $$
    so, we have proved this $f(x)$ is convex
    
    Secondly, we know that the sum of convex functions is still convex, so the original function can be seen as the sum of several max functions. 
    
    Finally, we can say  $f(x)=\sum_{i=1}^{k} x_{[i]}$ is convex



1. original function is:
   $$
   \begin{aligned}
   \min \frac{1}{2} &x^TQx \\
   \text {s.t. } Ax &= b \\
   Cx &\leq d
   \end{aligned}
   $$
   Lagrange function is:
   $$
   L(x,\lambda, v)=\frac{1}{2} x^TQx + \lambda^T (Ax-b) + v^T (Cx-d)
   $$
   because:
   $$
   \nabla_x L(x,\lambda,v) = Qx+A^T\lambda + C^T v
   $$
   when $x =  -Q^{-1}(A^T\lambda+C^Tv)$ Lagrange function reach minimum. Therefore the dual function is:
   $$
   \begin{aligned}
   g(\lambda,v)
   &=\frac{1}{2}(A^T\lambda+C^Tv)^TQ^{-1}(A^T\lambda+C^Tv)-\lambda^T(AQ^{-1}(A^T\lambda+C^Tv)+b)-v^T(CQ^{-1}(A^T\lambda+C^Tv)+d)\\
   &=\frac{1}{2}(A^T\lambda+C^Tv)^TQ^{-1}(A^T\lambda+C^Tv)-(\lambda^TA+v^TC)Q^{-1}(A^T\lambda+C^Tv)-\lambda^Tb-v^Td
   \end{aligned}
   $$
   so the Lagrange dual problem is:
   $$
   \begin{aligned}
   &\max g(\lambda, v) \\
   &\text {s.t. } \lambda \geq 0
   \end{aligned}
   $$

2. original function is:
   $$
   \begin{aligned}
   \min \sum_{i=1}^k & x_{[i]}\\
   \text {s.t. } Ax &= b \\
   Cx &\leq d
   \end{aligned}
   $$
   Lagrange function is:
   $$
   L(x,\lambda, v)=\sum_{i=1}^k x_{[i]} + \lambda^T (Ax-b) + v^T (Cx-d)
   $$
   the dual function is:
   $$
   g(\lambda,v) = \inf _{x} L(x,\lambda,v) = -\lambda^T b - v^T d +\inf _{x}\left(\sum_{i=1}^k x_{[i]} + (\lambda^T A+v^T C)x\right)
   $$
   since a linear function is bounded below only when it is identically zero, so:
   $$
   g(\lambda, \nu)=
   \left\{\begin{array}{ll}
   -\lambda^T b - v^T d + kx_{[k]} & \lambda^T A+v^T C=0 \\
   -\infty & \text { otherwise }
   \end{array}\right.
   $$
   so the Lagrange dual problem is:
   $$
   \begin{aligned}
   &\max g(\lambda, v) \\
   &\text {s.t. } \lambda \geq 0
   \end{aligned}
   $$



(a) the rank problem is:
$$
\begin{aligned}
&\text { minimize } \quad\|\mathbf{B r}-\mathbf{v}\|_{2}\\
&\text { subject to } \quad \mathbf{r}^{T} \mathbf{1}=0
\end{aligned}
$$
$$
\begin{aligned}
&\text { maximize } \quad\|\mathbf{B r}-\mathbf{v}\|_{2}\\
&\text { subject to } \quad \mathbf{r}^{T} \mathbf{1}=0
\end{aligned}
$$

\frac{\Gamma(2 x)}{x(\Gamma(x))^{2}}





$\Gamma(\theta x_1+(1-\theta) x_2) \leq \Gamma(x_1)^\theta  \Gamma(x_2)^{(1-\theta)}$


$$
\int_{0}^{+\infty} u^{\left(x_{1}-1\right) \theta} e^{-\theta u} \cdot t^{\left(x_{2}-1\right) (1-\theta))} e^{-(1-\theta) u} d u
$$

$$
\begin{aligned}
\left(\int_{0}^{+\infty} u^{x_{1}-1} e^{-u} d u\right)^{\theta}\left(\int_{0}^{+\infty} u^{x_{2}-1} e^{-u} d u\right)^{1-\theta} 
&\geq \int_{0}^{+\infty} u^{\left(x_{1}-1\right) \theta} e^{-\theta u} \cdot u^{\left(x_{2}-1\right) (1-\theta))} e^{-(1-\theta) u} d u \\
&= \int_{0}^{+\infty} u^{\theta x_1 + (1-\theta)x_2 -1} e^{-u} d u 
\end{aligned}
$$

## (a)

$$
\begin{aligned}
x^{(k+1)} &= x^{(k)} - \mu A^T(Ax^{(k)}-b)\\
x^{(k)}-x^{(k+1)} &= \mu A^T(Ax^{(k)}-b)
\end{aligned}
$$

now if:
$$
x^{(k+1)} = x^{(k)}
$$
so we can get:
$$
\begin{aligned}
\mu A^T(Ax^{(k)}-b) &= 0 \\
A^TAx^{(k)}-A^Tb &= 0 \\
x^{(k)} &= (A^TA)^{-1}A^Tb
\end{aligned}
$$
and because 
$$
\widehat{x} = (A^TA)^{-1}A^Tb
$$
Hence 
$$
x^{(k)} = \widehat{x}
$$
   and:
$$
f^*=
   \left\{\begin{array}{ll}
   0 & \lambda^T A+v^T C=0 \\
   -\infty & \text { otherwise }
   \end{array}\right.
$$
so:
$$
g(\lambda, v)=
   \left\{\begin{array}{ll}
   -u & \lambda^T A+v^T C=0 \\
   -\infty & \text { otherwise }
   \end{array}\right.
$$
so the Lagrange dual problem is:
$$
\begin{aligned}
   &\max g(\lambda, v) \\
   &\text {s.t. } \lambda \geq 0
   \end{aligned}
$$



$$
\Gamma(2 x)=(2 \pi)^{-1 / 2} 2^{2 x-1 / 2} \Gamma(x) \Gamma(x+1 / 2)
$$

$$
F(x)=(2 \pi)^{-1 / 2} 2^{2 x-1 / 2} \frac{\Gamma(x+1 / 2)}{\Gamma(x+1)}
$$

$$
(\log F(x))^{\prime \prime}=\sum_{k=0}^{+\infty}\left(\frac{1}{(x+1 / 2)^{2}}-\frac{1}{(x+1)^{2}}\right)>0 \quad \text { for } x>0
$$

$$
(\log G(x))^{\prime \prime}=4 \Psi^{\prime}(2 x)-2 \Psi^{\prime}(x)
$$

$$
\Psi^{\prime}(x)=\sum_{k=0}^{+\infty} \frac{1}{(x+k)^{2}}
$$

$$
4 \Psi^{\prime}(2 x)=\Psi^{\prime}(x)+\Psi^{\prime}(x+1 / 2)
$$

$$
(\log G(x))^{\prime \prime}=\Psi^{\prime}(x+1 / 2)-\Psi^{\prime}(x)<0
$$