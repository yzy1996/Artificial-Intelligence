>*reference* :
>
>[Geometric Programming for Communication Systems](http://www.princeton.edu/~chiangm/gp.pdf)
>
>[A tutorial on geometric programming](https://web.stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf)



# Problem 1

Turn this nonconvex problem into a Generalized GP:
$$
\begin{aligned}
\min \ \ \ \ & \max \left\{\left(x_{1}+x_{2}^{-1}\right)^{0.5}, x_{1} x_{3}\right\}+\left(x_{2}+x_{3}^{-2.9}\right)^{1.5} \\
s.t. \ \ \ \ 
& \frac{\left(x_{2} x_{3}+x_{2} / x_{1}\right)^{\pi}}{x_{1} x_{2}-\max \left\{x_{1}^{2} x_{3}^{3}, x_{1}+x_{3}\right\}} \leq 10 \\
\text{variables} \ \ \ \ &\quad x_{1}, x_{2}, x_{3}
\end{aligned}
$$

**!!! important !!! The denominator should be greater than 0**

Generalized GP:
$$
\begin{aligned}
\min \ \ \ \ &t_1+t_2 \\
s.t. \ \ \ \ 
& (x_1 + x_2^{-1})^{0.5} \leq t_1 \\
& x_1x_3 \leq t_1 \\
& (x_2+x_3^{-2.9})^{1.5} \leq t_2 \\
& (x_2x_3 + x_2x_1^{-1})^{\pi} \leq t_3 \\
& x_1^2x_3^3 \leq t_4 \\ 
& x_1 + x_3 \leq t_4 \\
& t_3 + 10t_4 \leq 10 x_1x_2 
\end{aligned}
$$

Standard GP:
$$
\begin{aligned}
\min \ \ \ \ &t_1+t_2 \\
s.t. \ \ \ \ 
& x_1t_1^{-2} + x_2^{-1}t_1^{-2} \leq 1 \\
& x_1x_3t_1^{-1} \leq 1 \\
& x_2t_2^{-2/3}+x_3^{-2.9}t_2^{-2/3} \leq 1 \\
& x_2x_3t_3^{-1/\pi} + x_2x_1^{-1}t_3^{-1/\pi} \leq 1 \\
& x_1^2x_3^3t_4^{-1} \leq 1 \\ 
& x_1t_4^{-1} + x_3t_4^{-1} \leq 1 \\
& 0.1t_3x_1^{-1}x_2^{-1} + t_4x_1^{-1}x_2^{-1} \leq 1
\end{aligned}
$$

```python
# primal problem 
import numpy as np
import cvxpy as cp

x1 = cp.Variable(pos=True)
x2 = cp.Variable(pos=True)
x3 = cp.Variable(pos=True)

objective_fn = cp.maximum(cp.sqrt(x1 + x2 ** (-1)), x1 * x3) + (x2 + x3 ** (-2.9)) ** (1.5)
constraints = [(x2 * x3 + x2 / x1) ** (np.pi) + 10 * cp.maximum(x1 ** 2 * x3 ** 3, x1 + x3) <= 10 * x1 * x2]
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
problem.solve(gp=True)
print("Optimal value: ", problem.value)
print("x1: ", x1.value)
print("x2: ", x2.value)
print("x3: ", x3.value)
```

```python
# Standard GP problem 
import numpy as np
import cvxpy as cp

x1 = cp.Variable(pos=True)
x2 = cp.Variable(pos=True)
x3 = cp.Variable(pos=True)
t1 = cp.Variable(pos=True)
t2 = cp.Variable(pos=True)
t3 = cp.Variable(pos=True)
t4 = cp.Variable(pos=True)


obj = cp.Minimize(t1 + t2)
constraints = [
    x1 + x2**(-1) <= t1**2,
    x1 * x3 <= t1,
    x2 + x3**(-2.9) <= t2**(2/3),
    x2 * x3 + x2 * x1**(-1) <= t3**(1/np.pi),
    x1**2 * x3**3 <= t4,
    x1 + x3 <= t4,
    t3 + 10 * t4 <= 10 * x1 * x2
]
problem = cp.Problem(obj, constraints)
problem.solve(gp=True)
print("Optimal value: ", problem.value)
print("x1: ", x1.value)
print("x2: ", x2.value)
print("x3: ", x3.value)
```



# Problem 2

[the 26th Vojtěch Jarník international mathematics competition question](https://vjimc.osu.cz/storage/uploads/j26solutions2.pdf)

Let 𝑎,𝑏,𝑐 be positive real numbers with $a+b+c =1$. Prove that
$$
\left(\frac{1}{a}+\frac{1}{b c}\right)\left(\frac{1}{b}+\frac{1}{a c}\right)\left(\frac{1}{c}+\frac{1}{a b}\right) \geq 1728
$$

turn into a Standard GP:
$$
\begin{aligned}
\min \ \ \ \ & t_1 t_2 t_3 \\
s.t. \ \ \ \ 
& \frac{1}{a t_1}+\frac{1}{b c t_1} \leq 1 \\
& \frac{1}{b t_2}+\frac{1}{a c t_2} \leq 1 \\
& \frac{1}{c t_3}+\frac{1}{a b t_3} \leq 1 \\
& a + b + c <= 1 \\
\end{aligned}
$$
**Think originally**

- using basic inequality separately

  - first try 

  $$
  \frac{1}{a}+\frac{1}{b c} \geq \frac{2}{\sqrt{abc}}
  $$

  Equality holds if $a = bc$, likewise there will be $b = ac$  ,  $c = ab$, they cannot be satisfied at the same time(because need $a = b = c = 1$ which contradict $a+b+c=1$).

  - second try

  $$
  \frac{1}{a}+\frac{1}{b c} \geq \frac{1}{a}+\frac{1}{b}+\frac{1}{c} \geq \frac{9}{a + b + c} = 9
  $$
  The condition for equal sign is $b = c$, likewise the final condition for equal sign is $a = b = c = 1/3$ , satisfied but not 1728, it’s relaxed. 

  **So we can get the conclusion that it is unprovable to use basic inequalities for each of the three terms**.

- using expand and monotonic as a whole

  $$
  \begin{aligned}
  \left(\frac{1}{a}+\frac{1}{b c}\right)\left(\frac{1}{b}+\frac{1}{a c}\right)\left(\frac{1}{c}+\frac{1}{a b}\right)
  & = \left(\frac{1}{ab}+\frac{1}{b^2c}+\frac{1}{a^2c}+\frac{1}{abc^2}\right)\left(\frac{1}{c}+\frac{1}{ab}\right) \\
  & = \frac{1}{abc}+\frac{1}{b^2c^2}+\frac{1}{a^2c^2}+\frac{1}{abc^3}+\frac{1}{a^2b^2}+\frac{1}{acb^3}+\frac{1}{bca^3}+\frac{1}{a^2b^2c^2} \\
  & \geq \frac{1}{abc}+\frac{3}{\sqrt[3]{(a b c)^4}}+\frac{3}{\sqrt[3]{(a b c)^5}}+\frac{1}{(abc)^2}
  \end{aligned}
  $$
  and according to 
  $$
  \sqrt[3]{abc} \leq \frac{1}{a+b+c} = \frac{1}{3} \Rightarrow \frac{1}{abc} \geq 27
  $$
  so,
  $$
  \frac{1}{abc}+\frac{3}{\sqrt[3]{(a b c)^4}}+\frac{3}{\sqrt[3]{(a b c)^5}}+\frac{1}{(abc)^2} \geq 27 + 243 + 729 +729 = 1728
  $$

```python
import numpy as np
import cvxpy as cp

a = cp.Variable(pos=True)
b = cp.Variable(pos=True)
c = cp.Variable(pos=True)
t1 = cp.Variable(pos=True)
t2 = cp.Variable(pos=True)
t3 = cp.Variable(pos=True)

obj = cp.Minimize(t1 * t2 * t3)
constraints = [
    a**(-1) * t1**(-1) + b**(-1) * c**(-1) * t1**(-1) <= 1,
    b**(-1) * t2**(-1) + a**(-1) * c**(-1) * t2**(-1) <= 1,
    c**(-1) * t3**(-1) + a**(-1) * b**(-1) * t3**(-1) <= 1,
    a + b + c <= 1]
problem = cp.Problem(obj, constraints)
problem.solve(gp=True)
print("Optimal value: ", problem.value)
print("a: ", a.value)
print("b: ", b.value)
print("c: ", c.value)
```



# Problem 3

[1995 IMO Problems](https://artofproblemsolving.com/wiki/index.php/1995_IMO_Problems/Problem_2)

Let 𝑎,𝑏,𝑐 be positive real numbers with $𝑎𝑏𝑐=1$. Prove that

$$
\frac{1}{a^{3}(b+c)} + \frac{1}{b^{3}(c+a)} + \frac{1}{c^{3}(a+b)} \geq \frac{3}{2}
$$

We can write:

$$
\begin{aligned}
\min \ \ \ \ & a^{-3} t_{1}^{-1}+b^{-3} t_{2}^{-1}+c^{-3} t_{3}^{-1} \\
s.t. \ \ \ \ 
& b+c \geq t_{1} \\
& a+c \geq t_{2} \\
& a+b \geq t_{3} \\
& a + b + c \leq 1 \\
\end{aligned}
$$

We can not turn it into GP. Only if we make it more relaxed, it can be turned into GP. The key point is to transform $b+c \geq t_1$ to $max\{b, c\} \geq \frac{t_1}{2}$, hence standard GP is:

$$
\begin{aligned}
\min \ \ \ \ & a^{-3} t_{1}^{-1}+b^{-3} t_{2}^{-1}+c^{-3} t_{3}^{-1} \\
s.t. \ \ \ \ 
& t_{1}/2b \leq 1 \\
& t_{1}/2c \leq 1 \\
& t_{1}/2a \leq 1 \\
& t_{1}/2c \leq 1 \\
& t_{1}/2a \leq 1 \\
& t_{1}/2b \leq 1 \\
& a + b + c \leq 1 \\
\end{aligned}
$$

```python
import numpy as np
import cvxpy as cp

a = cp.Variable(pos=True)
b = cp.Variable(pos=True)
c = cp.Variable(pos=True)
t1 = cp.Variable(pos=True)
t2 = cp.Variable(pos=True)
t3 = cp.Variable(pos=True)

obj = cp.Minimize(1 / (a**3 * t1) + 1 / (b**3 * t2) + 1 / (c**3 * t3))
constraints = [
    t1 / (2 * b)  <= 1,
    t1 / (2 * c)  <= 1,
    t2 / (2 * a)  <= 1,
    t2 / (2 * c)  <= 1,
    t3 / (2 * a)  <= 1,
    t4 / (2 * b)  <= 1,
    a * b * c <= 1]
problem = cp.Problem(obj, constraints)
problem.solve(gp=True)
print("Optimal value: ", problem.value)
print("a: ", a.value)
print("b: ", b.value)
print("c: ", c.value)
```



