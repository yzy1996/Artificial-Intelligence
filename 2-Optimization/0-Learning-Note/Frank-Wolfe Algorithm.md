# Frank-Wolfe Algorithm

回忆一下为了求解函数最小值
$$
\min_x f(x)
$$
我们最熟悉的就是最速梯度下降法：
$$
x^{(k+1)} = x^{(k)} - \gamma \nabla f(x^{(k)})
$$
上述设定 $x$ 是无约束的，当 $x$ 被约束在一个凸集 $\mathcal{D}$ 上时，就不能再使用上述方法了。Frank和Wolfe提出了一个新的求解方法来求解问题：
$$
\min _{x \in \mathcal{D}} f(x)
$$

---

因为不能取负梯度方向为下降方向，理所当然地我们找到一个解方向 $s$ 与负梯度方向最近（内积最大），于是有：
$$
\max_{s \in \mathcal{D}} -{\nabla f(x^{(k)})}^T s
$$
将负号去掉：
$$
\min_{s \in \mathcal{D}} {\nabla f(x^{(k)})}^T s
$$
于是可以取到的下降方向为： $s - x^{(k)}$ 

---

我们还可以换个思路，[参考了](https://blog.csdn.net/hanlin_tan/article/details/48108301)

带约束的问题非线性问题不好求解，我们就将非线性问题转换为线性问题，将 $f(x)$ 在 $x^{(k)}$ 处进行一阶泰勒展开：
$$
f(x) \approx f(x^{(k)}) + \nabla {f(x^{(k)})}^T (x - x^{(k)})
$$
因此求解的问题（1）就变成了：
$$
\min_{x \in \mathcal{D}} f(x^{(k)}) + \nabla {f(x^{(k)})}^T (x - x^{(k)})
$$
 去掉常数项后：
$$
\min_{x \in \mathcal{D}} \nabla {f(x^{(k)})}^T x
$$

---







A general constrained convex optimization problem:
$$
\min _{\boldsymbol{x} \in \mathcal{D}} f(\boldsymbol{x})
$$
where the objective function $f$ is convex and continuously differentiable. The domain $\mathcal{D}$ is a compact convex subset of any vector space. For such optimization problems, one of the simplest and earliest known iterative optimizers is given by the Frank-Wolfe method (1956),  described in Algorithm 1, also known as the conditional gradient method. [^1]

![image-20200904143302726](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20200904143305.png)

第二项改为：
$$
x^{(k+1)} := x^{(k)} + \gamma (s - x^{(k)})
$$


### Hilbert space 希尔伯特空间

指完备的内积空间，是有限维欧几里得空间的一个推广。

完备指的是：任意一个柯西序列都收敛

![img](https://pic4.zhimg.com/80/v2-820f1a2e7aa093e1a372565747b185ed_720w.jpg?source=1940ef5c)

### Euclidean vector space 欧几里得空间

### 





[^1]: Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization

