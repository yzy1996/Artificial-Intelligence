# numpy使用积累



## array数组

### 一维向量array

```
a = [1, 2, 3]

# 变array一维向量
b = np.array(a)

# 行向量变列向量
c1 = b.reshape(-1, 1)
c2 = np.array([a]).T
```



### narray

```python
# array 和 narray 转换
a = np.array([1, 2, 3])
a.reshape(-1, 1)
>>> array([[1, 2, 3]])

b = np.array([[1, 2, 3]])
b.reshape(-1)
>>> array([1, 2, 3])
```





增加行列

```
import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[0,0,0]])
c = np.r_[a,b]
d = np.c_[a,b.T]
```



往后追加



或者重新赋值

```python
a  = np.zeros((5, 2))
# a  = np.empty((w, h))

a[0:1] = 1
a[1:2] = 2
```



**增加轴**

```python
x = np.array([1, 2])
>>> x.shape = (2,)
y = np.expand_dims(x, axis=0)
>>> y.shape = (1, 2)
```

**交换轴**

```python
x = np.ones((1, 2, 3))
>>> x.shape = (1, 2, 3)
y = np.transpose(x, (1, 0, 2))
>>> y.shape = (2, 1, 3)
```







## random随机数

> Random sampling ([`numpy.random`](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html))

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`rand`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html#numpy.random.rand)(d0, d1, ..., dn) | Random values in a given shape.                              |
| [`randn`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.randn.html#numpy.random.randn)(d0, d1, ..., dn) | Return a sample (or samples) from the “standard normal” distribution. |
| [`randint`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.randint.html#numpy.random.randint)(low[, high, size, dtype]) | Return random integers from *low* (inclusive) to *high* (exclusive). |
| [`random_integers`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.random_integers.html#numpy.random.random_integers)(low[, high, size]) | Random integers of type np.int between *low* and *high*, inclusive. |
| [`random_sample`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample)([size]) | Return random floats in the half-open interval [0.0, 1.0).   |
| [`random`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.random.html#numpy.random.random)([size]) | Return random floats in the half-open interval [0.0, 1.0).   |
| [`ranf`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.ranf.html#numpy.random.ranf)([size]) | Return random floats in the half-open interval [0.0, 1.0).   |
| [`sample`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.sample.html#numpy.random.sample)([size]) | Return random floats in the half-open interval [0.0, 1.0).   |
| [`choice`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html#numpy.random.choice)(a[, size, replace, p]) | Generates a random sample from a given 1-D array             |
| [`bytes`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.bytes.html#numpy.random.bytes)(length) | Return random bytes.                                         |

## linalg 矩阵

> Linear algebra ([`numpy.linalg`](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html))





np.linspace() 比 range() 好，因为它包含了终值

