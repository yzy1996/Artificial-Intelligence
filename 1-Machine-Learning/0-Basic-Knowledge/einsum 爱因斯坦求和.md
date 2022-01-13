# einsum 爱因斯坦求和

主要优势体现在处理处理关于坐标的方程式



支持 `numpy`, `pytorch`, `tensorflow`

## 基本用法



### 转置

```python
import numpy as np

a = np.arange(9).reshape(3, 3)
b = np.einsum('ij->ji', a)

print(b)

>>>
[[0 3 6]
[1 4 7]
[2 5 8]]
```



### 所有元素求和

```python
import numpy as np

a = np.arange(9).reshape(3, 3)
b = np.einsum('ij->', a)

print(b)

>>> 36
```



### 按列求和

$\sum_j a_{ij}$

```python
import numpy as np

a = np.arange(9).reshape(3, 3)
b = np.einsum('ij->i', a)
# equal to np.sum(a, axis=1)
print(b)

>>> [ 3 12 21]
```

> 有那么一丝丝反过来表示的感觉。其实`ij->i`也就是说沿着 j (第1维，axis=1) 求和，就还剩下 i。



### 矩阵对应维度相乘

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
b = np.arange(4).reshape(4)
c = np.einsum('ij,j->ij', a, b)
print(c)

>>> [[ 0  1  4  9]
 [ 0  5 12 21]
 [ 0  9 20 33]]
```





$a \cdot b$

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
b = np.arange(4).reshape(4)
c = np.einsum('ij,j->i', a, b)
print(c)

>>> [14 38 62]
```

> 相当于先乘，然后沿着j维求和