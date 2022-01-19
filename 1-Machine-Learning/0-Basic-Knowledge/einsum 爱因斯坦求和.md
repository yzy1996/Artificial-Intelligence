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





PyTorch尝试

```python
import torch
A = torch.randn(16, 8, 5, 128, 128)
B = torch.randn(16, 8, 5, 128, 128)
print('A:', A.size())
print('B:', B.size())
A = A.unsqueeze(3)
B = B.unsqueeze(2)
print('Viewed A:', A.size())
print('Viewed B:', B.size())
C = torch.einsum('ijklno,ijlmno->ijkmno', [A, B])
print('C:', C.size())

Output：
A: torch.Size([16, 8, 5, 128, 128])
B: torch.Size([16, 8, 5, 128, 128])
Viewed A: torch.Size([16, 8, 5, 1, 128, 128])
Viewed B: torch.Size([16, 8, 1, 5, 128, 128])
C: torch.Size([16, 8, 5, 5, 128, 128])
```

