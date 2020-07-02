# Pytorch使用指南



## 1.查看版本

```bash
import torch
print(torch.__version__)
```



## 2. 基本函数

torch.tensor 具有属性

```python
# 是否可以求导
requires_grad=True/False 

# 操作名称
grad_fn = <AddBackward0> / <MulBackward0> / <MeanBackward0> / <SumBackward0>
```

计算导数使用

```python
# 计算
.backward()

# 结果
.grad
```



## 3. 搭建Le-Net

