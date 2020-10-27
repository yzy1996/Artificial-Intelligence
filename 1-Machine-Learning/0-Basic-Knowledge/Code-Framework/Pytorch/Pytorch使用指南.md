# Pytorch使用指南

学习教程

https://github.com/lyhue1991/eat_pytorch_in_20_days

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







## 存储模型

```python
# 保存整个网络
torch.save(model, 'model.pkl') 
# 保存网络中的参数, 速度快，占空间少
torch.save(model.state_dict(), 'model_parameter.pkl')
#--------------------------------------------------
#针对上面一般的保存方法，加载的方法分别是：
model_dict=torch.load('model.pkl')
model_dict=model.load_state_dict(torch.load('model_parameter.pkl'))
```





## 张量

创建一个张量



获得张量元素 `item()`





### view()

相当于reshape

```python
t = torch.rand(4, 4)
b = t.view(2, 8)

>>> tensor([[0.1263, 0.5635, 0.4487, 0.6130, 0.3470, 0.7098, 0.6912, 0.9653],
        [0.2700, 0.3094, 0.6009, 0.0180, 0.5841, 0.0036, 0.8699, 0.4085]])
```

