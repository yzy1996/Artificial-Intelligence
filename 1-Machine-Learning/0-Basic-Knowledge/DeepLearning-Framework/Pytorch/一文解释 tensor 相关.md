# 一文解释 tensor 相关



## Table of Contents

- Tensor 和 Numpy 互转
- Tensor 在 CPU 和 GPU 互转



#### Tensor 和 Numpy 互转

> 1. numpy to tensor

```python
a = np.random.randn(2)

# as_tensor是浅拷贝，数据是共享的，因此修改b也会修改a，需要注意
b = torch.as_tensor(a).float()

# 如果不想a, b互相影响，则可以采用
c = torch.tensor(a).float()

# 另外，如果需要转到 GPU，as_tensor也可以使用，因为只有cpu上的数据是共享的
d = torch.as_tensor(a).float().cuda()


torch.from_numpy(np_array)
```



> 2. tensor to numpy  

```python
a = torch.tensor(2.).cuda().requires_grad_()

b = a.detach().cpu().numpy()

# 这里可以发现很严谨的一点是，tensor是先到gpu再让求梯度，回到numpy是先去掉梯度再到cpu
# 如果是在计算图里 则需要 .detach()
# 如果是在GPU上，则需要 .cpu()
# 否则直接 .numpy()
```



#### Tensor 在 CPU 和 GPU 互转

这一点很简单，一般指令就是 `.cuda()`，`.to(device)`，其中 `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`

warning：记得一定是先 cuda，再requires_grad_()



#### Tensor 变数据类型

直接要什么数据类型就在最后加上什么

```python
a = torch.tensor(1., dtype=torch.double).float()
# output dtype=float32
```

如果是对于整个model而言的话是：

model.to(torch.double)



#### 初始化

```python
x = torch.randn(4, 4) # 正态分布
x = torch.rand(4, 4) # 均匀分布
x = torch.zeros(4, 4)
x = torch.ones(4, 4)
x = torch.tensor(4)
x = torch.arange(4)
x = torch.linspace(-10, 10, steps=5)

x = torch.zeros(5, 5).uniform_(-0.5, 0.5) # 得到的其实是一个均匀分布
```



### 查看size

```python
xx.size()
```



### 尺寸变换

```python
xx.view()
xx.reshape()

# view + contigious == reshape
```



### 维度变换

使用einops

```python
# 数据不变，增加一维

# 交换维度
x = torch.randn(2, 3, 5)
x.size()
>>> torch.Size([2, 3, 5])
x.permute(2, 0, 1).size()
>>> torch.Size([5, 2, 3])

# 减小维度
# 去掉“维数为1”的的维度
x.squeeze() # 括号内也可以跟某一维度

# 增加维度
x.unsqueeze(0) # 在最开头增加一维

# 扩展维度
x.expand((3,4)) # 将原本（3，1）变到 （3，4）

tile

# 交换维度
movedim(-1,0)

permute
transpose

transpose与permute的异同

'''
input = size(3, 3)
input.unsqueeze(0).unsqueeze(0) == input.expand(1, 1, 3, 3)
'''
拼接
cat
stack

拆分
chunk
split

[*[1]*2] = [1, 1]
[[1]*2] = [[1, 1]]
[1]*2 = [1, 1]

[[1]*2,2] =[[1, 1], 2]
[*[1]*2,2] = [1, 1, 2]
```



### Embed 初始化

```python
embd = nn.Embedding(5, 10)

# 访问
idx = torch.tensor([1, 2])
embd(idx)

# 查看权重，也就是具体数值
embd.weight.detach().numpy()

# 修改初始化权重
embd1.weight.data.normal_(1, 1) # 由N(0,1)改为N(1, 1)
```



类似于 list.append() 的拼接操作，逐步增加，

```python
a = [] 
for _ in range(2):
	b = ..
	b.unsqueeze_(1)
	a.append(b)
```





one layer with input size:
$$
(N, C, H, W)
$$
where N is a batch size, C denotes a number of channels, H is a height of input planes in pixels, and W is width in pixels.



kernel的组成

