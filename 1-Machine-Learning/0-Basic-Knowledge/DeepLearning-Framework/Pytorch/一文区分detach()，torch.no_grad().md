`detach()`

会返回一个新的Tensor对象，不会在反向传播中出现

```python
x = torch.tensor(2., requires_grad=True)
y = 2 * x.detach()
print(y.requires_grad) # False
```





还可以赋值切断



与 data() detach() 比较









`torch.no_grad()`

通常是在推断(inference)的时候，用来禁止梯度计算，仅进行前向传播

```python
x = torch.tensor(2., requires_grad=True)
with torch.no_grad():
	y = 2 * x
print(y.requires_grad) # False
```





在训练过程中，`torch.no_grad()`就像画了个圈，来，在我这个圈里面跑一下，都不需要计算梯度，就正向传播一下。

`detach()`是相当于复制了一个变量，将它原本`requires_grad=True`变为了`requires_grad=False`

那是不是我对初始起点的值用一下 detach()，然后就跟`torch.no_grad()`一样了呢？





```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = 2 * x * y
print(z.requires_grad) # True
```

```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
with torch.no_grad():
	z = 2 * x * y
print(z.requires_grad) # False
```

```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = 2 * x * y.detach() 
print(z.requires_grad) # True
```

```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = 2 * x.detach() * y.detach() 
print(z.requires_grad) # False
```

