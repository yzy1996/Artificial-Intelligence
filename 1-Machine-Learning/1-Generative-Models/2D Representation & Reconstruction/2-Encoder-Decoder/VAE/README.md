# VAE

VQ-VAE

[Neural Discrete Representation Learning]()  
*Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu*  
**[`NeurIPS 2017`] (`DeepMind`)**



什么是 Vector Quantization

计算机智能处理离散的数字信号，所以在将模拟信号转换为数字信号时，可以用区间内的某一个值去代替一个区间，比如：[0, 1]上的值全变为0，[1,2]上的值全变为1。这样VQ就将一个向量空间中的点用其中一个有限子集来进行编码的过程。

【这样其实实现的是一种压缩的效果】

