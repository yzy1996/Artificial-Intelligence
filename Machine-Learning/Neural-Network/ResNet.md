<h1 align="center">Residual Network</h1>
<div align="center">


![python-version](https://img.shields.io/badge/python-3.7-blue) ![country](https://img.shields.io/badge/country-China-red)

Yann LeCun, Leon Bottou, Yosuha Bengio and Patrick Haffner proposed a neural network architecture for handwritten and machine-printed character recognition in 1990’s which they called LeNet-5.

published: 《Gradient-based learning applied to document recognition》

</div>

Architecture of LeNet-5:![Figure](https://www.researchgate.net/profile/Vladimir_Golovko3/publication/313808170/figure/fig3/AS:552880910618630@1508828489678/Architecture-of-LeNet-5_W640.jpg)

### Description

input is a 32x32 pixel image

**C1-First Layer**(convolutional layer):  

input = 32×32×1, number = 6, size = 5×5, padding = 0, stride = 1, output = 28×28×6

trainable parameters = weight + bias= 5×5×6+6 = (size+bias)×number = (5×5+1)×6 = 156

connections = 28×28×156 = output×(size + bias)×number = 28×28×(5×5+1)×6 = 1222304.



**S2-Second Layer**(pooling layer):  

sigmoid(a×average(x)+b)​

input = 28×28×6, number = 6, size = 2×2, padding = 0, stride = 2, output = 14×14×6

trainable parameters =  (coefficient + bias)×number = (1+1)×6 =12

connections = output×(size + bias)×number = 14×14×(2×2+1)×6 = 5880 【为什么不是乘以12】【参数少，但计算次数没变】



**C3-Third Layer**(convolutional layer):  

input = 14×14×6, number = 16, size = 5×5, padding = 0, stride = 1, output = 10×10×6

In this layer, only 10 out of 16 feature maps are connected to 6 feature maps of the previous layer 【是通过叠加的方式吗】【应该是】

trainable parameters =  weight + bias = (5×5×6×10)+16 = 6×(3×25+1)+6×(4×25+1)+3×(4×25+1)+(25×6+1) = 1516

connections = 10×10×1516=151600



**S4-Fourth Layer**(pooling layer):  

input = 10×10×16, number = 16, size = 2×2, padding = 0, stride = 2, output = 5×5×16

trainable parameters =  (coefficient + bias)×number = (1+1)×16 =32, connections = 5×5×80=2000



**C5-Fifth Layer**(fully connected convolutional layer)

input = 5×5×16, number = 120, size = 5×5, padding = 0, stride = 1, output = 1×1×120

trainable parameters = 5×5×16×120+120 = 48120

connections  = trainable parameters = 48120



**F6-Sixth Layer**(fully connected layer)

input = 1×1×120, output = 1×1×84

trainable parameters = 120×84+84 = 10164



**O7-Seventh Layer**(fully connected softmax)



### Implementation

see the [code](https://github.com/yzy1996/Artificial-Intelligence/blob/master/Machine-Learning/Image-Classification/Mnist/tensorflow/train_cnn.py)



### reference: 

https://www.cnblogs.com/34fj/p/9469784.html

https://engmrk.com/lenet-5-a-classic-cnn-architecture/



