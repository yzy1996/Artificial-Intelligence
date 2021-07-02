# Predict 3D object shape from several images

这是我们的目标，最理想的情况是 **from a single image**



> 为什么

3维模型建好后，可以生成多视角

借助已有先验信息，预测多视角结果，

应用场景可以是：多视角身份不变性，行人重识别



> 怎么做

当我们在谈论一个2D object 的时候，一幅图像的载体就是像素值 pixel，先不去想彩色图，甚至不是灰度图，而就是黑白图，也就是经过二值化后的灰度图。是不是就能知道这个object的形状。

而3D object，载体可以是体素值（类比像素值）voxel grids；为什么是可以是，因为还可以是point clouds，meshes。为什么呢，因为看到他们我们也可以知道这个object的形状呀。这些表征的特点是：**离散**，对complex geometry 的 fidelity **（保真度）高**。

根据占有关系 (occupancy) 或者 (signed distance)

RGB image as input

