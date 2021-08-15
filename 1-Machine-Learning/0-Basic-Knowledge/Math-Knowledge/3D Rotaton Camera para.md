> 注意量纲

真实世界坐标系的坐标



针孔模型是简化模型，简化了什么呢？

没有了景深，

## 三个坐标系

首先要明确我们是要建立三个主要坐标系的关系：

- 世界坐标系 （真实的三维世界）
- 相机坐标系
- 图像坐标系 



### 图像坐标系

图像坐标系又可以分为：图像**像素**坐标系 和 图像**物理**坐标系

<img src="https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210430114322.svg" alt="1111" style="zoom:300%;" />

如上图所示，<u>图像像素坐标系</u>由原点 A 和 坐标轴 $(u, v)$ 构成，坐标单位为 <u>像素</u>；<u>图像物理坐标系</u>由原点 B 和 坐标轴 $(x, y)$ 构成，坐标单位为 <u>毫米</u>。他们之间的关系存在 单位像素的物理尺寸换算，和 位移 这两个关系。

记 B 点 的像素坐标为 $(u_0, v_0)$；C 点的物理坐标为 $(x_1, y_1)$，像素坐标为 $(u_1, v_1)$。

他们的转换关系为：（其中 $dx$ 和 $d_y$ 是 每个/单位 像素的物理尺寸，单位 毫米/像素）
$$
\left\{
\begin{array}{l}
u_1=\frac{x_1}{d x}+u_{0} \\
v_1=\frac{y_1}{d y}+v_{0}
\end{array}
\right.
$$

### 世界坐标系

这就是我们的 $(X_w, Y_w, Z_w)$

### 相机坐标系

相机坐标系的参照物原点是相机本身，我们可以认为是一个光心，光心到像平面的垂线距离为焦距。在这个坐标系下，我们需要建立真实物体的坐标 $(X_c, Y_c, Z_c)$ 以及 图像的物理坐标  $(x_c, y_c[, f])$ 。

这两者的转换关系为：（利用简单的三角相似关系）
$$
\left\{
\begin{array}{l}
x_{c}=f \frac{X_{c}}{Z_{c}} \\
y_{c}=f \frac{Y_{c}}{Z_{c}}
\end{array} \quad \Leftrightarrow \quad
Z_c \left(
\begin{array}{c}
x_{c} \\
y_{c}
\end{array}\right)=
\left(\begin{array}{ccc}
f & 0 \\
0 & f
\end{array}
\right)
\left(\begin{array}{c}
X_c \\
Y_c 
\end{array}\right)\right.
$$
如果要到 图像的像素坐标 $(u_c, v_c)$ ，则：（增加一个平移）
$$
\left\{
\begin{array}{l}
u_c=\frac{f}{dx} \frac{X_c}{Z_c} + u_0\\
v_c=\frac{f}{dy} \frac{Y_c}{Z_c} + v_0
\end{array} \quad \Leftrightarrow \quad
Z_c \left(
\begin{array}{c}
u_{c} \\
v_{c} \\
1
\end{array}\right)=
\left(\begin{array}{ccc}
f_x & 0 & u_0 \\
0 & f_y & v_0 \\
0 & 0 & 1
\end{array}
\right)
\left(\begin{array}{c}
X_c \\
Y_c \\
1
\end{array}\right)\right.
$$
可以注意到这里有一个 $\frac{f}{dx}, \frac{f}{dy}$ ，通常我们见到的会写成 $f_x, f_y$。因为我们很多时候不需要知道CCS的单个像素尺寸。

## 相机参数模型

### 内参矩阵

上式就建立了相机的**内参模型**：（也可以分解成两部分）
$$
\begin{aligned}
s
&=\underbrace{\left(\begin{array}{ccc}
1 & 0 & u_{0} \\
0 & 1 & v_{0} \\
0 & 0 & 1
\end{array}\right)}_{\text {2D Translation }} \times \underbrace{\left(\begin{array}{ccc}
f_{x} & 0 & 0 \\
0 & f_{y} & 0 \\
0 & 0 & 1
\end{array}\right)}_{\text {2D Scaling }}
\end{aligned}
$$

### 外参矩阵

变换过程可以理解为如何让相机坐标系的原点和轴方向与世界坐标系的原点和轴方向重合





可以分解为平移和旋转，

一般记作：$[R \mid t]$，其中 $R$ 是一个 $3 \times 3$ 的旋转矩阵，$t$ 是一个 $3 \times 1$ 的平移量。（这样的写法不是条件概率那样的意思，而只是为了便于表明这个 $3 \times 4$ 矩阵是由什么构成的。
$$
[\mathbf{R} \mid \mathbf{t}] = 
\underbrace{\left[\begin{array}{c|c}
I & \mathbf{t}
\end{array}\right]}_{\text{3D Translation}} 
\times
\underbrace{\left[\begin{array}{c|c}
R & 0 \\
\hline 0 & 1
\end{array}\right]}_{\text{3D Rotation}}
$$

---

相机的内外参矩阵

用的是针孔模型



首先需要一个相机的图

![](https://pic1.zhimg.com/80/v2-4a6c0264ff5beee7ab54c815d0e98b4c_1440w.jpg)







从相机坐标系到像素坐标系

从相机坐标系到世界坐标系

需要旋转和平移

$$
\left(\begin{array}{cc}
R & T \\
0^{3} & 1
\end{array}\right)
$$

$$
P_{c}=\left(\begin{array}{c}
x_{c} \\
y_{c} \\
z_{c} \\
1
\end{array}\right)=\left(\begin{array}{cc}
R & T \\
0^{3} & 1
\end{array}\right)\left(\begin{array}{c}
x_{w} \\
y_{w} \\
z_{w} \\
1
\end{array}\right)=R P_{w}+T
$$







SO3， 旋转矩阵



3D rotation group, often denoted S0(3)

所有环绕着三维欧几里得空间的原点的旋转，组成的群，定义为旋转群。





参考

http://ksimek.github.io/2013/08/13/intrinsic/

https://zhuanlan.zhihu.com/p/144307108









 and 





### SO(3)

In math and geometry, the 3D rotation group is the group of all rotations about the origin of three-dimensional Euclidean space. This group is often denoted SO(3) which is the abbreviation for 'special orthogonal 3-dimensional group'. Every rotation $R$​ can be represented by a mapping from an orthonormal basis of $\mathbb{R}^3$​ to another orthonormal basis. By the way, we can fix one dimension of $R$​ which is called the axis of rotation and then specify it with an axis and an angle of rotation about this axis. For example, counterclockwise rotation about the positive z-axis by angle $\phi$​​ is given by:
$$
R_{z}(\phi)=
\left[\begin{array}{ccc}
\cos \phi & -\sin \phi & 0 \\
\sin \phi & \cos \phi & 0 \\
0 & 0 & 1
\end{array}\right]
$$

### SE(3)

Except rotation, 3D rigid transformation also include a translation $t \in \mathbb{R}^3$​. The group of SO(3) together with a translational part is defined as 'special Euclidean 3-dimensional group', denoted SE(3). To describe the transformation from the camera coordinate system to the world coordinate system, we express the camera extrinsic as a homogenous matrix $T=[R \mid t]$.

special 3D Homography matrix $\mathbf{T}$ constructed with translation \(t\) and rotation \(R\).



To summarize, the set of camera parameters we need to utilize or optimize in our method are the camera intrinsics $f_x$ and $f_y$ shared by all input images, and the camera extrinsics parameterized by $\phi_i$ and $t_i$ specific to each image.







special Euclidean group













