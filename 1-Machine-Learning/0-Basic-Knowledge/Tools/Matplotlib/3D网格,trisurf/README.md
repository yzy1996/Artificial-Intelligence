Chinese Tutorial: 

https://www.matplotlib.org.cn/gallery/mplot3d/trisurf3d.html

https://www.matplotlib.org.cn/gallery/mplot3d/trisurf3d_2.html



English Tutorial:

https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html#tri-surface-plots



原理介绍



三角形只跟x, y有关，

自动三角化的方法是-德劳内三角化



先要构成一个Triangulation类，triangles属性能输出三角形的索引，size=[n_size, 3]

