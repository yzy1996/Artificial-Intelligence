# Tensorflow使用指南



查看数据类型，

tensor.dtype

tensor.name

tensor.op

tensor.eval()

tf.shape



## 加载数据集

tensorflow 官方支持了很多数据集，看这里 [Datasets](https://tensorflow.google.cn/datasets/catalog/overview) 

使用一行代码即可，利用 `tfds.load` 构建 `tf.data.Dataset`

```python
<datasetname> = tfds.load(name="<datasetname>")
```

指定下载路径，默认是 `~/tensorflow_datasets`

```python
<datasetname> = tfds.load(name="<datasetname>", data_dir="dir_path")
```





切分数据集可以

```python
mnist_train = tfds.load(name="mnist", split="train")
or
mnist_train = tfds.load(name="mnist")["train"]
```

如果要显示一张图片

```python
for mnist_example in mnist_train.take(1):  # 只取一个样本
    image, label = mnist_example["image"], mnist_example["label"]

    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    print("Label: %d" % label.numpy())
```



最后要输入流水线

```python
mnist_train = mnist_train.shuffle(1024).batch(32)

# prefetch 将使输入流水线可以在模型训练时异步获取批处理。
mnist_train = mnist_train.prefetch(tf.data.experimental.AUTOTUNE)
```

可视化一些样本

```python
mnist_test, info = tfds.load("mnist", split="test", with_info=True)
fig = tfds.show_examples(info, mnist_test)
```



数据处理

```python
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
```

数学运算

```

```



## 1.使用脚本进行转换版本代码

### Single file

```
tf_upgrade_v2 --infile filename.py --outfile filename_v2.py
```

### Directory tree


```bash
# upgrade all the .py files and copy all the other files to the outtree
tf_upgrade_v2 --intree foldername --outtree foldername_v2 --reportfile report.txt
```



## 2.保存并加载模型

```python
model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')
```



## 3.官方文档

[Tensorflow Core r2.0 python api](https://www.tensorflow.org/api_docs/python/tf) 

[Keras中文文档](https://keras.io/zh/)

[Keras Documentation](https://keras.io/)



[简单粗暴 TensorFlow 2](https://tf.wiki/index.html)





tensorflow有两种API使用方式

### 1. Sequential/Functional API 模式建立模型

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
```

```python
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
```



### 2. Subclassing API 

即对 `tf.keras.Model` 类进行扩展以定义自己的新模型，同时手工编写了训练和评估模型的流程

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output
```



 Train and evaluate

```python
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

```python
model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```







显示模型结构并画出

```python
model.summary()

keras.utils.plot_model(model, 'model.svg')

keras.utils.plot_model(model, 'model.svg', show_shapes=True) # display the input and output shapes
```

保存模型

```python
model.save('path_to_my_model')

model = keras.models.load_model('path_to_my_model')
```





## Layer 

Dense 全连接层

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    **kwargs
)
```

It implements the operation: `output = activation(dot(input, kernel) + bias)`

The most common situation is `(batch_size, input_dim)`, and the corresponding output is `(batch_size, units)`

`(batch_size × input_dim) × (input_dim × units)  = (batch_size × units)`

So the kernel (a weight matrix created by the layer) size is `(input_dim, units)`





