# Tensorflow使用指南



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

