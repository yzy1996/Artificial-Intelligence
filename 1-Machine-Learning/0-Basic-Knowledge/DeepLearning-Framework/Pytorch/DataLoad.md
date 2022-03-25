# DataLoader 与 DataSet

> 这里仅就**map-style datasets**展开介绍，还有**iterable-style datasets**这里不讲。



## Dataset

以图片为例，我们通常理解的数据集是放在文件夹里的一系列图片，而要转换为Pytorch所能处理的，需要借助 `torch.utils.data.Dataset()` 这样一个类，这个类表示了一个从 索引(index) 到 样本(sample) 的映射(map)。

```python
class Dataset(object):
    def __init__(self):
        ...
        
    def __getitem__(self, index):
        return ...
    
    def __len__(self):
        return ...
```





有了这样一个映射后，我们就只需要决定如何采样index，就能获取图像数据；而这样的采样需要 `Sampler` 完成，可以有顺序的，也可以有乱序的等等







torch.utils.data.DataLoader()



```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

- dataset 支持 `[ | ]` 两种，