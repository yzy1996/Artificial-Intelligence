# Tensorboard

对应的文件是 `events.out.tfevents.1611103773.xx`



运行命令：

```python
tensorboard --logdir [folder]
```

打开网页：

```html
http://localhost:6006/
```





服务器端使用

```
ssh -L 16006:127.0.0.1:6006 username@remote_server_ip
tensorboard --logdir=XXX --port=6006
127.0.0.1:16006/
```

