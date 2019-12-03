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

