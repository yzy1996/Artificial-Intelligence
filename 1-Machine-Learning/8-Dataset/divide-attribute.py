'''
按属性切分CelebA数据集，给定属性，划分为正样本和负样本
'''

import os
import shutil
from pathlib import Path


def main():

    positive_dir = output_path / 'positive'
    negative_dir = output_path / 'negative'

    if not os.path.isdir(positive_dir):
        os.makedirs(positive_dir)
    if not os.path.isdir(negative_dir):
        os.makedirs(negative_dir)

    with open(attribute_path) as f:
        lines = f.readlines()
        lines = lines[2:]

        for line in lines:
            info = line.split()
            filename = info[0]
            filepath_old = data_path / filename

            if os.path.isfile(filepath_old):
                if int(info[attribute_type]) == 1:
                    filepath_new = positive_dir / filename
                    shutil.copyfile(filepath_old, filepath_new)

                else:
                    filepath_new = negative_dir / filename
                    shutil.copyfile(filepath_old, filepath_new)




if __name__ == "__main__":

    # 设定要切分的类别
    attribute_type = 21  # Male

    # 设置路径
    path = Path('D:/Data/Face/celeba')
    data_path = path / 'image'
    attribute_path = path / 'attribute.txt'
    output_path = path / 'Male'

    main()
