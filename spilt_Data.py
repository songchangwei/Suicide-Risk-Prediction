import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# 设定数据集的路径
images_path = '/home/user416/songcw/data/naochuxue/dataset/image/'  # 图片文件夹路径
labels_path = '/home/user416/songcw/data/naochuxue/dataset/label/'  # 标签文件夹路径

# 设定划分比例
train_size = 0.7
val_size = 0.15
test_size = 0.15

# 获取文件夹内的文件名
images = os.listdir(images_path)
labels = os.listdir(labels_path)

# 确保文件名匹配
assert len(images) == len(labels), "The number of images and labels does not match!"
assert all(os.path.splitext(im)[0] == os.path.splitext(lb)[0] for im, lb in zip(images, labels)), "Image and label file names do not match!"

# 划分训练集、验证集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size + val_size, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=test_size/(test_size + val_size), random_state=42)

# 创建文件夹
dataset_paths = {
    'train': ('train_images', 'train_labels'),
    'val': ('val_images', 'val_labels'),
    'test': ('test_images', 'test_labels')
}

for phase in ['train', 'val', 'test']:
    for data_type in ['images', 'labels']:
        path = os.path.join('dataset', phase, data_type)
        if not os.path.exists(path):
            os.makedirs(path)

# 复制文件到对应的文件夹
def copy_files(files, source, destination):
    for file in files:
        shutil.copy(os.path.join(source, file), os.path.join(destination, file))

copy_files(train_images, images_path, 'train_val_test/train/images')
copy_files(val_images, images_path, 'train_val_test/val/images')
copy_files(test_images, images_path, 'train_val_test/test/images')
copy_files(train_labels, labels_path, 'train_val_test/train/labels')
copy_files(val_labels, labels_path, 'train_val_test/val/labels')
copy_files(test_labels, labels_path, 'train_val_test/test/labels')

print("Files have been successfully copied into train, val, and test folders.")
