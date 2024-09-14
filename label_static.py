import numpy as np

# 替换成你的.npy文件路径
file_path = '/home/user416/songcw/data/train_val_test/test/whisper_test_suicide.npy'

# 加载.npy文件
data = np.load(file_path)

# 统计0和1的数量
count_zeros = np.sum(data == 0)
count_ones = np.sum(data == 1)

print(f"0的个数: {count_zeros}")
print(f"1的个数: {count_ones}")
