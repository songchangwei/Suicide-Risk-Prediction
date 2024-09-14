import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# 加载.npy文件
y_true = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_suicide.npy')  # 真实标签
scores = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_score.npy')    # 分数

# 将1-8的分数转换为0，9-16的分数转换为1
y_pred = np.where(scores <= 8, 0, 1)

# 计算性能指标
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)

# 输出结果
print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
