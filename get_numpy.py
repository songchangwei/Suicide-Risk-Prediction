import numpy as np
import pandas as pd

score = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_score.npy')
suicide = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_suicide.npy')
score_processed = np.where((score >= 0) & (score <= 4), 0, 1)
print(score)
print(score_processed)
print(suicide)
# 将两个数组合并为一个形状为 (2, 310) 的数组
combined_array = np.vstack((suicide, score_processed))
print(combined_array)

# 创建一个 Pandas DataFrame
df = pd.DataFrame(combined_array.T, columns=['true_label', 'prediction'])

# 保存 DataFrame 为 CSV 文件
df.to_csv('AttentionLSTM.csv', index=False)