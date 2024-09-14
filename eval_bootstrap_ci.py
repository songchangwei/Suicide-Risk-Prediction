import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import bootstrap

# 读取CSV文件
data = pd.read_csv('mutiltask_base/probalility_5.csv')
predicted = data['预测值'].values
actual = data['真实值'].values

# 将预测概率转换为二分类预测（阈值设为0.5）
threshold = 0.5
binary_predicted = (predicted > threshold).astype(int)

# 定义计算指标的函数
def calculate_metrics(binary_predicted, actual):
    precision = precision_score(actual, binary_predicted)
    recall = recall_score(actual, binary_predicted)
    f1 = f1_score(actual, binary_predicted)
    accuracy = accuracy_score(actual, binary_predicted)
    return precision, recall, f1, accuracy

# 使用Bootstrap计算指标的95%置信区间
data = (binary_predicted, actual)
bootstrap_ci = bootstrap(data, calculate_metrics, n_resamples=1000, confidence_level=0.95, method='percentile')

# 输出结果
metrics = calculate_metrics(binary_predicted, actual)
print(f"Precision: {metrics[0]}")
print(f"Recall: {metrics[1]}")
print(f"F1-score: {metrics[2]}")
print(f"Accuracy: {metrics[3]}")

print("95% Bootstrap Confidence Intervals:")
print(f"Precision: {bootstrap_ci.confidence_interval[0][0]} - {bootstrap_ci.confidence_interval[1][0]}")
print(f"Recall: {bootstrap_ci.confidence_interval[0][1]} - {bootstrap_ci.confidence_interval[1][1]}")
print(f"F1-score: {bootstrap_ci.confidence_interval[0][2]} - {bootstrap_ci.confidence_interval[1][2]}")
print(f"Accuracy: {bootstrap_ci.confidence_interval[0][3]} - {bootstrap_ci.confidence_interval[1][3]}")
