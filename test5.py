import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pandas as pd
from model5 import GRUModel,GRUModel,BiGRUModel,BiGRUModel


torch.manual_seed(42)


# Function to create binary labels based on a threshold
def binarize_labels(predictions, threshold=0.5):
    return (predictions > threshold).astype(int)


# Function to calculate precision, recall, and F1 score for binary classification
def calculate_classification_metrics(model, data_loader, threshold=0.5):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            output = model(batch_X)
            predictions.extend(output.numpy())
            targets.extend(batch_y.numpy())

    binarized_predictions = binarize_labels(np.array(predictions), threshold)
    binarized_targets = binarize_labels(np.array(targets), threshold)

    precision = precision_score(binarized_targets, binarized_predictions)
    recall = recall_score(binarized_targets, binarized_predictions)
    f1 = f1_score(binarized_targets, binarized_predictions)

    return precision, recall, f1

X_test = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_data.npy')
y_test = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_suicide.npy').reshape((-1,1))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 64
input_dim = 1280
hidden_dim = 512
output_dim = 1
num_layers = 2

model = GRUModel(input_dim, hidden_dim, output_dim, num_layers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load('GRUModel/model_weight_2.pth'))

num_epochs = 500
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0
best_accuracy = 0.0

model.eval()
test_loss = 0
criterion = nn.BCELoss()

with torch.no_grad():
        predictions = []
        targets = []
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
            predictions.extend(output.detach().numpy())
            #print(output.detach().numpy())
            targets.extend(batch_y.detach().numpy())

        binarized_predictions = binarize_labels(np.array(predictions), 0.5).reshape((-1))
        binarized_targets = binarize_labels(np.array(targets), 0.5).reshape((-1))
        #print(binarized_targets)
        #print(binarized_predictions)
        print(np.array(predictions))
        df = pd.DataFrame({'预测值':np.array(predictions).reshape((-1)),'真实值':binarized_targets})
        df.to_csv('GRUModel/probalility_2.csv',index=False)

        precision = precision_score(binarized_targets, binarized_predictions)
        recall = recall_score(binarized_targets, binarized_predictions)
        f1 = f1_score(binarized_targets, binarized_predictions)
        accuracy = accuracy_score(binarized_targets, binarized_predictions)
        
        # 将两个数组合并为一个形状为 (2, 310) 的数组
        combined_array = np.vstack((binarized_targets, binarized_predictions))
        print(combined_array)

        # 创建一个 Pandas DataFrame
        df = pd.DataFrame(combined_array.T, columns=['true_label', 'prediction'])

        # 保存 DataFrame 为 CSV 文件
        df.to_csv('GRUModel.csv', index=False)

        print(f'Test Loss: {test_loss / len(test_loader)}')
        print('precision:',precision,'recall:',recall,'f1:',f1,'accuracy',accuracy)
