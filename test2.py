import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import AttentionLSTM,MyModel
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pandas as pd
from model2 import LSTMAttentionClassifier

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

#X_test = np.load('/home/user416/songcw/data/train_val_test/test/all_test2.npy')
#y_test = np.load('/home/user416/songcw/data/train_val_test/test/whisper_test_suicide.npy').reshape((-1,1))
#y_test = np.full((521, 1), 1)

print(X_test.shape,y_test.shape)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 64
input_size = 1280
hidden_size = 128
output_size = 1
hidden_size1 = 64
hidden_size2 = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = (LSTMAttentionClassifier(input_size, hidden_size, output_size,0.5))
model.load_state_dict(torch.load('LSTMAttentionClassifier/model_weight_2.pth'))

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
            #loss = criterion(output, batch_y)
            #test_loss += loss.item()
            predictions.extend(output.detach().numpy())
            #print(output.detach().numpy())
            targets.extend(batch_y.detach().numpy())

        binarized_predictions = binarize_labels(np.array(predictions), 0.5).reshape((-1))
        binarized_targets = binarize_labels(np.array(targets), 0.5).reshape((-1))
        #print(binarized_targets)
        #print(binarized_predictions)
        print(np.array(predictions))
        df = pd.DataFrame({'预测值':np.array(predictions).reshape((-1))})
        df.to_csv('LSTMAttentionClassifier/test_probalility_2.csv',index=False)

        precision = precision_score(binarized_targets, binarized_predictions)
        recall = recall_score(binarized_targets, binarized_predictions)
        f1 = f1_score(binarized_targets, binarized_predictions)
        accuracy = accuracy_score(binarized_targets, binarized_predictions)

        print(f'Test Loss: {test_loss / len(test_loader)}')
        print('precision:',precision,'recall:',recall,'f1:',f1,'accuracy',accuracy)
