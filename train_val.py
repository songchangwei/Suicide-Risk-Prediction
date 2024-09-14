import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import AttentionLSTM,MyModel
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from model2 import LSTMAttentionClassifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# Assume you have training and testing data (X_train, y_train, X_test, y_test)
X_train = np.load('/home/user416/songcw/data/train_val_test/train/whisper_train_data_fold_5.npy')
y_train = np.load('/home/user416/songcw/data/train_val_test/train/whisper_train_suicide_fold_5.npy').reshape((-1,1))
X_test = np.load('/home/user416/songcw/data/train_val_test/val/whisper_val_data_fold_5.npy')
y_test = np.load('/home/user416/songcw/data/train_val_test/val/whisper_val_suicide_fold_5.npy').reshape((-1,1))

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, loss function, and optimizer
input_size = 1280
hidden_size = 128
output_size = 1
hidden_size1 = 64
hidden_size2 = 32

model = AttentionLSTM(input_size, hidden_size, output_size).to(device)
#model = LSTMAttentionClassifier(input_size, hidden_size, output_size, 0.5)
#model = MyModel(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00001,weight_decay=0.01)


# Training loop
num_epochs = 500
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0
best_accuracy = 0.0
best_threshold = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0


    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        #print(output.shape,batch_y.shape)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()




    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    # Testing loop
    model.eval()
    test_loss = 0

    with torch.no_grad():
        predictions = []
        targets = []
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

        binarized_predictions = binarize_labels(np.array(predictions), 0.5).reshape((-1))
        binarized_targets = binarize_labels(np.array(targets), 0.5).reshape((-1))
        #print(binarized_targets)
        #print(binarized_predictions)

        precision = precision_score(binarized_targets, binarized_predictions)
        recall = recall_score(binarized_targets, binarized_predictions)
        f1 = f1_score(binarized_targets, binarized_predictions)
        accuracy = accuracy_score(binarized_targets, binarized_predictions)

        print(f'Test Loss: {test_loss / len(test_loader)}')
        print('precision:',precision,'recall:',recall,'f1:',f1,'accuracy',accuracy)

        if best_f1 < f1:
            best_accuracy = accuracy
            best_recall = recall
            best_precision = precision
            best_f1 = f1
            torch.save(model.state_dict(), 'AttentionLSTM/model_weight_5.pth')
print("最好的结果：",'precision:',best_precision,'recall:',best_recall,'f1:',best_f1,'accuracy',best_accuracy)