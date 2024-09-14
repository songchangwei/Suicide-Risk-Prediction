import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import AttentionLSTM,MyModel
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from model2 import LSTMAttentionClassifier
from model3 import TransformerEncoder
from model4 import SIT,PositionalEncoding
import random,os
from torch.optim.lr_scheduler import _LRScheduler, StepLR, ExponentialLR, ReduceLROnPlateau


def make_deterministic(seed):
    # https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


make_deterministic(21)
# 定义设备，优先使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    precision = precision_score(binarized_targets, binarized_predictions, zero_division=0)
    recall = recall_score(binarized_targets, binarized_predictions, zero_division=0)
    f1 = f1_score(binarized_targets, binarized_predictions, zero_division=0)

    return precision, recall, f1

# Assume you have training and testing data (X_train, y_train, X_test, y_test)
X_train = np.load('/home/user416/songcw/data/train_val_test/train/whisper_train_data_fold_3.npy')
y_train = np.load('/home/user416/songcw/data/train_val_test/train/whisper_train_suicide_fold_3.npy').reshape((-1,1))
X_test = np.load('/home/user416/songcw/data/train_val_test/val/whisper_val_data_fold_3.npy')
y_test = np.load('/home/user416/songcw/data/train_val_test/val/whisper_val_suicide_fold_3.npy').reshape((-1,1))

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''
# Instantiate the model, loss function, and optimizer
input_size = 1280
hidden_size = 128
output_size = 1
hidden_size1 = 64
hidden_size2 = 32

#model = AttentionLSTM(input_size, hidden_size, output_size)
model = LSTMAttentionClassifier(input_size, hidden_size, output_size, 0.5)
#model = MyModel(input_size, hidden_size1, hidden_size2, output_size)
'''

# 定义超参数
num_patches = 64  # 假设输入由256个 patches 组成
embed_dim = 1280   # 嵌入的维度
num_heads = 10    # 注意力头的数量
num_layers = 8    # 编码器层的数量
num_classes = 1  # 分类的类别数

# 创建 Transformer 模型实例
model = TransformerEncoder(
    num_patches=num_patches,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes
).to(device)

'''
model = SIT(
        sequece_length=64,
        num_classes=1,
        dim=1280,
        depth=6,
        heads=8,
        mlp_dim = 512
).to(device)
'''


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.05)

# 固定步长衰减
scheduler_step = StepLR(optimizer, step_size=100, gamma=0.9)

# 指数衰减
scheduler_exp = ExponentialLR(optimizer, gamma=0.95)

# 基于验证集性能的动态衰减（如果验证集损失没有改善，则降低学习率）
scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

# 例子中使用的参数
d_model = 1280  # 嵌入的维度
max_len = 64 # 可以处理的最大序列长度
pos_encoder = PositionalEncoding(d_model, max_len, device)


# Training loop
num_epochs = 100
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
        #batch_X = batch_X + pos_encoder(batch_X)

        optimizer.zero_grad()
        output = model(batch_X)
        #print(output.shape,batch_y.shape)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()  # 更新权重
        total_loss += loss.item()
    # 更新学习率（固定步长衰减或指数衰减）
    #scheduler_step.step()  # 或者 scheduler_exp.step()




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
            #batch_X = batch_X + pos_encoder(batch_X)

            output = model(batch_X)
            loss = criterion(output, batch_y)
            test_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

        
        #print(binarized_targets)
        #print(binarized_predictions)

        
        # 更新学习率（基于验证集的性能）
        #scheduler_plateau.step(test_loss / len(test_loader))
        # 查看优化器的所有参数组的当前学习率
        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])
            
        # 计算不同阈值下的F1分数
        local_thresholds = np.linspace(0, 1, num=100)
        local_best_threshold = 0
        local_best_f1 = 0
        local_best_precision = 0
        local_best_recall = 0
        local_best_accuracy = 0

        for local_threshold in local_thresholds:
            # 将概率转换为二进制预测
            binarized_predictions = binarize_labels(np.array(predictions), local_threshold).reshape((-1))
            binarized_targets = binarize_labels(np.array(targets), 0.5).reshape((-1))
            precision = precision_score(binarized_targets, binarized_predictions, zero_division=0)
            recall = recall_score(binarized_targets, binarized_predictions, zero_division=0)
            f1 = f1_score(binarized_targets, binarized_predictions, zero_division=0)
            accuracy = accuracy_score(binarized_targets, binarized_predictions)
            #print(binarized_targets)
            #print(binarized_predictions)
            if local_best_f1 < f1:
                local_best_precision = precision
                local_best_recall = recall
                local_best_accuracy = accuracy
                local_best_f1 = f1
                local_best_threshold = local_threshold
                

        print(f'Test Loss: {test_loss / len(test_loader)}')
        print('precision:',local_best_precision,'recall:',local_best_recall,'f1:',local_best_f1,'accuracy',local_best_accuracy,'local_best_threshold',local_best_threshold)

        if best_f1 < local_best_f1:
            best_accuracy = local_best_accuracy
            best_recall = local_best_recall
            best_precision = local_best_precision
            best_f1 = local_best_f1
            best_threshold = local_best_threshold
            torch.save(model.state_dict(), 'TransformerEncoder/model_weight_3.pth')
print("最好的结果：",'precision:',best_precision,'recall:',best_recall,'f1:',best_f1,'accuracy',best_accuracy,'best_threshold',best_threshold)
