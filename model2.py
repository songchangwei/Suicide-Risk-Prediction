import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, lstm_output, hidden):
        # lstm_output: [batch_size, seq_len, hidden_size]
        # hidden: [1, batch_size, hidden_size]
        hidden = hidden[-1]  # 取最后一层的hidden state
        attn_weights = F.softmax(self.attn(lstm_output), dim=2)
        attn_applied = torch.sum(torch.bmm(attn_weights.transpose(1, 2), lstm_output).squeeze(1),dim=1)
        return attn_applied, attn_weights


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1, dropout_prob=0.5):
        super(LSTMAttentionClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=64)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        # 初始化权重
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)
        # 初始化LSTM和Attention层的权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias' in name:
                init.zeros_(param)
        # 初始化Attention权重
        init.xavier_uniform_(self.attention.attn.weight)
        init.zeros_(self.attention.attn.bias)

    def forward(self, x):
        x = self.batch_norm(x)
        # LSTM层
        lstm_out, (hidden, _) = self.lstm(x)
        # 添加层归一化
        #lstm_out = self.layer_norm(lstm_out)

        # Attention层
        attn_output, attn_weights = self.attention(lstm_out, hidden)

        # Dropout层
        attn_output = self.dropout(attn_output)

        # 全连接层
        out = F.relu(self.fc1(attn_output))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__=='__main__':
    #X_train = np.load('C:/Users/admin/train_val_test/train_val_test/train/whisper_train_data_fold_5.npy')
    #y_train = np.load('C:/Users/admin/train_val_test/train_val_test/train/whisper_train_suicide_fold_5.npy').reshape(
    #    (-1, 1))
    #X_test = np.load('C:/Users/admin/train_val_test/train_val_test/val/whisper_val_data_fold_5.npy')
    #y_test = np.load('C:/Users/admin/train_val_test/train_val_test/val/whisper_val_suicide_fold_5.npy').reshape((-1, 1))
    #print(X_train.shape)


    # 假设输入维度为10，序列长度为50，隐藏层大小为128，分类数为2
    input_size = 1280
    hidden_size = 128
    num_classes = 1
    seq_length = 64
    dropout_prob = 0.5

    # 创建模型
    model = LSTMAttentionClassifier(input_size, hidden_size, num_classes, dropout_prob)

    # 输入数据
    input_tensor = torch.randn(1, seq_length, input_size)

    # 获取网络输出
    output, attn_weights = model(input_tensor)

    # 查看输出形状
    print("Output shape:", output.shape)
    print("Attention shape:", attn_weights.shape)
