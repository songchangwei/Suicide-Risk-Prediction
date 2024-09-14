import torch
import torch.nn as nn
import torch.nn.functional as F

class Mutiltask_base2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size1,output_size2):
        super(Mutiltask_base2, self).__init__()
        self.nrom = nn.BatchNorm1d(num_features=64)
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        # Attention layer
        self.attention1 = nn.Linear(hidden_size, 1, bias=False)
        self.attention2 = nn.Linear(hidden_size, 1, bias=False)


        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, output_size1)
        self.fc2 = nn.Linear(hidden_size, output_size2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM input: (batch_size, sequence_length, input_size)
        # LSTM output: (batch_size, sequence_length, hidden_size)
        x = self.nrom(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        print(lstm_out.shape)
        
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attention_out = torch.sum(attention_weights * lstm_out, dim=1)
        
        print(attention_out.shape)

        # Attention weights
        attention_weights1 = F.softmax(self.attention1(attention_out), dim=1)
        attention_weights2 = F.softmax(self.attention2(attention_out), dim=1)

        # Attention output: (batch_size, hidden_size)
        attention_out1 = torch.sum(attention_weights1 * attention_out, dim=1)
        attention_out2 = torch.sum(attention_weights2 * attention_out, dim=1)
        
        print(attention_out1.shape)
        print(attention_out2.shape)

        # Fully connected layer
        output1 = self.fc1(attention_out1)
        output2 = self.fc2(attention_out2)
        output1 = self.sigmoid(output1)
        output2 = self.softmax(output2)
        return output1,output2

if __name__=='__main__':
    # Example usage
    input_size = 1280
    hidden_size = 64
    output_size1 = 1
    output_size2 = 16
    sequence_length = 64
    batch_size = 32
    # Create a random input sequence
    input_sequence = torch.rand((batch_size, sequence_length, input_size))
    # Instantiate the model
    model = Mutiltask_base2(input_size, hidden_size, output_size1,output_size2)
    # Forward pass
    output1,output2 = model(input_sequence)
    print("Output shape:", output1.shape,output2.shape)