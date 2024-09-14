import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig


class MambaClassifier(nn.Module):
    def __init__(self, d_model,  n_layers=2,d_state =16, d_conv=4, dropout=0.5, num_classes=1):
        super(MambaClassifier, self).__init__()
        self.d_model = d_model
        
        
        # Transformer Encoder Layer
        mambaConfig = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv
        )
        self.mamba = Mamba(mambaConfig)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, src, src_key_padding_mask=None):

        
        output = self.mamba(src)
        
        # 取编码器的第一个序列输出
        output = output.mean(dim=1)
        
        # 分类
        logits = self.classifier(output)
        logits = self.sigmoid(logits)

        return logits


if __name__=='__main__':
    
    model = MambaClassifier(
        d_model=1280,
        n_layers=2,
        d_state=16,
        d_conv=4,
        dropout=0.1,
        num_classes=1
    )
    B, L, D = 64, 64, 1280
    x = torch.randn(B, L, D)
    y = model(x)
    print(x.shape)
    print(y.shape)