import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init



class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len,_ = x.size()
        # [batch_size = 128, seq_len = 30]
        out = self.encoding[:seq_len, :]
        
        # 首先增加一个新的维度，使形状变为 [1, 64, 1280]
        out_unsqueezed = out.unsqueeze(0)

        # 使用 expand 方法扩展新的维度到 64，形状变为 [batch_size, 64, 1280]
        x_expanded = out_unsqueezed.expand(batch_size, -1, -1)

        return x_expanded
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class SIT(nn.Module):
    def __init__(self, sequece_length, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=64)
        

        #self.pos_embedding = nn.Parameter(torch.randn(1, sequece_length + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        #self.fc = nn.Linear(dim,mlp_dim)


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        #self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            #nn.Linear(dim, mlp_dim//2), # 第一个全连接层，
            #nn.ReLU(),                                # 非线性激活函数ReLU
            #nn.Dropout(p=0.5),                        # Dropout层，丢弃率设置为0.5
            nn.Linear(dim, num_classes)  # 第二个全连接层
        )
        self.sigmoid = nn.Sigmoid()
        #self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用适当的初始化方法，例如kaiming初始化或xavier初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # 对于自定义的任何其他层或参数，你可以根据需要添加初始化代码

    def forward(self, x):
        x = self.batch_norm(x)

        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #x += self.pos_embedding[:, :(n + 1)]
        #x = self.dropout(x)
        #x = self.fc(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = self.to_latent(x)
        x = self.mlp_head(x)
        out = self.sigmoid(x) 
        return out
    
    
if __name__=='__main__':

    # 创建 Transformer 模型实例
    model = SIT(
        sequece_length=64,
        num_classes=1,
        dim=1280,
        depth=6,
        heads=10,
        mlp_dim = 512
        
    )
    model = PositionalEncoding(
        d_model = 1280,
        max_len = 64,
        device = 'cpu'
    )

    # 示例用法
    src = torch.randn(128, 64,1280)  # 假定输入大小为 [batch_size, num_patches, embed_dim]
    preds = model(src)  # 获取预测结果
    print(preds.shape)  # 打印预测结果