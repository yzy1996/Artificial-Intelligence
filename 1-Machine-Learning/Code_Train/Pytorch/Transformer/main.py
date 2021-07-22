from os import name
import torch
from torch import nn
from torch import tensor
from torch._C import TensorType
import torch.nn.functional as f
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# (batch_size, seq_length, num_features)
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor: 
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k, dim_v):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

def position_encoding(seq_len: int, dim_model: int) -> Tensor:
    pose = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)

    # pytorch >1.7 support 'rounding_mode'
    if torch.__version__ >= '1.8':
        phase = pose / 1e4 ** (2 * (torch.div(dim, 2, rounding_mode='floor')) / dim_model)
    else:
        phase = pose / 1e4 ** (2 * (dim // 2) / dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

# 残差模块
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

# Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 num_heads: int = 8,
                 dim_model: int = 512, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        dim_q = dim_k = dim_v = dim_model // num_heads

        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout)

        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dim_model: int = 512, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(num_heads, dim_model, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
        return src

# Decoder 
# input: [target, memorry]
class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_heads: int = 8,
                 dim_model: int = 512,  
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        dim_q = dim_k = dim_v = dim_model // num_heads

        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout)

        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout)

        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dim_model: int = 512,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(num_heads, dim_model, dim_feedforward, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return torch.softmax(self.linear(tgt), dim=-1)

class Transformer(nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8, 
        dim_model: int = 512, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))

if __name__ == '__main__':
    
    src = torch.rand(64, 16, 512, device=device)
    tgt = torch.rand(64, 16, 512, device=device)
    out = Transformer().to(device)
    out = out(src, tgt)
    print(out.size())
    