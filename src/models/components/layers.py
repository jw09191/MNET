import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class MlpHead(nn.Module):
    def __init__(self, d_model=512, out_dim=25*6, mlp_dim=2048, tanh=False):
        super().__init__()
        self.tanh = tanh
        self.op = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, out_dim))

    def forward(self, x):
        if self.tanh:
            return torch.tanh(self.op(x))
        else:
            return self.op(x)


class ModulatedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, x, w):
        return self.gamma * w * self.norm(x) + self.beta * w


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MappingNet(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        self.unshared = nn.ModuleList()
        for _ in range(10):
            self.unshared.append(nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim)
            ))

    def forward(self, x, genre):
        s = self.shared(x)
        sList = []
        for unshare in self.unshared:
            sList.append(unshare(s))
        s = torch.stack(sList, dim=1)
        idx = torch.LongTensor(range(len(genre))).to(genre.device)
        return s[idx, genre]


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.pe = nn.Parameter(torch.randn(480, 480))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k=None, v=None):
        b, n, _, h = *q.shape, self.heads
        k = q if k is None else k
        v = q if v is None else v

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        _, _, dots_w, dots_h = dots.shape
        dots += self.pe[:dots_w, :dots_h]
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim), Attention(dim, heads, dropout),
                nn.LayerNorm(dim), FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for ln1, attn, ln2, ff in self.layers:
            x = x + attn(ln1(x))
            x = x + ff(ln2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim), Attention(dim, heads, dropout),
                nn.LayerNorm(dim), Attention(dim, heads, dropout),
                nn.LayerNorm(dim), FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, w):
        for ln1, attn1, ln2, attn2, ln3, ff in self.layers:
            x = x + attn1(ln1(x))
            x = x + attn2(ln2(x), w, w)
            x = x + ff(ln3(x))
        return x
