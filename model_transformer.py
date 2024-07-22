import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_heads=4, ff_dim=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # For multihead attention
        x = self.transformer_block1(x)
        x = x.permute(1, 2, 0)  # For pooling
        x = self.global_avg_pool(x).squeeze(-1)
        return self.output_layer(x)