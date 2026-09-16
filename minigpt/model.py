import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """Token embedding + learnable position embedding."""

    def __init__(self, vocab_size, max_seq_len, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x):
        _, seq_len = x.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.position_embedding.num_embeddings}."
            )

        token_embedding = self.token_embedding(x)
        position_ids = torch.arange(seq_len, device=x.device)
        position_embedding = self.position_embedding(position_ids).unsqueeze(0)
        return token_embedding + position_embedding


class SelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask):
        batch_size, seq_len, embedding_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_len, embedding_dim
        )
        return self.out_proj(attention_output)


class FeedForward(nn.Module):
    """Position-wise MLP used inside each Transformer block."""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with residual connections."""

    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        attention_output = self.attention(self.layer_norm1(x), mask)
        x = x + self.dropout1(attention_output)

        feed_forward_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout2(feed_forward_output)
        return x


class MiniGPT(nn.Module):
    """A small decoder-only Transformer language model."""

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_heads,
        vocab_size,
        max_seq_len,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        _, seq_len = x.shape
        causal_mask = torch.tril(
            torch.ones(
                (1, 1, seq_len, seq_len),
                dtype=torch.bool,
                device=x.device,
            )
        )

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.layer_norm(x)
        return self.lm_head(x)
