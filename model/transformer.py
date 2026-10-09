
"""Building blocks for our Tiny GPT model."""

import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """A small neural network applied independently at each token."""

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """One pre-LayerNorm Transformer block."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        dropout=0.0,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, dropout)

    def forward(self, x):
        # First sublayer: attention plus residual connection.
        x = x + self.attention(self.ln1(x))

        # Second sublayer: feed-forward network plus residual.
        x = x + self.feed_forward(self.ln2(x))

        return x

class TinyGPT(nn.Module):
    """A small decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        context_length=128,
        dropout=0.0,
    ):
        super().__init__()

        self.context_length = context_length

        # Convert token IDs into learned vectors.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Give each position a learned positional vector.
        self.position_embedding = nn.Embedding(
            context_length, embed_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Stack multiple Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        # Convert each token representation into vocabulary scores.
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Input exceeds the model's context length.")

        # Shape: (sequence_length,)
        positions = torch.arange(
            sequence_length, device=input_ids.device
        )

        # Both embeddings have shape (batch, sequence, embed_dim).
        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.embedding_dropout(x)

        x = self.blocks(x)
        x = self.final_norm(x)

        # Shape: (batch, sequence, vocab_size)
        logits = self.lm_head(x)

        return logits
