
"""Causal self-attention implemented with PyTorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head causal self-attention."""

    def __init__(self, embed_dim, context_length):
        super().__init__()

        self.embed_dim = embed_dim
        self.context_length = context_length

        # Learnable projections for queries, keys, and values.
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Learnable projection applied after attention.
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Lower-triangular mask: a token can see itself
        # and earlier positions, but not future positions.
        mask = torch.tril(
            torch.ones(context_length, context_length)
        )

        # A buffer moves with the model but is not trainable.
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x has shape: (batch_size, sequence_length, embed_dim)
        batch_size, sequence_length, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError("Unexpected embedding dimension.")

        if sequence_length > self.context_length:
            raise ValueError("Sequence exceeds context length.")

        # Project each token representation into Q, K, and V.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Similarity scores between each query and every key.
        scores = Q @ K.transpose(-2, -1)

        # Scale the scores to stabilize their magnitude.
        scores = scores / math.sqrt(self.embed_dim)

        # Block attention to future tokens.
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Convert allowed scores into attention probabilities.
        weights = F.softmax(scores, dim=-1)

        # Combine value vectors using the attention weights.
        context = weights @ V

        # Transform the result with a final learned projection.
        output = self.output_projection(context)

        return output

class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, embed_dim, num_heads, context_length, dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Calculate queries, keys and values together.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Combine the outputs from all attention heads.
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Each position can attend only to itself and earlier positions.
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        if sequence_length > self.causal_mask.shape[0]:
            raise ValueError("Sequence exceeds context length.")

        # x: (batch, sequence, embedding)
        qkv = self.qkv(x)

        # Split into queries, keys and values.
        qkv = qkv.reshape(
            batch_size, sequence_length, 3,
            self.num_heads, self.head_dim
        )

        # Put the Q/K/V axis first, then split it.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)

        # Calculate attention scores.
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Block access to future tokens.
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # Gather information from earlier tokens.
        context = weights @ values

        # Merge all heads back into one embedding.
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, sequence_length, self.embed_dim)

        output = self.output_projection(context)
        return self.output_dropout(output)
