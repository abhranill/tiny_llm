
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
        # Attention followed by a residual connection.
        x = x + self.attention(self.ln1(x))

        # Feed-forward network followed by a residual connection.
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

        # Learn a vector for each token position.
        self.position_embedding = nn.Embedding(
            context_length, embed_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Stack Transformer blocks.
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

        # Produce one score for each vocabulary token.
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """Convert token IDs into next-token prediction logits."""

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Input exceeds the model's context length."
            )

        # Positions: (sequence_length,)
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        # Both embeddings have shape (batch, sequence, embed_dim).
        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.embedding_dropout(x)

        x = self.blocks(x)
        x = self.final_norm(x)

        # Logits: (batch, sequence, vocab_size)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=40,
        temperature=1.0,
        top_k=None,
    ):
        """Generate new token IDs autoregressively."""

        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
    raise ValueError("max_new_tokens must be a non-negative integer.")

if max_new_tokens < 0:
    raise ValueError("max_new_tokens must be a non-negative integer.")

if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
    raise ValueError("temperature must be a positive number.")

if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
    raise ValueError("temperature must be finite and greater than zero.")

if top_k is not None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer or None.")

if input_ids.ndim != 2 or input_ids.size(0) < 1 or input_ids.size(1) < 1:
    raise ValueError("input_ids must have shape (batch_size, sequence_length).")

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # Keep only the tokens that fit the context window.
                input_context = input_ids[:, -self.context_length:]

                # Get scores and select the final position.
                logits = self(input_context)
                logits = logits[:, -1, :] / temperature

                # Optionally restrict sampling to the top-k tokens.
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    top_values, _ = torch.topk(logits, k, dim=-1)
                    threshold = top_values[:, -1].unsqueeze(-1)

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf"),
                    )

                probabilities = torch.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

                input_ids = torch.cat(
                    (input_ids, next_token),
                    dim=1,
                )
        finally:
            self.train(was_training)

        return input_ids
