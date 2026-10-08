                Raw Text
                   │
                   ▼
             ┌───────────┐
             │ Tokenizer │
             └─────┬─────┘
                   │
                   ▼
              Token IDs
                   │
                   ▼
            Token Embeddings
                   │
                   ▼
        ┌─────────────────────┐
        │   Transformer       │
        │                     │
        │  Self Attention     │
        │        ↓            │
        │  Feed Forward       │
        │        ↓            │
        │  Layer Normalization│
        │        ↓            │
        │  Residual Connection│
        └──────────┬──────────┘
                   │
                   ▼
                 Logits
                   │
                   ▼
          Next Token Prediction
                   │
                   ▼
              Generated Text
