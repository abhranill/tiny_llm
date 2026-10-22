
# TinyGPT Checkpoint Audit — Day 24

## Objective

Inspect the existing TinyGPT checkpoint for model compatibility and training reproducibility without retraining the model or modifying project files.

## Audited artifacts

- Checkpoint: `checkpoints/tiny_gpt.pt`
- Tokenizer: `tokenizer/tokenizer.json`
- Current corpus: `data/raw/tiny_corpus.txt`

## Existing checkpoint metadata

| Field | Value |
|---|---|
| Training epoch | 4 |
| Validation loss | 2.9576310052 |
| Vocabulary size | 71 |
| Sequence length | 16 |
| BPE merge count | 25 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Context length | 128 |
| Dropout | 0.0 |

## Audit results

- Checkpoint loaded successfully using `weights_only=True`.
- Model reconstructed successfully from the saved configuration.
- Checkpoint and tokenizer vocabulary sizes match.
- Recommended metadata is missing: random seed, corpus SHA-256, training configuration, optimizer configuration, and explicit best epoch.
- Matching vocabulary sizes do not prove that the tokenizer is identical to the tokenizer used during training.
- The current corpus hash was calculated, but the historical training corpus cannot be confirmed from the checkpoint alone.

## Risks identified

1. The training script saves the tokenizer before model training completes. An interrupted run could leave a tokenizer that no longer matches the existing checkpoint.
2. The checkpoint does not record all settings needed to reproduce the training run.
3. Python and NumPy random seeds are not explicitly set in the training script.
4. The checkpoint uses `training_epoch` rather than a separate `best_epoch` field.

## Safety

The audit was read-only. It did not retrain the model or modify the checkpoint, tokenizer, or corpus.

## Recommended future improvements

- Save a complete training configuration and corpus hash in each checkpoint.
- Record random seeds and optimizer settings.
- Save the tokenizer and checkpoint as a matched experiment artifact.
- Keep experimental checkpoints separate from the main baseline.
- Add a test that verifies checkpoint-tokenizer compatibility beyond vocabulary size.

## Reproduce the audit

Run from the repository root:

`python audit_checkpoint.py`
