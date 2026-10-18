# TinyGPT Dataset Size Experiment

## Objective

Investigate whether adding relevant training paragraphs improves TinyGPT's validation performance.

## Dataset preparation

- Original corpus: 14 paragraphs, 595 words.
- Additional corpus: 12 paragraphs, 441 words.
- Combined corpus: 26 paragraphs, 1,036 words.
- Duplicate paragraphs in the combined corpus: 0.
- Paragraphs shorter than 30 words: 0.

The original corpus was preserved. The additional and combined corpora were stored separately.

## Experimental setup

- Model: TinyGPT
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Context length: 128
- Sequence length: 16
- Batch size: 8
- Optimizer: AdamW
- Learning rate: 0.0003
- Dropout: 0.2
- Random seed: 42
- Maximum epochs: 30
- Early-stopping patience: 5
- Tokenizer: shared BPE tokenizer trained on the original training paragraphs
- Validation: the same three original held-out paragraphs in both runs

The original experiment used 11 training paragraphs. The expanded experiment used those same 11 paragraphs plus 12 additional paragraphs. Both runs used the same 71-token vocabulary and the same validation text.

## Results

| Metric | Original | Expanded |
|---|---:|---:|
| Training paragraphs | 11 | 23 |
| Training tokens | 2,279 | 4,522 |
| Validation paragraphs | 3 | 3 |
| Validation tokens | 663 | 663 |
| Vocabulary size | 71 | 71 |
| Best epoch | 13 | 14 |
| Best validation loss | 2.882821 | 2.471727 |

The expanded experiment achieved approximately 14.3% lower validation loss.

## Validation checks

- No normalized paragraph overlap between training and validation sets.
- Eight unknown tokens were found in 2,241 encoded tokens from the additional corpus, an unknown-token rate of approximately 0.36%.
- The main checkpoint and tokenizer were not modified.
- Existing evaluation, generation-validation, and tokenizer-robustness tests passed.

## Limitations

This is preliminary evidence from one run per condition and a validation set containing only three paragraphs. The result does not establish that the expanded dataset will improve performance on general text. Multiple random seeds, a larger held-out dataset, and qualitative generation checks would strengthen the conclusion.

The validation loss is a diagnostic metric for this controlled experiment, not a definitive measure of general language-model quality.

## Reproducibility

Run the experiment with:

`python experiment_dataset_size.py`

The experiment saves its checkpoints, tokenizers, histories, and summary under `experiments/dataset_comparison/`. Experimental model weights should not be committed to Git.