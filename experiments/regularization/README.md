# TinyGPT Regularization Experiment

## Objective

Compare dropout rates of 0.0, 0.1, and 0.2 to investigate whether dropout improves validation performance.

## Experimental setup

- Model: TinyGPT, with the same architecture across experiments
- Random seed: 42
- Optimizer: AdamW
- Weight decay: 0.01
- Learning rate: 0.0003
- Sequence length: 16
- Batch size: 8
- Maximum epochs: 15
- Early-stopping patience: 5
- Dataset split: fixed paragraph-based training and validation split
- Vocabulary size: 71

Each experiment uses the same data split and training settings, with dropout as the intended model difference. The experiments are reproducible runs, but differences in dropout randomness and data-loader randomness mean this is not a perfect isolation of dropout's effect.

## Results

| Experiment | Dropout | Best epoch | Best validation loss |
|---|---:|---:|---:|
| Baseline | 0.0 | 4 | 2.9576 |
| Dropout 0.1 | 0.1 | 8 | 2.8839 |
| Dropout 0.2 | 0.2 | 11 | 2.8757 |

Validation perplexity is calculated as the exponential of the best validation loss.

- Baseline perplexity: approximately 19.252
- Dropout 0.1 perplexity: approximately 17.883
- Dropout 0.2 perplexity: approximately 17.738

Dropout 0.2 achieved approximately 2.77% lower validation loss than the baseline in this experiment.

## Generation comparison

The baseline and dropout 0.2 candidate were compared using the same prompts:

- `hello world`
- `The`

Both used a maximum of 40 new tokens, temperature 1.2, top-k 20, and random seed 42.

Both models generated largely incoherent text. The samples do not provide convincing qualitative evidence that the candidate generates better text.

## Conclusion

Dropout 0.2 is the provisional candidate because it achieved the lowest validation loss among the three tested configurations.

This result is preliminary: the validation set contains only three paragraphs, and a single run per configuration is not sufficient to establish generalization improvement. Further experiments with multiple random seeds and a larger validation corpus would be needed for a stronger conclusion.

The experiment checkpoints and CSV reports are stored in this directory. The main checkpoint and tokenizer are kept separate and are not replaced by these experiments.