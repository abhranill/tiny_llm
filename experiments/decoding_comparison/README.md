
# TinyGPT Decoding Strategy Comparison

## Objective

Compare greedy decoding, unrestricted sampling, top-k sampling, and lower-temperature top-k sampling using the same TinyGPT checkpoint and prompts.

## Experimental setup

- Checkpoint: `checkpoints/tiny_gpt.pt`
- Tokenizer: `tokenizer/tokenizer.json`
- Device: CUDA when available
- Prompts: `hello world`, `machine learning`, `The`
- Maximum new tokens: 40
- Random seed: 42
- Strategies:
  - Greedy decoding
  - Sampling with temperature 1.0 and no top-k restriction
  - Sampling with temperature 1.0 and top-k 10
  - Sampling with temperature 0.7 and top-k 10

The experiment did not retrain the model or modify the main checkpoint or tokenizer.

## Output files

- `results.csv`: generated text and experiment settings
- `results.txt`: readable generation results
- `metrics.csv`: whitespace-based diversity and repetition diagnostics
- `quality_metrics.csv`: BPE-token and character-level diagnostics

## Preliminary observations

Greedy decoding showed more repeated BPE n-grams for some prompts. Unrestricted sampling showed higher unique-token ratios in some cases, while top-k strategies produced mixed results. No strategy consistently produced coherent text.

The results illustrate that decoding settings influence output diversity and repetition, but do not compensate for limited training data or an undertrained language model.

## Limitations

- Only three prompts and one random seed were used.
- The experiment is exploratory, not a statistically reliable benchmark.
- Unique-token ratios and repeated n-gram rates do not measure grammar, factuality, or semantic coherence.
- Character-level repetition metrics can miss other forms of repetition.
- Results depend on the specific checkpoint and tokenizer.

## Reproducibility

Run the experiment from the project root:

`python experiment_decoding.py`

Then calculate the metrics:

`python evaluate_decoding_results.py`

`python evaluate_decoding_quality.py`

The experiment saves its outputs under `experiments/decoding_comparison/`. The main checkpoint and tokenizer should remain unchanged.
