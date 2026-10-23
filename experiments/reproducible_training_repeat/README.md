# Day 26: Repeated-Run Reproducibility Check

## Objective

Determine whether two training runs using the same seed, corpus, model
configuration, and deterministic settings produce identical artifacts.

## Method

- Run A: `experiments/reproducible_training/`
- Run B: `experiments/reproducible_training_repeat/`
- Seed: 42
- Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- PyTorch: 2.14.1+cu130
- Corpus SHA-256:
  `7a849d5ddfcc5ad48b11f36ca408cd4fa4d7acb15e3c15d1e783213dec12e430`
- Deterministic algorithms enabled.
- Both runs used the same training script and hyperparameters, with
  separate output directories.

## Results

| Check | Result |
|---|---|
| Tokenizer files byte-identical | PASS |
| Training history files byte-identical | PASS |
| Metadata identical | PASS |
| Checkpoint metadata identical | PASS |
| Model state keys identical | PASS |
| All model tensors exactly equal | PASS |
| Original baseline checkpoint hash unchanged | PASS |
| Original baseline tokenizer hash unchanged | PASS |

Both runs selected epoch 4 as the best epoch, with validation loss
approximately 2.951692. Both stopped after epoch 9.

## Conclusion

The two runs were exactly reproducible for the checked artifacts in the
current environment. The original baseline checkpoint and tokenizer
matched their previously recorded SHA-256 hashes.

## Limitations

- Only two runs were compared.
- Both runs used the same machine, GPU, operating system, and software stack.
- This does not establish reproducibility across different hardware,
  PyTorch/CUDA versions, or operating systems.
- Identical results in these runs do not prove every future run will match
  under all environmental conditions.
