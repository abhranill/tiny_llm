"""Compare baseline and dropout-regularized TinyGPT generation."""

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


BASELINE_PATH = "checkpoints/tiny_gpt.pt"
CANDIDATE_PATH = "experiments/regularization/dropout_02.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPTS = [
    "hello world",
    "The",
]

MAX_NEW_TOKENS = 40
TEMPERATURE = 1.2
TOP_K = 20
SEED = 42


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def generate_text(model, tokenizer, prompt):
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError(f"Prompt produced no tokens: {prompt!r}")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # Reset the random seed so both models start from the same RNG state.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

    return tokenizer.decode(output_ids[0].tolist())


def main():
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    baseline, baseline_checkpoint = load_model(BASELINE_PATH)
    candidate, candidate_checkpoint = load_model(CANDIDATE_PATH)

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary size:", len(tokenizer.vocab))
    print(
        "Baseline checkpoint epoch:",
        baseline_checkpoint.get("training_epoch", "unknown"),
    )
    print(
        "Dropout candidate epoch:",
        candidate_checkpoint.get("training_epoch", "unknown"),
    )
    print("Generation settings:", {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "seed": SEED,
    })

    for prompt in PROMPTS:
        print("\n" + "=" * 65)
        print("PROMPT:", repr(prompt))
        print("=" * 65)

        print("\n--- BASELINE (dropout=0.0) ---")
        print(generate_text(baseline, tokenizer, prompt))

        print("\n--- CANDIDATE (dropout=0.2) ---")
        print(generate_text(candidate, tokenizer, prompt))

    print("\nGeneration comparison complete.")
    print("No checkpoint or tokenizer files were modified.")


if __name__ == "__main__":
    main()