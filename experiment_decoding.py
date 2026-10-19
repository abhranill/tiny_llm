
"""Compare decoding strategies using the same TinyGPT checkpoint."""

import csv
import random
from pathlib import Path

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_PATH = Path("checkpoints/tiny_gpt.pt")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
OUTPUT_DIR = Path("experiments/decoding_comparison")

PROMPTS = [
    "hello world",
    "machine learning",
    "The",
]

MAX_NEW_TOKENS = 40
SEED = 42

STRATEGIES = [
    {"name": "greedy", "mode": "greedy", "temperature": 1.0, "top_k": None},
    {"name": "sampling", "mode": "sample", "temperature": 1.0, "top_k": None},
    {"name": "top_k_10", "mode": "sample", "temperature": 1.0, "top_k": 10},
    {"name": "cool_top_k_10", "mode": "sample", "temperature": 0.7, "top_k": 10},
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if len(tokenizer.vocab) != checkpoint["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary size does not match the checkpoint."
        )

    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_text(model, tokenizer, prompt, strategy, device):
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError(f"Prompt produced no tokens: {prompt!r}")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    set_seed(SEED)

    for _ in range(MAX_NEW_TOKENS):
        context = input_ids[:, -model.context_length:]
        logits = model(context)[:, -1, :]

        if strategy["mode"] == "greedy":
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / strategy["temperature"]

            top_k = strategy["top_k"]
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k, dim=-1)
                threshold = top_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf"),
                )

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer, checkpoint = load_model(device)

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("training_epoch", "unknown"))
    print("Vocabulary size:", len(tokenizer.vocab))
    print("Seed:", SEED)
    print("New tokens per generation:", MAX_NEW_TOKENS)

    results = []

    for prompt in PROMPTS:
        print("\n" + "=" * 70)
        print("PROMPT:", repr(prompt))
        print("=" * 70)

        for strategy in STRATEGIES:
            generated = generate_text(
                model, tokenizer, prompt, strategy, device
            )

            row = {
                "prompt": prompt,
                "strategy": strategy["name"],
                "temperature": strategy["temperature"],
                "top_k": strategy["top_k"],
                "seed": SEED,
                "generated_text": generated,
            }
            results.append(row)

            print(f"\n--- {strategy['name']} ---")
            print(generated)

    csv_path = OUTPUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    text_path = OUTPUT_DIR / "results.txt"
    with text_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(f"PROMPT: {row['prompt']}\n")
            file.write(f"STRATEGY: {row['strategy']}\n")
            file.write(f"TEXT: {row['generated_text']}\n")
            file.write("-" * 70 + "\n")

    print("\nExperiment complete.")
    print("CSV results:", csv_path)
    print("Text results:", text_path)
    print("Main checkpoint and tokenizer were not modified.")


if __name__ == "__main__":
    main()
