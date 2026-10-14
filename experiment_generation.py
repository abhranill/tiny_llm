
"""Compare TinyGPT generation settings and save experiment results."""

from pathlib import Path

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_PATH = "checkpoints/tiny_gpt.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

PROMPT = "A language model"
MAX_NEW_TOKENS = 40
SEED = 42

OUTPUT_PATH = Path("experiments/generation_results.txt")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    # Load the trained checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Load the tokenizer used during training.
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    # Rebuild the model using the saved configuration.
    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", DEVICE)
    print(
        "Checkpoint epoch:",
        checkpoint.get("training_epoch", "unknown"),
    )
    print("Vocabulary size:", len(tokenizer.vocab))
    print("Prompt:", PROMPT)

    # Generation configurations to compare.
    experiments = [
        {
            "name": "Low temperature",
            "temperature": 0.5,
            "top_k": 5,
        },
        {
            "name": "Baseline",
            "temperature": 0.8,
            "top_k": 10,
        },
        {
            "name": "High variety",
            "temperature": 1.2,
            "top_k": 20,
        },
    ]

    # Encode the prompt once.
    prompt_ids = tokenizer.encode(PROMPT)

    if not prompt_ids:
        raise ValueError("The prompt produced no token IDs.")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    results = []

    for experiment in experiments:
        # Reset the random seed for repeatable experiments.
        torch.manual_seed(SEED)

        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(SEED)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids.clone(),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=experiment["temperature"],
                top_k=experiment["top_k"],
            )

        generated_text = tokenizer.decode(
            output_ids[0].tolist()
        )

        result = {
            "name": experiment["name"],
            "temperature": experiment["temperature"],
            "top_k": experiment["top_k"],
            "prompt": PROMPT,
            "generated_text": generated_text,
        }

        results.append(result)

        print(f"\n--- {result['name']} ---")
        print("Temperature:", result["temperature"])
        print("Top-k:", result["top_k"])
        print(result["generated_text"])

    # Save all results to a text report.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write("TinyGPT Generation Experiments\n")
        file.write("=" * 60 + "\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Checkpoint epoch: {checkpoint.get('training_epoch', 'unknown')}\n")
        file.write(f"Prompt: {PROMPT}\n")
        file.write(f"Max new tokens: {MAX_NEW_TOKENS}\n")
        file.write(f"Random seed: {SEED}\n\n")

        for result in results:
            file.write(f"Experiment: {result['name']}\n")
            file.write(
                f"Temperature: {result['temperature']}\n"
            )
            file.write(f"Top-k: {result['top_k']}\n")
            file.write(f"Prompt: {result['prompt']}\n")
            file.write(
                f"Generated text: {result['generated_text']}\n"
            )
            file.write("-" * 60 + "\n")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
