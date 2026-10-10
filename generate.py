
"""Generate text using a trained TinyGPT checkpoint."""

import torch

from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_PATH = "checkpoints/tiny_gpt.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    # Load the saved checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Load the exact tokenizer used during training.
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    # Rebuild the model with the saved configuration.
    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", DEVICE)
    print("Checkpoint epoch:", checkpoint.get("training_epoch", "unknown"))
    print("Vocabulary size:", len(tokenizer.vocab))

    prompt = input("Enter a prompt (try 'hello world'): ").strip()

    if not prompt:
        prompt = "hello world"

    # Convert the prompt into token IDs.
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # Generate new tokens.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=40,
            temperature=0.8,
            top_k=10,
        )

    # Convert all generated IDs back into text.
    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print("\n--- Generated text ---")
    print(generated_text)


if __name__ == "__main__":
    main()
