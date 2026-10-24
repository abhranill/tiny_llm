🧠 Tiny LLM — Building a Small Language Model from Scratch

A hands-on implementation of a tiny GPT-style language model using Python and PyTorch.

Tiny LLM is an educational project focused on understanding how Large Language Models (LLMs) work internally. Instead of using a pretrained model, this project implements the fundamental components of a GPT-style language model, including tokenization, token embeddings, positional embeddings, self-attention, Transformer blocks, training, validation, text generation, and model checkpointing.

The project is designed to demonstrate the fundamental principles behind modern language models while exploring the complete machine learning workflow, from preparing text data to training and evaluating a neural network.

«Project goal: Understand how GPT-style language models work by building, training, testing, and validating a small Transformer-based language model from the ground up.»

---

📌 Table of Contents

- "Overview" (#-overview)
- "Project Objectives" (#-project-objectives)
- "Key Features" (#-key-features)
- "How Does an LLM Work?" (#-how-does-an-llm-work)
- "Model Architecture" (#-model-architecture)
- "Project Structure" (#-project-structure)
- "Technology Stack" (#-technology-stack)
- "Requirements" (#-requirements)
- "Installation" (#-installation)
- "Dataset Preparation" (#-dataset-preparation)
- "Training the Model" (#-training-the-model)
- "How Training Works" (#-how-training-works)
- "Generating Text" (#-generating-text)
- "Checkpoint Validation" (#-checkpoint-validation)
- "Testing" (#-testing)
- "GPU Acceleration" (#-gpu-acceleration)
- "Understanding the Results" (#-understanding-the-results)
- "Current Limitations" (#-current-limitations)
- "Future Improvements" (#-future-improvements)
- "Learning Outcomes" (#-learning-outcomes)
- "Contributing" (#-contributing)
- "License" (#-license)

---

🚀 Overview

Large Language Models such as ChatGPT, GPT, and other generative AI systems use neural networks to learn patterns in language and generate text.

Although modern LLMs can contain billions of parameters, their fundamental architecture can be explored using much smaller models.

This project implements a miniature GPT-style language model called TinyGPT.

The model uses the Transformer architecture to learn patterns from a small text corpus. During training, it learns to predict the next token in a sequence based on the tokens that came before it.

For example, given the input:

hello world

The model learns to predict what token is likely to appear next based on patterns in its training data.

After training, the model can generate text autoregressively, meaning that each newly generated token becomes part of the input used to predict the following token.

Why build a Tiny LLM?

Rather than treating an LLM as a black box, this project explores the engineering concepts behind one.

It covers:

- Text tokenization and vocabulary construction.
- Converting text into numerical token IDs.
- Token and positional embeddings.
- Causal self-attention.
- Multi-head attention.
- Feed-forward neural networks.
- Transformer blocks and normalization.
- Next-token prediction.
- Training and validation loss.
- Early stopping and checkpoint management.
- Autoregressive text generation.
- Automated testing and checkpoint integrity checks.

The model is intentionally small so that its components can be studied, tested, and trained on consumer hardware.

---

🎯 Project Objectives

The main objectives of this project are:

1. Understand tokenization: Learn how text is converted into numerical representations that a neural network can process.

2. Implement the Transformer architecture: Build the essential components of a GPT-style model using PyTorch.

3. Understand self-attention: Explore how a model uses previous tokens to determine which information is important for predicting the next token.

4. Build a training pipeline: Implement forward propagation, loss calculation, backpropagation, and parameter optimization.

5. Evaluate model performance: Monitor training loss and validation loss to understand how the model learns and when it begins to overfit.

6. Implement reproducible experiments: Save model checkpoints and validate the files required to reproduce a training run.

7. Use GPU acceleration: Train the model on a CUDA-compatible NVIDIA GPU when the required software is available.

8. Develop practical machine learning skills: Apply software engineering, testing, experimentation, and debugging techniques to a real deep learning project.

---

✨ Key Features

Feature| Description
Custom tokenizer| Converts text into numerical token IDs.
Transformer architecture| Implements a compact GPT-style neural network.
Causal self-attention| Allows each position to attend to permitted previous positions.
Multi-head attention| Learns multiple attention patterns in parallel.
Positional embeddings| Provide information about token positions.
Next-token prediction| Trains the model to predict subsequent tokens.
Training pipeline| Performs optimization over multiple epochs.
Validation monitoring| Evaluates model performance on held-out data.
Early stopping| Can stop training when validation performance stops improving.
Checkpoint saving| Saves model parameters for later use.
Tokenizer persistence| Saves tokenizer information for consistent encoding and decoding.
Text generation| Generates sequences using autoregressive sampling.
Top-k sampling| Restricts token selection to a specified number of high-probability candidates.
Temperature control| Adjusts the randomness of token sampling.
Automated tests| Tests important components of the implementation.
Checkpoint integrity validation| Checks whether required model, tokenizer, and corpus files are present.
CUDA support| Uses a compatible NVIDIA GPU when available.

---

🧩 How Does an LLM Work?

A language model learns statistical patterns from text and uses those patterns to predict the next token.

TinyGPT follows a simplified version of the GPT training and generation process.

Step 1: Input text

The model receives a text corpus containing examples of language.

For example:

hello world.
hello machine learning.
machine learning helps computers learn patterns from text.

This is the source material from which the model learns.

Step 2: Tokenization

The tokenizer converts the text into numerical token IDs.

Conceptually:

Text
  ↓
Tokenizer
  ↓
Token IDs

For example, a character-level tokenizer might convert:

hello

into a sequence of integer IDs.

Each supported character is associated with an entry in the vocabulary.

The exact mapping depends on the tokenizer implementation.

Step 3: Create training sequences

The token sequence is divided into smaller training examples.

For next-token prediction, the input and target sequences are shifted relative to each other.

For example:

Input:  h e l l
Target: e l l o

Each target token represents the next token the model should predict at the corresponding position.

Step 4: Convert tokens into embeddings

Token IDs are converted into dense numerical vectors using a learned embedding layer.

Positional embeddings are added so that the model can distinguish between different token positions.

Token IDs
   +
Token Embeddings
   +
Positional Embeddings
   ↓
Input Representations

Step 5: Process tokens through Transformer blocks

The input representations pass through a series of Transformer blocks.

Each block combines self-attention and feed-forward computation to learn contextual representations of the input sequence.

Step 6: Predict the next token

The final hidden representations are projected into vocabulary-sized logits.

These logits represent the model's unnormalized scores for possible next tokens.

A softmax operation converts the scores into probabilities when required for prediction or sampling.

Step 7: Calculate the loss

During training, the model's predictions are compared with the actual next tokens.

Cross-entropy loss measures the difference between the predicted token distribution and the correct targets.

The optimizer uses backpropagation to update the model parameters and improve its predictions.

Step 8: Generate text

During inference, the model predicts a distribution over the next token.

A token is selected using the configured generation strategy and appended to the input sequence.

The process repeats until the generation limit or another stopping condition is reached.

Prompt
  ↓
Tokenizer
  ↓
TinyGPT
  ↓
Next-token probabilities
  ↓
Token selection
  ↓
Append generated token
  ↓
Repeat
  ↓
Generated text

This process is called autoregressive generation.

---

🏗️ Model Architecture

TinyGPT is based on a decoder-only Transformer architecture inspired by GPT-style language models.

The model implementation includes token embeddings, positional embeddings, Transformer blocks, normalization, and an output projection layer.

Architecture overview

                Input Text
                    |
                    v
                 Tokenizer
                    |
                    v
                Token IDs
                    |
                    v
              Token Embedding
                    |
                    + <---- Positional Embedding
                    |
                    v
              Embedding Dropout
                    |
                    v
          +------------------------+
          |    Transformer Block   |
          |                        |
          |   Multi-Head Attention |
          |            |           |
          |      Residual Path     |
          |            |           |
          |   Feed-Forward Network |
          |            |           |
          |      Residual Path     |
          +------------------------+
                    |
                    v
              Repeat Blocks
                    |
                    v
               Final Norm
                    |
                    v
               Language Head
                    |
                    v
             Vocabulary Logits
                    |
                    v
              Next-Token Prediction

This diagram illustrates the conceptual data flow. Exact normalization placement and residual connections depend on the implementation.

1. Token Embedding

The token embedding layer converts each token ID into a learned vector representation.

In the current configuration, the embedding dimension is 64.

2. Positional Embedding

Self-attention does not inherently understand the order of tokens.

Positional embeddings provide information about each token's position in the sequence.

TinyGPT uses learned positional embeddings.

3. Multi-Head Self-Attention

Self-attention allows the model to determine which earlier tokens are relevant to the current prediction.

Multi-head attention performs multiple attention computations in parallel, allowing the network to learn different relationships between tokens.

The attention operation is commonly expressed as:

[
\operatorname{Attention}(Q,K,V)

\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}+M
\right)V
]

Where:

- Q represents queries.
- K represents keys.
- V represents values.
- d_k is the key dimension.
- M is the causal attention mask.

The causal mask prevents a token from accessing future tokens during next-token prediction.

4. Feed-Forward Network

The feed-forward network transforms the representation of each token independently after the attention operation.

It introduces additional nonlinear computation that helps the model learn complex patterns.

5. Transformer Blocks

TinyGPT stacks two Transformer blocks.

Each block combines attention and feed-forward processing with residual connections and normalization according to the model implementation.

6. Final Normalization

A final normalization layer processes the hidden representations before the output projection.

7. Language Modeling Head

The output projection maps the hidden representation of each token to a vector whose size equals the vocabulary size.

The resulting logits are used to predict the next token.

Current model configuration

The current configuration recorded during development is:

Parameter| Value
Embedding dimension| 64
Number of attention heads| 4
Number of Transformer blocks| 2
Maximum context length| 128 tokens
Dropout| 0.0
Trainable parameters| Approximately 114,738
Vocabulary size| Determined by the tokenizer

The vocabulary size depends on the tokenizer and corpus configuration.

These values describe the current small experimental model, not a fixed requirement for every future version of the project.

---

📁 Project Structure

The project has been developed as a modular Python application.

The following structure represents the main components and scripts used during development. Some filenames may differ depending on the current repository version.

tiny_llm/
│
├── data/
│   └── raw/
│       └── tiny_corpus.txt
│
├── tokenizer/
│   └── tokenizer.json
│
├── model/
│   └── transformer.py
│
├── utils/
│   └── ...
│
├── checkpoints/
│   └── tiny_gpt.pt
│
├── experiments/
│   └── reproducible_training_repeat/
│       └── best_model.pt
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_attention.py
│   ├── test_attention_gpu.py
│   └── test_tiny_gpt.py
│
├── train.py
├── generate.py
├── validate_checkpoint.py
├── requirements.txt
├── .gitignore
└── README.md

Note: The directory tree is illustrative. Keep only the filenames and directories that actually exist in your repository.

Important files and directories

"data/raw/tiny_corpus.txt"

Contains the text used for training and experimentation.

"tokenizer/"

Contains tokenizer-related code or saved tokenizer artifacts, depending on the current implementation.

The saved tokenizer allows text to be encoded and decoded consistently across training and inference.

"model/transformer.py"

Contains the TinyGPT architecture and its Transformer components.

The implementation includes token embeddings, positional embeddings, attention, feed-forward processing, and the language modeling output layer.

"utils/"

Contains reusable utilities used by the training or inference pipeline.

"train.py"

Runs the training process, including data preparation, model optimization, validation, and checkpoint saving.

The current training pipeline includes validation monitoring and early-stopping logic.

"generate.py"

Provides the entry point for generating text with a trained model.

"validate_checkpoint.py"

Checks checkpoint integrity and the presence of the associated files required by the validation procedure.

"tests/"

Contains tests for individual components, including the tokenizer, dataset, attention implementation, and complete model.

"checkpoints/"

Stores saved model checkpoints.

"experiments/"

Organizes experiment-specific artifacts, such as the best model from a training run.

---

🛠️ Technology Stack

Technology| Purpose
Python| Main programming language
PyTorch| Neural network implementation and training
CUDA| GPU acceleration
NVIDIA GPU| Hardware acceleration for compatible operations
JSON| Tokenizer artifact storage
Git| Version control
PowerShell| Development environment on Windows

Development hardware

The project has been tested in the following development environment:

- Operating system: Windows
- Python: 3.14.2
- Deep learning framework: PyTorch 2.14.1+cu130, as recorded during development
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU memory: Approximately 6 GB
- System RAM: 16 GB

Your installed package versions may differ. The version numbers above describe the development environment and are not universal installation requirements.

---

📋 Requirements

Before starting, ensure that you have:

- Python installed.
- Git installed if you want to clone the repository.
- A terminal such as PowerShell, Command Prompt, or a Linux shell.
- PyTorch installed with a build compatible with your Python version and hardware.
- A CUDA-compatible NVIDIA GPU if you want GPU acceleration.

A GPU is optional for a model this small. CPU execution is also possible, although training speed may differ.

---

⚙️ Installation

Step 1: Clone the repository

Replace "<your-repository-url>" with the actual Git repository URL.

git clone <your-repository-url>
cd tiny_llm

Step 2: Create a virtual environment

On Windows PowerShell:

python -m venv .venv

Activate the environment:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks script execution, use an appropriate environment-specific solution or activate the environment through another supported terminal.

Step 3: Install dependencies

If the repository includes a "requirements.txt" file:

python -m pip install -r requirements.txt

Otherwise, install the dependencies required by the current source code.

For PyTorch, use the official installation selector to choose a build compatible with your operating system, Python version, and accelerator:

"PyTorch Installation Guide" (https://reference-url-citation.invalid/0)

Avoid blindly installing a different CUDA build over a working environment. Python, PyTorch, and CUDA compatibility must be considered together.

Step 4: Verify the installation

Run:

python --version

Check the PyTorch installation:

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

Save the snippet as a Python file or execute it in the Python interpreter.

A successful CUDA check indicates that PyTorch can access a compatible CUDA device.

If CUDA is unavailable, investigate the installed PyTorch build, NVIDIA driver, and environment configuration.

---

📚 Dataset Preparation

TinyGPT learns from the text provided in its training corpus.

The development corpus is stored at:

data/raw/tiny_corpus.txt

An example of the kind of small corpus used during early experiments is:

hello world.
hello world.
hello machine learning.
machine learning helps computers learn patterns from text.

This example is illustrative; use the actual corpus from your repository for reproducible results.

Why does the dataset matter?

The dataset determines which language patterns the model can learn.

A small dataset can be useful for debugging because it allows the training pipeline to run quickly.

However, a model trained on a tiny corpus cannot be expected to acquire broad language understanding or produce consistently meaningful text.

Training and validation data

The training pipeline separates tokenized data into training and validation portions.

Training data is used to update the model parameters.

Validation data is used to evaluate performance on data excluded from the optimization process.

The validation set helps identify overfitting and assess whether the model's predictions generalize beyond the training examples.

For meaningful evaluation, the split must be constructed carefully, especially when the corpus contains repeated or highly overlapping text.

---

🏋️ Training the Model

Once the environment and dataset are prepared, run the training script:

python train.py

The exact command-line options depend on the current implementation of "train.py".

During training, the script prepares the data, initializes the model, performs optimization, monitors validation performance, and saves checkpoint artifacts.

What happens during training?

The general training process is:

1. Load the text corpus.
2. Build or load the tokenizer.
3. Encode the corpus into token IDs.
4. Split the token sequence into training and validation data.
5. Initialize the TinyGPT model.
6. Select the available computation device.
7. Generate batches of input and target sequences.
8. Perform a forward pass.
9. Calculate next-token prediction loss.
10. Calculate gradients through backpropagation.
11. Update model parameters using the optimi
