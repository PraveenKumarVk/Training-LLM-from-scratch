# Training LLMs from Scratch

This repository provides tools, scripts, and configurations for training Large Language Models (LLMs) from scratch. It encompasses the complete workflow, from dataset preparation and tokenizer creation to model training, fine-tuning, and inference.

## Features

- **Dataset Handling**:
  - Scripts for cloning, preparing, and filtering datasets.
- **Tokenizer Creation**:
  - Tools for creating custom tokenizers for LLM training.
- **Training Pipelines**:
  - Support for frameworks like DeepSpeed and Fully Sharded Data Parallel (FSDP).
- **Fine-Tuning and Optimization**:
  - Modules for Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO).
- **Inference Tools**:
  - Scripts and notebooks for evaluating and deploying trained models.

## Project Structure

```plaintext
training-llms-from-scratch/
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
├── Core-Training-and-Dataset-Preparation/               # Core training and dataset preparation
│   ├── training/          # Training scripts and configurations
│   ├── tokenizer_creation/ # Utilities for tokenizer creation
│   └── dataset_creation/  # Dataset handling and preprocessing
├── Fine-Tuning-and-Optimization/               # Fine-tuning and inference
│   ├── dpo/               # Direct Preference Optimization
│   ├── sft/               # Supervised Fine-Tuning
│   └── ppo/               # Proximal Policy Optimization
```

## Installation

### Clone the repository:

    `git clone https://github.com/your-username/training-llms-from-scratch.git
    cd training-llms-from-scratch`

### Install the dependencies:

    `pip install -r requirements.txt`

**Ensure you have the necessary tools like `DeepSpeed`, `transformers`, and `PyTorch` installed.**

## Usage

### 1. Dataset Preparation

Prepare datasets using the scripts in `Core-Training-and-Dataset-Preparation/dataset_creation`:

`python Core-Training-and-Dataset-Preparation/dataset_creation/prepare_hf_dataset.py --config dataset_config.yaml`

### 2. Tokenizer Creation

Generate a custom tokenizer:

`python Core-Training-and-Dataset-Preparation/tokenizer_creation/create_tokenizer.py --vocab_size 50000`

### 3. Model Training

Run training pipelines using DeepSpeed or FSDP:

`bash Core-Training-and-Dataset-Preparation/training/run_deepspeed.sh`

### 4. Fine-Tuning

Fine-tune the model using SFT, PPO, or DPO strategies:

`# Example: Running Supervised Fine-Tuning
bash Fine-Tuning-and-Optimization/sft/run_sft.sh`

### 5. Inference

Evaluate or deploy the model using the provided notebooks and scripts:

`# Example: DPO-based chatbot inference
jupyter notebook Fine-Tuning-and-Optimization/dpo/dpo_inference_chatbot.ipynb`

## Configuration

Update the YAML configuration files in `Core-Training-and-Dataset-Preparation/training/configs` for your training setup:

- `deepspeed_config.yaml`: Configuration for DeepSpeed.
- `fsdp_config.yaml`: Configuration for Fully Sharded Data Parallel.

Modify parameters such as batch size, learning rate, or model architecture as required.

## Acknowledgements

- [DeepSpeed](https://www.deepspeed.ai/)
- Hugging Face Transformers
- [PyTorch](https://pytorch.org/)
