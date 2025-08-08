# GRPO-LLM-FINETUNING

GRPO implementation for enhancing mathematical reasoning capabilities of Qwen2.5-0.5B-Instruct using structured XML output and multi-objective rewards.

## Setup

```bash
pip install -r requirements.txt

# Download model
mkdir ./Qwen2.5-0.5B-Instruct 
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct

# Optional: Download GSM8K dataset locally
mkdir -p ./data/gsm8k
wget -O ./data/gsm8k/train.jsonl "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
wget -O ./data/gsm8k/test.jsonl "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
```

## Usage

```bash
python check_model.py  # Verify setup
python train.py         # Start GRPO training
```

## Features

- Multi-objective reward functions (correctness, format, reasoning structure)
- Structured XML output training with `<reasoning>` and `<answer>` tags
- Automatic fallback to local dataset if online loading fails
- SwanLab integration for experiment tracking

## Configuration

Edit training parameters in `train.py`:
- Learning rate, batch size, epochs
- Reward function weights
- Output directory paths
