# Training Documentation

## Overview

This project uses LoRA (Low-Rank Adaptation) to fine-tune the Salesforce/codegen-350M-mono model for Python code generation.

## Training Process

### 1. Dataset Format

The training data is stored in `data/train.jsonl` with the following format:

```json
{"prompt": "Write a function that returns the square of a number", "completion": "def square(n):\n    return n * n"}
{"prompt": "Create a list comprehension that returns squares from 0 to 9", "completion": "squares = [x**2 for x in range(10)]"}
```

### 2. Running Training

Basic training:
```bash
python train.py
```

Custom training options:
```bash
python train.py --train_file data/custom.jsonl --num_epochs 10 --output_dir custom_model
```

### 3. Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_file` | `data/train.jsonl` | Path to training dataset |
| `--model_name` | `Salesforce/codegen-350M-mono` | Base model to fine-tune |
| `--output_dir` | `model` | Directory to save trained model |
| `--num_epochs` | `5` | Number of training epochs |

### 4. LoRA Configuration

The model uses the following LoRA settings:

- **Rank (r)**: 8
- **Alpha**: 16
- **Dropout**: 0.05
- **Bias**: "none"
- **Task type**: "CAUSAL_LM"

### 5. Training Results

Based on the last training run:
- **Training Loss**: Decreased from 0.9581 to 0.2504
- **Training Time**: ~11.7 minutes for 5 epochs
- **Dataset Size**: 20 examples
- **Steps**: 50 total training steps

## Adding Training Data

To improve the model, add more examples to your JSONL file:

```json
{"prompt": "Your instruction here", "completion": "# Your Python code here"}
```

### Best Practices for Training Data

1. **Clear Instructions**: Write specific, clear prompts
2. **Quality Code**: Ensure completions are correct and well-formatted
3. **Variety**: Include different types of Python constructs
4. **Consistency**: Use consistent formatting and style

### Example Training Data Categories

- Basic functions
- List comprehensions
- Class definitions
- Control flow structures
- Common algorithms
- File operations
- Error handling
- Decorators and context managers

## Model Output

The trained model will be saved to the specified output directory with:
- LoRA adapter weights
- Tokenizer files
- Configuration files
- Training checkpoints (every 100 steps)
