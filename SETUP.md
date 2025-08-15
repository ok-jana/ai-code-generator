# Setup Guide

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training/inference)
- Hugging Face account and token

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Hugging Face Authentication

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. You can get a token from: https://huggingface.co/settings/tokens

### 3. Run the Application

```bash
python app.py
```

The web interface will be available at: http://localhost:5000

## Environment Variables

You can configure the following environment variables:

- `MODEL_DIR`: Path to the trained model directory (default: "model")
- `CUDA_DEVICE_INDEX`: GPU device index (-1 for CPU, 0+ for GPU)
- `HUGGINGFACE_HUB_TOKEN`: Your Hugging Face authentication token

## Troubleshooting

### Authentication Error (401)
```
Solution: Run `huggingface-cli login` and enter a valid token
```

### CUDA Out of Memory
```
Solution: Set CUDA_DEVICE_INDEX=-1 to use CPU
```

### Model Not Found
```
Solution: Ensure the model directory exists and contains trained files
