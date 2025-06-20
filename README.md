# TinyLlama Fine-Tuning with LoRA

This project demonstrates how to fine-tune the [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
model using the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique for efficient parameter-efficient training. 
The training is optimized for CPU environments with limited RAM (e.g., 16GB).

## Project Structure

- **src/TinyLlama.py**: Main script for fine-tuning TinyLlama with LoRA.
- **dataset/dataset.json**: Training data in JSON format.

## Dataset Format

The dataset should be a JSON file containing a list of objects, each with `input` and `output` fields. Example:
```json
[
  {
    "input": "What is the capital of Brazil?",
    "output": "Brasília"
  }
]
```

## Fine-Tuning Details

- **Model**: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Adapter**: LoRA (Low-Rank Adaptation)
- **Target Modules**: `q_proj`, `v_proj`
- **LoRA Config**: `r=8`, `alpha=16`, `dropout=0.05`
- **Batch Size**: 1 (adjustable)
- **Epochs**: 1 (increase for better results)
- **Device**: CPU only (`no_cuda=True`)

## Training

To start fine-tuning, run:
```sh
python src/TinyLlama.py
```

The script will:
- Load and preprocess the dataset.
- Apply LoRA adapters to the model.
- Train using Hugging Face's `Trainer` API.
- Save the fine-tuned model and tokenizer to the `TinyLlama-lora-out` directory.

## Output

- **Fine-tuned Model**: Saved in `TinyLlama-lora-out/`
- **Logs**: Saved in `logs/`

## Requirements

- Python 3.8+
- [transformers](https://pypi.org/project/transformers/)
- [datasets](https://pypi.org/project/datasets/)
- [peft](https://pypi.org/project/peft/)
- torch

Install dependencies:
```sh
pip install -r src/requirements.txt
```

## Notes

- The script is optimized for CPU training. For GPU, set `no_cuda=False` and adjust `fp16` as needed.
- LoRA enables efficient fine-tuning with minimal memory usage.
- Adjust hyperparameters (epochs, batch size) based on your hardware and dataset size.

## References

- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
