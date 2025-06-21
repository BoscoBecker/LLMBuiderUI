# TinyLlama Fine-Tuning with LoRA

This project demonstrates how to fine-tune the [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
model using the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique for efficient parameter-efficient training. The training is optimized for CPU environments with limited RAM (e.g., 16GB).

## Project Structure

- **src/TrainTinyLlama.py**: Main script for fine-tuning TinyLlama with LoRA.
- **dataset/dataset.json**: Training data in JSON format.

## Dataset Format

The dataset should be a JSON file containing a list of objects, each with `input` and `output` fields. Example:

```json

  {
    "input": "Generate a form  with a panel with color white",
    "output": {
                "Type": "TForm",
                "Name": "FrmMainForm",
                "Caption": "Sample Form",
                "Width": 800,
                "Height": 600,
                "Children": [
                  {
                    "Type": "TPanel",
                    "Name": "Panel1",
                    "Left": 10,
                    "Top": 10,
                    "Width": 200,
                    "Height": 100,
                    "Color": "#FFFFFF"
                  }
                ]
              }
  }

```

## Fine-Tuning Details

- **Model**: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Adapter**: LoRA (Low-Rank Adaptation)
- **Target Modules**: `q_proj`, `v_proj`
- **LoRA Config**: `r=8`, `alpha=16`, `dropout=0.05`
- **Batch Size**: 1 (adjustable)
- **Epochs**: 1 (increase for better results)
- **Device**: CPU only (`use_cpu=True`)

## Training

To start fine-tuning, run:

```sh
python src/TrainTinyLlama.py
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

## Merge LoRA weights into base model

```sh
python src\Merge_lora.py
```

## Convert to gguf

python convert_hf_to_gguf.py ../TinyLlama-merged --outfile ./tinyllama-custom.gguf

## Import to Ollama

Windows > %USERPROFILE%\.ollama\models  

> Create a Modelfile
FROM ./tinyllama-custom.gguf

```sh
ollama create tinyllama-custom -f Modelfile
```

## llama.cpp

- The `.gguf` format is compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp), a C++ project for efficient execution of Llama models on CPU and GPU.
- To use your custom model with `llama.cpp`, simply copy the `.gguf` file to the models folder and follow the instructions in the repository.
- Documentation: [llama.cpp README](https://github.com/ggerganov/llama.cpp#readme)
- Model conversion: [convert.py](https://github.com/ggerganov/llama.cpp/blob/master/convert.py)

**Example usage:**

```sh
./main -m ./tinyllama-custom.gguf -p "Your prompt here"
```

## Notes

- The script is optimized for CPU training. For GPU, set `no_cuda=False` and adjust `fp16` as needed.
- LoRA enables efficient fine-tuning with minimal memory usage.
- Adjust hyperparameters (epochs, batch size) based on your hardware and dataset size.

## References

- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [llama.cpp (GitHub)](https://github.com/ggerganov/llama.cpp)
