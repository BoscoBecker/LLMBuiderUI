import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Caminho do dataset (relativo ao script)
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "dataset", "dataset.json")

# Carrega o dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Garante que "output" seja string JSON formatada
dataset_data = []
for item in data:
    dataset_data.append({
        "input": item["input"],
        "output": json.dumps(item["output"], ensure_ascii=False, indent=2)
    })

# Cria dataset Hugging Face
dataset = Dataset.from_list(dataset_data)

# Tokenizer e modelo base Phi-2
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Pré-processamento
def format_example(example):
    output_str = str(example['output'])
    prompt = f"<|system|>\nVocê é um BuiderUI.\n<|user|>\n{example['input']}\n<|assistant|>\n{output_str}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(format_example, batched=False)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Diretórios de saída (Windows-friendly)
output_dir = os.path.join(base_dir, "TinyLlama-lora-out")
logs_dir = os.path.join(base_dir, "logs")

# Treinamento otimizado para CPU e 16GB RAM
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # 5 épocas para rodar durante a noite
    per_device_train_batch_size=1,  # tente 2, aumente se não der OOM
    gradient_accumulation_steps=1,  # batch efetivo de 4
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=logs_dir,
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=1,
    fp16=False,  # garante que não tentará usar float16
    no_cuda=True # força uso de CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

if __name__ == "__main__":
    trainer.train()
    # Salva modelo e tokenizer LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
