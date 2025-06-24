import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "dataset", "dataset.jsonl")


data = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

dataset_data = []
for item in data:
    dataset_data.append({
        "input": item["instruction"],  
        "output": json.dumps(item["output"], ensure_ascii=False) 
    })

dataset = Dataset.from_list(dataset_data)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

def format_example(example):
    output_str = str(example['output'])
    prompt = (
        "<|system|>\n"
        "You are BuilderUI. Always answer ONLY with JSON in the BuilderUI pattern, never XML or HTML or plain text.\n"
        "<|user|>\n"
        f"{example['input']}\n"
        "<|assistant|>\n"
        f"{output_str}"
    )
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

output_dir = os.path.join(base_dir, "TinyLlama-lora-out")
logs_dir = os.path.join(base_dir, "logs")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,  
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=2, 
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=logs_dir,
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=1,
    fp16=False,  
    use_cpu=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
