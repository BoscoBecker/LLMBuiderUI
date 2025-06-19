from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os 
# Merge LoRA weights into base model and save
base_dir = os.path.dirname(__file__)

base_model_id = os.path.join(base_dir, "TinyLlama-1.1B-Chat-v1.0")
lora_model_dir = os.path.join(base_dir, "TinyLlama-lora-out")
output_dir = os.path.join(base_dir, "TinyLlama-merged")

model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, lora_model_dir)
model = model.merge_and_unload()
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

print("✅ Merge complet", output_dir)
