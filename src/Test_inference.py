from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Teste de inferência com modelo LoRA treinado
base_dir = os.path.dirname(__file__)

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_dir = os.path.join(base_dir, "TinyLlama-lora-out")
output_dir = "TinyLlama-merged"

model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, lora_model_dir)
model = model.merge_and_unload()
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

# Prompt de teste
prompt = (
    "<|system|>\n"
    "You are BuilderUI. Always answer ONLY with JSON in the BuilderUI pattern, never XML or HTML.\n"
    "<|user|>\n"
    "Create two login forms using the BuilderUI JSON pattern, similar to Delphi forms. Example:\n"
    "Input: Generate a simple login form.\n"
    "Output:\n"
    "{\n"
    '  "Type": "TForm",\n'
    '  "Name": "FrmLogin",\n'
    '  "Children": [\n'
    '    {"Type": "TLabel", "Name": "LblUser", "Text": "User:"},\n'
    '    {"Type": "TEdit", "Name": "EdtUser"},\n'
    '    {"Type": "TLabel", "Name": "LblPassword", "Text": "Password:"},\n'
    '    {"Type": "TEdit", "Name": "EdtPassword"},\n'
    '    {"Type": "TButton", "Name": "BtnLogin", "Caption": "Login"}\n'
    '  ]\n'
    '}\n'
    "Now, create two login forms as JSON only.\n"
    "<|assistant|>\n"
)

# Tokeniza
inputs = tokenizer(prompt, return_tensors="pt")

# Gera resposta
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
        )

# Decodifica e mostra a resposta
print(tokenizer.decode(outputs[0], skip_special_tokens=True))