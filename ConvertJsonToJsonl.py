import json

# Caminho do seu dataset original
input_path = "src\\dataset\\dataset.json"
output_path = "src\\dataset\\dataset.jsonl"

# Carrega o dataset
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Corrige os registros
fixed_data = []
for entry in data:
    fixed_entry = {
        "instruction": entry.get("input", ""),  # renomeia input
        "input": "",  # campo obrigatório no padrão Alpaca
        "output": json.dumps(entry["output"], ensure_ascii=False, indent=2)  # converte JSON → string
    }
    fixed_data.append(fixed_entry)

# Salva como JSONL (um JSON por linha)
with open(output_path, "w", encoding="utf-8") as f:
    for item in fixed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✔ Dataset convertido e salvo em: {output_path}")
