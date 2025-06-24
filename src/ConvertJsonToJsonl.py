import json

input_path = "src\\dataset\\dataset.json"
output_path = "src\\dataset\\dataset.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_data = []
for entry in data:
    fixed_entry = {
        "instruction": entry.get("input", ""),  
        "input": "",  
        "output": json.dumps(entry["output"], ensure_ascii=False, indent=2)
    }
    fixed_data.append(fixed_entry)


with open(output_path, "w", encoding="utf-8") as f:
    for item in fixed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✔ Process done, path: {output_path}")
