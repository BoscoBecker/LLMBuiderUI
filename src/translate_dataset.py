import json
from googletrans import Translator

# Dicionário para tradução de nomes de componentes (exemplo, adicione mais conforme necessário)
name_map = {
    "LblContact": "LblContato",
    "LblName": "LblNome",
    "LblUser": "LblUsuario",
    "LblPassword": "LblSenha",
    "LblEmail": "LblEmail",
    "LblAddress": "LblEndereco",
    "LblCity": "LblCidade",
    "LblState": "LblEstado",
    "LblCountry": "LblPais",
    "LblPhone": "LblTelefone",
    "LblHelp": "LblAjuda",
    "LblHome": "LblInicio",
    "LblAbout": "LblSobre",
    "LblFeedback": "LblFeedback",
    "LblNotes": "LblNotas",
    "LblResult": "LblResultado",
    "LblDescription": "LblDescricao",
    "LblOption": "LblOpcao",
    # Adicione mais conforme necessário
}

def translate_text(text, translator):
    try:
        return translator.translate(text, src='en', dest='pt').text
    except Exception:
        return text

def translate_obj(obj, translator):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Traduzir nomes de componentes
            if k == "Name" and isinstance(v, str) and v in name_map:
                new_obj[k] = name_map[v]
            # Traduzir Caption ou Text
            elif k in ("Caption", "Text") and isinstance(v, str):
                new_obj[k] = translate_text(v, translator)
            # Traduzir recursivamente
            else:
                new_obj[k] = translate_obj(v, translator)
        return new_obj
    elif isinstance(obj, list):
        return [translate_obj(item, translator) for item in obj]
    else:
        return obj

def main():
    translator = Translator()
    with open('src/dataset/dataset.jsonl', encoding='utf-8') as fin, \
         open('src/dataset/dataset_ptbr.jsonl', 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            # Traduzir o campo "instruction"
            if "instruction" in item and isinstance(item["instruction"], str):
                item["instruction"] = translate_text(item["instruction"], translator)
            # Traduzir o campo "input"
            if "input" in item and isinstance(item["input"], str):
                item["input"] = translate_text(item["input"], translator)
            print(f"Traduzindo: {item.get('instruction', '')} | {item.get('input', '')}")    
            # Traduzir recursivamente o campo "output"
            if "output" in item:
                item["output"] = translate_obj(item["output"], translator)
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()