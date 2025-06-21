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
    with open('src/dataset/dataset.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Traduzir o campo "input"
        if "input" in item and isinstance(item["input"], str):
            item["input"] = translate_text(item["input"], translator)
        # Traduzir recursivamente o campo "output"
        if "output" in item:
            item["output"] = translate_obj(item["output"], translator)

    with open('src/dataset/dataset_pt.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()