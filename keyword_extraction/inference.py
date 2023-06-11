import argparse
import os, json
from keyword_extraction import model as ke_model

def run_inference(text, model_name):    
    cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    from_scratch = False 
    config['model_name'] = model_name
    model = ke_model.KELightningModel(config, from_scratch, inference=True)
    model.eval()

    labels = model.infer(text)[0]
    print(model.config.max_length)
    tokens = model.tokenizer.tokenize(text,truncation=True, padding=True, max_length=256)
    keywords = []
    for i in range(len(tokens)):
        if labels[i] == 1:
            keyword = tokens[i]
            for j in range(i+1, len(tokens)):
                if labels[j] == 2:
                    keyword += " " + tokens[j]
                else:
                    break
            keyword = model.tokenizer.convert_tokens_to_string(model.tokenizer.tokenize(keyword))
            keywords.append(keyword)

    return keywords




