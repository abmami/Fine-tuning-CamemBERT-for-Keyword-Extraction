import streamlit as st
import json, os
from keyword_extraction.model import KELightningModel

def load_model(model_name):
    with open("../config.json") as f:
        config = json.load(f)
    from_scratch = False 
    cfg = config['models']['ke']
    cfg.update({"models_folder": "../" + config['models']['models_folder'], "num_labels": 3, "model_name": model_name})
    return KELightningModel(cfg, from_scratch)

def extract_keywords(text, model):
    # use infer method to get the labels
    labels = model.infer(text)[0]
    # get the tokens from the text
    tokens = model.tokenizer.tokenize(text,truncation=True, padding=True, max_length=256)
    # get the keywords from the tokens and labels
    keywords = []
    for i in range(len(tokens)):
        if labels[i] == 1:
            keyword = tokens[i]
            for j in range(i+1, len(tokens)):
                if labels[j] == 2:
                    keyword += " " + tokens[j]
                else:
                    break
            # convert the keyword to the original string
            keyword = model.tokenizer.convert_tokens_to_string(model.tokenizer.tokenize(keyword))
            keywords.append(keyword)

    return keywords


st.title('Keyword Extraction Demo')
st.write('This app extracts keywords from French text using a fine-tuned CamemBERT model')
st.write('Please enter your text in the box below')
text = st.text_area('Text to analyze', height=200)

models = {
    'CamemBERT + Pre-finetuning + Finetuning': 'camembert-ss-ke',
    #'CamemBERT + Pre-finetuning + Finetuning + Quantization': 'quantized-camembert-ss-ke',
    'CamemBERT + Finetuning': 'camembert-ke',
    #'CamemBERT + Finetuning + Quantization': 'quantized-camembert-ke'
}

selected_model = st.selectbox('Choose a model', list(models.keys()))

# extract keywords

if st.button('Extract keywords'):
    loaded_model = load_model(models[selected_model])
    keywords = extract_keywords(text, loaded_model)
    for keyword in keywords:
        text = text.replace(keyword, "<span style='background-color: #e6f7ff; font-weight: bold;'>"+keyword+"</span>")
    st.markdown(text, unsafe_allow_html=True)
    st.write('Keywords: ', keywords)