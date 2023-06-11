import streamlit as st
from keyword_extraction import inference

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
    keywords = inference.run_inference(text, models[selected_model])
    for keyword in keywords:
        text = text.replace(keyword, "<span style='background-color: #e6f7ff; font-weight: bold;'>"+keyword+"</span>")
    st.markdown(text, unsafe_allow_html=True)
    st.write('Keywords: ', keywords)