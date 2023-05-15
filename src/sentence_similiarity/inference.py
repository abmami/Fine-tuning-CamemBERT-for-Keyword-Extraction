from pprint import pprint
import functools
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm
import pandas as pd
from datasets import Dataset
import numpy as np
import random
import warnings

from sentence_similiarity import model as ss_model

# Do inference for Sentence Similarity Task using saved model at models/ss

def run_inference(task_name, sentences, quantized):
    if task_name == "ss":
        model = ss_model.SSLightningModel("camembert-base", 2, lr=3e-5, weight_decay=0., from_scratch=False)   
        model.to('cuda')
        inference = model.inference(sentence_1=sentences[0], sentence_2=sentences[1])
        return inference

    elif task_name == "ke":
        pass
    else:
        raise ValueError("Unknown task name")
    

import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Inference on a task')
    parser.add_argument('--task', type=str, default="ss",help='task name')
    parser.add_argument('--quantized', type=bool, default=False, help='use quantized model')
    parser.add_argument('--sentence1', type=str, default="Le chat est sur le tapis.", help='sentence 1 to run inference on')
    parser.add_argument('--sentence2', type=str, default="Le chien est sur le canapé.", help='sentence 2 to run inference on')

    args = parser.parse_args()
    sentences = [args.sentence1, args.sentence2]
    sentences = [
    "La NVIDIA TITAN V a été annoncé officiellement par Nvidia le 7 décembre 2017.",
    "Le 7 décembre 2017 NVIDIA a officiellement annoncé le lancement de la Nvidia TITAN V."
    ]
    time_start = time.time()
    predicted_label = run_inference(args.task, sentences, args.quantized)
    time_end = time.time()
    print("Time taken: {} seconds".format(time_end-time_start))
    print(sentences)
    print(predicted_label)
    if predicted_label == 0:
        print("The sentences are not similar")
    else:
        print("The sentences are similar")

if __name__ == '__main__':
    main()



