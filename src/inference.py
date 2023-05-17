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
import argparse
import time

from sentence_similiarity import model as ss_model
from utils import *

# inference using command line
def run_inference(sentences):
    model_name = "camembert-base"
    num_labels = 2
    lr = models['ss']['lr']
    weight_decay = models['ss']['weight_decay']
    max_length = models['ss']['max_length']
    from_scratch = False # for inference we load local model
    model = ss_model.SSLightningModel(model_name, num_labels, lr, weight_decay, max_length, from_scratch=from_scratch)
    model.to('cuda')
    inference = model.inference(sentence_1=sentences[0], sentence_2=sentences[1])
    return inference


def main():
    parser = argparse.ArgumentParser(description='Inference on a task')
    parser.add_argument('--sentence1', type=str, default="Le chat est sur le tapis.", help='sentence 1 to run inference on')
    parser.add_argument('--sentence2', type=str, default="Le chien est sur le canapé.", help='sentence 2 to run inference on')
    args = parser.parse_args()
    sentences = [args.sentence1, args.sentence2]

    time_start = time.time()
    predicted_label = run_inference(sentences)
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



