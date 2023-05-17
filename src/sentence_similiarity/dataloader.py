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
warnings.filterwarnings("ignore")

from utils import data

class SSDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        # Get the datasets:
        train = pd.read_csv(self.data['data_folder'] + self.data['PAWS-C-FR']['train'], delimiter='\t', on_bad_lines='skip')
        dev = pd.read_csv(self.data['data_folder'] + self.data['PAWS-C-FR']['dev'], delimiter='\t', on_bad_lines='skip')
        test = pd.read_csv (self.data['data_folder'] + self.data['PAWS-C-FR']['test'], delimiter='\t', on_bad_lines='skip')

        train.drop(columns=['id'], inplace=True)
        dev.drop(columns=['id'], inplace=True)
        test.drop(columns=['id'], inplace=True)

        train.dropna(inplace=True)
        dev.dropna(inplace=True)
        test.dropna(inplace=True)

        # pipeline test
        #train = train.sample(frac=0.05, random_state=42)
        #dev = dev.sample(frac=0.05, random_state=42)
        #test = test.sample(frac=0.05, random_state=42)

        train['label'] = train['label'].astype(int)
        dev['label'] = dev['label'].astype(int)
        test['label'] = test['label'].astype(int)
        self.train_dataset = Dataset.from_pandas(train)
        self.dev_dataset = Dataset.from_pandas(dev)
        self.test_dataset = Dataset.from_pandas(test)

    def tokenize_batch(self, samples):
        sentence_1 = [sample['sentence1'] for sample in samples]
        sentence_2 = [sample['sentence2'] for sample in samples]
        labels = torch.tensor([sample["label"] for sample in samples])
        str_labels = [sample["label"] for sample in samples]
        text = [[str(x), str(y)] for x,y in zip(sentence_1, sentence_2)]
        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length = self.max_length, truncation=True)

        return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}
    
    def loader(self, dataset, batch_size, shuffle=True, pin_memory=True):
        return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=pin_memory,
        collate_fn=functools.partial(self.tokenize_batch)
    )

    def train_dataloader(self):
        return self.loader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.loader(self.dev_dataset, self.batch_size,shuffle=False)

    def test_dataloader(self):
        return self.loader(self.test_dataset, self.batch_size, shuffle=False)


def load_ss(batch_size=16, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    batch_size = batch_size
    ss_data_module = SSDataModule(data, tokenizer, batch_size=batch_size, max_length=max_length)
    ss_data_module.setup()
    train_dataloader = ss_data_module.train_dataloader()
    val_dataloader = ss_data_module.val_dataloader()
    test_dataloader = ss_data_module.test_dataloader()
    print("Train dataset size: ", len(ss_data_module.train_dataset))
    print("Validation dataset size: ", len(ss_data_module.dev_dataset))
    print("Test dataset size: ", len(ss_data_module.test_dataset))
    return train_dataloader, val_dataloader, test_dataloader