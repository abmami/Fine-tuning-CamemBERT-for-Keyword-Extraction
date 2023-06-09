import torch
import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def preprocess(data):
    data = data.dropna()
    data = data[data['keywords'] != '']
    data = data.drop_duplicates(subset=['keywords'])
    data['keywords'] = data['keywords'].apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').replace(',', '\t'))
    data['keywords'] = data['keywords'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
    return data


class KEDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_csv = cfg['train']
        self.val_csv = cfg['val']
        self.test_csv = cfg['test']
        self.model_name = "camembert-base"
        self.max_length = cfg['max_length']
        self.batch_size = cfg['batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage=None):
        self.train_data = KEDataset(self.train_csv, self.tokenizer, self.max_length)
        self.val_data = KEDataset(self.val_csv, self.tokenizer, self.max_length)
        self.test_data = KEDataset(self.test_csv, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

class KEDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = preprocess(pd.read_csv(csv_file))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {"B-KEY": 1, "I-KEY": 2, "O": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        keywords = self.data.iloc[index]['keywords'].split('\t')
        
        tokens = self.tokenizer.tokenize(text,truncation=True, padding=True, max_length=self.max_length)
        labels = ["O"] * len(tokens)

        for keyword in keywords:
            keyword_tokens = self.tokenizer.tokenize(keyword,truncation=True, padding=True, max_length=self.max_length)
            # find the start index of the keyword in the tokens
            keyword_start_idx = -1
            for i in range(len(tokens) - len(keyword_tokens) + 1):
                if tokens[i:i+len(keyword_tokens)] == keyword_tokens:
                    keyword_start_idx = i
                    break
            if keyword_start_idx >= 0:
                labels[keyword_start_idx] = "B-KEY"
                for i in range(keyword_start_idx+1, keyword_start_idx+len(keyword_tokens)):
                    labels[i] = "I-KEY"

        label_ids = [self.label2id[label] for label in labels]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        label_ids = label_ids + ([0] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }
    


def load_data(config):
    dm = KEDataModule(config)
    dm.setup()
    return dm

