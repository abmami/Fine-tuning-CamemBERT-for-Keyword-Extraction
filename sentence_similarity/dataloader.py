import functools
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from transformers import AutoTokenizer, CamembertTokenizer
import pandas as pd
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

class SSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = CamembertTokenizer.from_pretrained(config['model_name'])
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']

    def setup(self, stage=None):
        # Get the datasets
        train = pd.read_csv(self.config['train'], delimiter='\t', on_bad_lines='skip')
        dev = pd.read_csv(self.config['val'], delimiter='\t', on_bad_lines='skip')
        test = pd.read_csv (self.config['test'], delimiter='\t', on_bad_lines='skip')

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
        device = torch.device("cuda")
        sentence_1 = [sample['sentence1'] for sample in samples]
        sentence_2 = [sample['sentence2'] for sample in samples]
        labels = torch.tensor([sample["label"] for sample in samples]).to(device)  # Move labels to GPU
        str_labels = [sample["label"] for sample in samples]
        text = [[str(x), str(y)] for x, y in zip(sentence_1, sentence_2)]
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        input_ids = tokens.input_ids.to(device)  # Move input_ids to GPU
        attention_mask = tokens.attention_mask.to(device)  # Move attention_mask to GPU

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}
    
    def loader(self, dataset, batch_size, shuffle=True, pin_memory=True):
        return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=functools.partial(self.tokenize_batch)
    )

    def train_dataloader(self):
        return self.loader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.loader(self.dev_dataset, self.batch_size,shuffle=False)

    def test_dataloader(self):
        return self.loader(self.test_dataset, self.batch_size, shuffle=False)


def load_data(config):
    dm = SSDataModule(config)
    dm.setup()
    return dm