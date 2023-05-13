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

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)


data_folder = "data/"

data = {
    "PAWS-C-FR":{
    "task":"ss",
    "train": data_folder + "PAWS-C-FR/translated_train.tsv",
    "dev": data_folder + "PAWS-C-FR/dev_2k.tsv",
    "test": data_folder + "PAWS-C-FR/test_2k.tsv"
    }
}

### Sentence Similarity Task

class SSDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, batch_size=16):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Get the datasets:
        train = pd.read_csv(self.data['train'], delimiter='\t', on_bad_lines='skip')
        dev = pd.read_csv(self.data['dev'], delimiter='\t', on_bad_lines='skip')
        test = pd.read_csv (self.data['test'], delimiter='\t', on_bad_lines='skip')

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
        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length = 128, truncation=True)

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


def load_ss(batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    batch_size = batch_size
    ss_data_module = SSDataModule(data['PAWS-C-FR'], tokenizer, batch_size=batch_size)
    ss_data_module.setup()
    train_dataloader = ss_data_module.train_dataloader()
    val_dataloader = ss_data_module.val_dataloader()
    test_dataloader = ss_data_module.test_dataloader()
    print("Train dataset size: ", len(ss_data_module.train_dataset))
    print("Validation dataset size: ", len(ss_data_module.dev_dataset))
    print("Test dataset size: ", len(ss_data_module.test_dataset))
    return train_dataloader, val_dataloader, test_dataloader

class SSLightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            ).to("cuda")
            self.model = AutoModelForSequenceClassification.from_config(config)
            self.config = config
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to("cuda")
            self.config = self.model.config
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def training_step(self, batch):
        out = self.forward(batch)
        logits = out.logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)
        preds = torch.max(out.logits, -1).indices
        acc = (batch["labels"] == preds).float().mean()
        self.log("valid/acc", acc)
        # If you’re trying to clear up the attached computational graph, use .detach() instead.
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def test_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)
        preds = torch.max(out.logits, -1).indices
        acc = (batch["labels"] == preds).float().mean()
        self.log("test/acc", acc)
        # If you’re trying to clear up the attached computational graph, use .detach() instead.
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        self.log("test/f1", f1)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)
    

### Keyword Extraction Task


def load_ke():
    pass



### Loaders and Trainers

def load_dataset(task_name, batch_size=16):
    """This will load, preprocess and return the dataloaders for the given task."""
    if task_name == "ss":
        return load_ss(batch_size=batch_size)
    elif task_name == "ke":
        return load_ke()
    else:
        raise ValueError("Task not found: %s" % (task_name))
    

    
def run_trainer(task_name,train_dataloader, val_dataloader, test_dataloader, epochs, lr, weight_decay, from_scratch):
    """This will run the trainer for the given task."""
    if task_name == "ss":
        model_name = "camembert-base"
        num_labels = 2

        if epochs is None:
            epochs = 1

        if lr is None:
            lr = 3e-5

        if weight_decay is None:
            weight_decay = 0.
        

        model = SSLightningModel(model_name, num_labels, lr, weight_decay, from_scratch=from_scratch)
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
            model_checkpoint,
        ]

    elif task_name == "ke":
        model_name = "camembert-base"
        num_labels = 2
    else:
        raise ValueError("Task not found: %s" % (task_name))
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu", devices="auto",
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
    return model, trainer
