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



class SSLightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            config = AutoConfig.from_pretrained("../models/ss")
            self.tokenizer = AutoTokenizer.from_pretrained("../models/ss")
            self.model = AutoModelForSequenceClassification.from_config(config).to("cuda")
            self.config = config
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to("cuda")
            self.config = self.model.config
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

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

    def inference(self, sentence_1, sentence_2):
        text = [[str(x), str(y)] for x,y in zip([sentence_1], [sentence_2])]
        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length = 128, truncation=True)
        out = self.model(**tokens.to('cuda'))
        preds = torch.max(out.logits, -1).indices
        return preds.detach().cpu().numpy()[0]
    



def run_trainer(train_dataloader, val_dataloader, test_dataloader, epochs, lr, weight_decay, from_scratch):
    print("Running task: ss")

    model_name = "camembert-base"
    num_labels = 2

    if epochs is None:
        epochs = 1

    if lr is None:
        lr = 3e-5

    if weight_decay is None:
        weight_decay = 0.
    

    model = SSLightningModel(model_name, num_labels, lr, weight_decay, from_scratch=True)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu", devices="auto",
        callbacks=callbacks,
        accumulate_grad_batches=8
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    return model, trainer
