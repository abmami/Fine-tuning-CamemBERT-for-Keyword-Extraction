# UNDER CONSTRUCTION! Pre-finetuning 

from pprint import pprint
import functools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm
import pandas as pd

data = {
    "train": "translated_train.tsv",
    "dev": "dev_2k.tsv",
    "test": "test_2k.tsv"
}

train = pd.read_csv(data['train'], delimiter='\t', on_bad_lines='skip')
dev = pd.read_csv(data['dev'], delimiter='\t', on_bad_lines='skip')
test = pd.read_csv (data['test'], delimiter='\t', on_bad_lines='skip')


train.drop(columns=['id'], inplace=True)
dev.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

train.dropna(inplace=True)
dev.dropna(inplace=True)
test.dropna(inplace=True)

train['label'] = train['label'].astype(int)

# Shape of the data
print(f"Total train samples : {train.shape[0]}")
print(f"Total validation samples: {dev.shape[0]}")
print(f"Total test samples: {test.shape[0]}")


tokenizer = AutoTokenizer.from_pretrained('camembert-base')

def tokenize_batch(samples, tokenizer):
    sentence_1 = [sample['sentence1'] for sample in samples]
    sentence_2 = [sample['sentence2'] for sample in samples]
    labels = torch.tensor([sample["label"] for sample in samples])
    str_labels = [sample["label"] for sample in samples]
    text = [[str(x), str(y)] for x,y in zip(sentence_1, sentence_2)]
    tokens = tokenizer(text, return_tensors="pt", padding='max_length', max_length = 128, truncation=True)

    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, dataset):

        self.label = [i['label'] for i in dataset]
        self.sentence_1 = [i['sentence1'] for i in dataset]
        self.sentence_2 = [i['sentence2'] for i in dataset]
        self.text_cat = [[str(x), str(y)] for x,y in zip(self.sentence_1, self.sentence_2)]

    def __len__(self):

        return len(self.text_cat)

    def get_batch_labels(self, idx):

        return torch.tensor(self.label[idx])

    def get_batch_texts(self, idx):

        return tokenizer(self.text_cat[idx], padding='max_length', max_length = 128, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def collate_fn(texts):

  num_texts = len(texts['input_ids'])
  features = list()
  for i in range(num_texts):
      features.append({'input_ids':texts['input_ids'][i], 'attention_mask':texts['attention_mask'][i]})
  
  return features

from datasets import Dataset

train_dataset = Dataset.from_pandas(train)
dev_dataset = Dataset.from_pandas(dev)
#train_dataset = DataSequence(train_dataset)
#train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)


train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    dev_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
)

batch = next(iter(train_dataloader))

print("\n".join(tokenizer.batch_decode(batch["input_ids"])))
batch["labels"]


class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) et pas les poids du modèle 
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
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
        # -------- MASKED --------
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est 
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    


lightning_model = LightningModel("camembert-base", 2, lr=3e-5, weight_decay=0.)


model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")

camembert_trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu", devices="auto",
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]
)


camembert_trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


lightning_model = LightningModel.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

## Inference https://colab.research.google.com/drive/1W0Fj7aXm2qPx34PbEo0F5s5_sm8vVrJl?usp=sharing#scrollTo=ErEV8E9GUQlP