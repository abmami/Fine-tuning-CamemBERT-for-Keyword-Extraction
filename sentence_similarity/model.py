import torch
import lightning.pytorch as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, CamembertForSequenceClassification, CamembertTokenizer, CamembertConfig
from sklearn.metrics import f1_score
import numpy as np
import random
import warnings
from torch.optim import Adam, AdamW

warnings.filterwarnings("ignore")

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)


class SSLightningModel(pl.LightningModule):
    def __init__(self, cfg, from_scratch):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg["model_name"]
        self.num_labels = cfg["num_labels"]
        self.lr = cfg["lr"]
        self.weight_decay = cfg["weight_decay"]
        self.max_length = cfg["max_length"]

        if from_scratch:
            self.model = CamembertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            ).to("cuda")
            self.config = self.model.config
            self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        else:
            model_path = cfg["models_folder"] + self.model_name
            # load model from pt file
            self.model = CamembertForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels).to("cuda")
            # load tokenizer from folder
            self.tokenizer = CamembertTokenizer.from_pretrained(model_path)
            # load config from folder
            self.config = CamembertConfig.from_pretrained(model_path)



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
        # cosine annealing warm restarts
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1, eta_min=1e-6)
        return [optimizer], [scheduler]
 
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)

    def inference(self, sentence_1, sentence_2):
        text = [[str(x), str(y)] for x,y in zip([sentence_1], [sentence_2])]
        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length = 128, truncation=True)
        out = self.model(**tokens.to('cuda'))        
        return out.logits.argmax(-1).detach().cpu().numpy()[0]

def run_trainer(dm, config, from_scratch) :

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    model = SSLightningModel(config, from_scratch=from_scratch)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/f1", mode="max")
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/f1", patience=3, mode="max"),
        model_checkpoint,
    ]

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator="gpu", devices="auto",
        callbacks=callbacks,
        
        accumulate_grad_batches=config['accumulate_grad_batches'],
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    return model, trainer
