import torch
import lightning.pytorch as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.metrics import f1_score
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
    def __init__(self, cfg, from_scratch):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg["model_name"]

        if from_scratch:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            ).to("cuda")
            self.config = self.model.config
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            model_path = cfg["models_folder"] + "ss/"
            config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_config(config).to("cuda")
            self.config = config
        self.lr = cfg["lr"]
        self.weight_decay = cfg["weight_decay"]
        self.num_labels = self.model.num_labels
        self.max_length = cfg["max_length"]

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
        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length = self.max_length, truncation=True)
        out = self.model(**tokens.to('cuda'))
        preds = torch.max(out.logits, -1).indices
        return preds.detach().cpu().numpy()[0]
    

def run_trainer(train_dataloader, val_dataloader, test_dataloader, config, from_scratch=False) :
    print("Running task: ss")

    from_scratch = True # will be removed later
    cfg = config['models']['ss']
    cfg.update({"models_folder": config['models']['models_folder'], "num_labels": 2, "model_name": "camembert-base"})

    model = SSLightningModel(cfg, from_scratch=from_scratch)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator="gpu", devices="auto",
        callbacks=callbacks,
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    return model, trainer
