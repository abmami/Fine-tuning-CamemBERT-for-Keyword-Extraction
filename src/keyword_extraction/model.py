import torch
import lightning.pytorch as pl
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig,AutoModelForSequenceClassification
import torch
import lightning.pytorch as pl
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
from torch.optim import Adam
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class KELightningModel(pl.LightningModule):
    def __init__(self, cfg, from_scratch, inference=False):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg["model_name"]
        self.max_length = cfg["max_length"]
        self.num_labels = cfg["num_labels"]
        self.lr = cfg["lr"]
        self.betas = eval(cfg["betas"])
        self.eps = cfg["eps"]

        if from_scratch:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            ).to("cuda")
            self.config = self.model.config
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        else:
            model_path = cfg["models_folder"] + self.model_name  
            # load model from pt file
            self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=self.num_labels).to("cuda")
            # load tokenizer from folder
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # load config from folder
            self.config = AutoConfig.from_pretrained(model_path)

            
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        acc = (outputs.logits.argmax(-1) == batch['labels']).float().mean()
        self.log('train_acc', acc)
        f1 = f1_score(batch['labels'].detach().cpu().numpy().flatten(), outputs.logits.argmax(-1).detach().cpu().numpy().flatten(), average='macro')
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        acc = (outputs.logits.argmax(-1) == batch['labels']).float().mean()
        f1 = f1_score(batch['labels'].detach().cpu().numpy().flatten(), outputs.logits.argmax(-1).detach().cpu().numpy().flatten(), average='macro')
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1', f1)
    
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        self.log('test_loss', loss)
        acc = (outputs.logits.argmax(-1) == batch['labels']).float().mean()
        self.log('test_acc', acc)
        f1 = f1_score(batch['labels'].cpu().numpy().flatten(), outputs.logits.argmax(-1).cpu().numpy().flatten(), average='macro')
        self.log('test_f1', f1)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, eps=self.eps, betas=self.betas)
    
    def infer(self, text):
        tokens = self.tokenizer.tokenize(text,truncation=True, padding=True, max_length=self.max_length)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        outputs = self.model(torch.tensor([input_ids]).to('cuda'), torch.tensor([attention_mask]).to('cuda'))
        return outputs.logits.argmax(-1)
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)
    

def run_trainer(train_dataloader, val_dataloader, test_dataloader, config, from_scratch):

    epochs = config['epochs']
    accumulate_grad_batches = config['accumulate_grad_batches']
    model = KELightningModel(config, from_scratch)
    trainer = pl.Trainer(accelerator='auto', max_epochs=epochs, devices=[0], accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    return model, trainer