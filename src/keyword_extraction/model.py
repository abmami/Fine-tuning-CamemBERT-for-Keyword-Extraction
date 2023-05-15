import torch
import lightning.pytorch as pl
from transformers import AutoModelForTokenClassification
import torch
import lightning.pytorch as pl
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
from torch.optim import Adam

class KELightningModel(pl.LightningModule):
    def __init__(self, num_labels, model_name="camembert-base", max_length=256):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)       
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=self.num_labels)
        self.config = self.model.config
        

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        acc = (outputs.logits.argmax(-1) == batch['labels']).float().mean()
        self.log('train_acc', acc)
        f1 = f1_score(batch['labels'].cpu().numpy().flatten(), outputs.logits.argmax(-1).cpu().numpy().flatten(), average='macro')
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        acc = (outputs.logits.argmax(-1) == batch['labels']).float().mean()
        f1 = f1_score(batch['labels'].cpu().numpy().flatten(), outputs.logits.argmax(-1).cpu().numpy().flatten(), average='macro')
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
        #return AdamW(self.parameters(), lr=5e-5)
        return Adam(self.parameters(), lr=2e-5, eps=1e-08, betas=(0.9, 0.999))
    
    def infer(self, text):
        tokens = self.tokenizer.tokenize(text,truncation=True, padding=True, max_length=self.max_length)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        outputs = self.model(torch.tensor([input_ids]), torch.tensor([attention_mask]))
        return outputs.logits.argmax(-1)
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)
    

def run_trainer(train_dataloader, val_dataloader, test_dataloader, epochs=1, lr=2e-5, max_length=256):
    print("Running task: ke")
    
    model = KELightningModel(num_labels=3, model_name="camembert-base", max_length=max_length)
    trainer = pl.Trainer(accelerator='auto', max_epochs=1, devices=[0], accumulate_grad_batches=8)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    return model, trainer