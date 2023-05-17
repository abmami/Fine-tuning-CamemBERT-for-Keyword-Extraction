import torch
import lightning.pytorch as pl
import numpy as np
import random
import warnings
import argparse
import time 
from sentence_similiarity import dataloader as ss_dataloader
from sentence_similiarity import model as ss_model
from keyword_extraction import dataloader as ke_dataloader
from keyword_extraction import model as ke_model

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)

from utils import models

def run_task(task_name, from_scratch):
    model_path = models["models_folder"] + task_name + "/"
    batch_size = models[task_name]["batch_size"]
    max_length = models[task_name]["max_length"]

    if task_name == "ss":
        train_dataloader, val_dataloader, test_dataloader = ss_dataloader.load_ss(batch_size=batch_size, max_length=max_length)
        model, trainer = ss_model.run_trainer(train_dataloader, val_dataloader, test_dataloader,from_scratch)

    elif task_name == "ke":
        train_dataloader, val_dataloader, test_dataloader = ke_dataloader.load_ke(batch_size=batch_size, max_length=max_length)
        model, trainer = ke_model.run_trainer(train_dataloader, val_dataloader, test_dataloader,from_scratch)
        
    else:
        raise ValueError("Unknown task name")
    
    model.save_pretrained(model_path)
    trainer.save_checkpoint(model_path + "trainer.ckpt")
    torch.save(model.state_dict(), model_path + "model.pt")
    

def main():

    parser = argparse.ArgumentParser(description='Fine-tune a model on a task')
    parser.add_argument('--task', type=str, default="ss",help='task name')
    args = parser.parse_args()
    task_name = args.task
    time_start = time.time()
    run_task(task_name, from_scratch=False)
    time_end = time.time() 

    print('Task completed in {} seconds'.format(time_end - time_start))
    
if __name__ == "__main__":
    main()




