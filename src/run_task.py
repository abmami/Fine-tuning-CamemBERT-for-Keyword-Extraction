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
import os
import json

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)

# Load config json
root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
os.chdir(root_dir)

with open("config.json") as f:
    config = json.load(f)

def run_task(task_name, from_scratch=True):
    model_path = config['models']["models_folder"] + task_name + "/"

    if task_name == "ss":
        train_dataloader, val_dataloader, test_dataloader = ss_dataloader.load_ss(config=config)
        model, trainer = ss_model.run_trainer(train_dataloader, val_dataloader, test_dataloader, config, from_scratch=from_scratch)

    elif task_name == "ke":
        train_dataloader, val_dataloader, test_dataloader = ke_dataloader.load_ke(config=config)
        model, trainer = ke_model.run_trainer(train_dataloader, val_dataloader, test_dataloader, config, from_scratch=from_scratch)
        
    else:
        raise ValueError("Unknown task name")
    
    model.save_pretrained(model_path)
    trainer.save_checkpoint(model_path + "trainer.ckpt")
    torch.save(model.state_dict(), model_path + "model.pt")
    

def main():

    parser = argparse.ArgumentParser(description='Fine-tune a model on a task')
    parser.add_argument('--task', type=str, default="ss",help='task name: ss or ke')
    args = parser.parse_args()
    task_name = args.task
    time_start = time.time()
    run_task(task_name)
    time_end = time.time() 

    print('Task completed in {} seconds'.format(time_end - time_start))
    
if __name__ == "__main__":
    main()




