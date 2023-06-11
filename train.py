import torch
import lightning.pytorch as pl
import numpy as np
import random
import warnings
import argparse
import time 
from sentence_similarity import dataloader as ss_dataloader
from sentence_similarity import model as ss_model
from keyword_extraction import dataloader as ke_dataloader
from keyword_extraction import model as ke_model
import os
import json

def run_task(task_name):
    print("[*] Running task: ", task_name)
    time_start = time.time()

    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    pl.trainer.seed_everything(42)

    if task_name == "ss":
        wdir = "sentence_similarity"
        model_save_dir = os.path.join(wdir, "models", "camembert-" + task_name + "/" )
        
        with open(os.path.join(wdir, "config.json")) as f:
            config = json.load(f)

        dm = ss_dataloader.load_data(config)
        model, trainer = ss_model.run_trainer(dm, config, from_scratch=True)

    elif task_name == "ke":
        wdir = "keyword_extraction"
        with open(os.path.join(wdir, "config.json")) as f:
            config = json.load(f)
        model_save_dir = os.path.join(wdir, "models", "camembert-" + task_name + "/" )
        dm = ke_dataloader.load_data(config)
        model, trainer = ke_model.run_trainer(dm, config, from_scratch=True)
    
    elif task_name == "ss-ke":
        # Fine-tuning the pre-finetuned camembert on Keyword Extraction
        # Doesn't run task ss, assumes it has already been run
        wdir = "keyword_extraction"
        with open(os.path.join(wdir, "config.json")) as f:
            config = json.load(f)
        config["model_name"] = "camembert-ss"
        model_save_dir = os.path.join(wdir, "models", "camembert-" + task_name + "/" )
        dm = ke_dataloader.load_data(config)
        model, trainer = ke_model.run_trainer(dm, config, from_scratch=False, inference=False)
    else:
        raise ValueError("Unknown task name")
    
    model.save_pretrained(model_save_dir)
    trainer.save_checkpoint(model_save_dir + "trainer.ckpt")
    torch.save(model.state_dict(), model_save_dir + "model.pt")
    time_end = time.time() 
    print('[*] Task completed in {} seconds'.format(time_end - time_start))
    print('[*] Model saved to', model_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a model on a task')
    parser.add_argument('--task', type=str, default="ss",help='task name: ss or ke')
    args = parser.parse_args()
    task_name = args.task
    run_task(task_name)




