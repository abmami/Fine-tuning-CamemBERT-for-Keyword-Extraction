from pprint import pprint
import functools
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
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
from utils import load_dataset, run_trainer

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)


def run_task(task_name, batch_size, epochs, lr, weight_decay, from_scratch=False, quantize=False):
    if task_name == "ss":
        train_dataloader, val_dataloader, test_dataloader = load_dataset("ss", batch_size=batch_size)
        model, trainer = run_trainer(task_name,train_dataloader, val_dataloader, test_dataloader, epochs, lr, weight_decay, from_scratch)
        ## save model
        model.save_pretrained("models/ss")

        ## save trainer 
        trainer.save_checkpoint("models/ss/trainer.ckpt")

        ## torch save pt
        torch.save(model.state_dict(), "models/ss/model.pt")

    elif task_name == "ke":
        train_dataloader, val_dataloader, test_dataloader = load_dataset("ke", batch_size=batch_size)
    else:
        raise ValueError("Unknown task name")
    

### main with arguments

import argparse
import time 
def main():
    # task_name, batch_size, epochs, lr, weight_decay

    parser = argparse.ArgumentParser(description='Fine-tune a model on a task')
    parser.add_argument('--task', type=str, default="ss",help='task name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--from_scratch', type=bool, default=False, help='from scratch')
    parser.add_argument('--quantize', type=bool, default=False, help='quantize')

    args = parser.parse_args()
    task_name = args.task
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    wd = args.wd
    from_scratch = args.from_scratch
    quantize = args.quantize

    print("Running task: ", task_name, " with batch size: ", batch_size, " and epochs: ", epochs, " and lr: ", lr, " and wd: ", wd, ", from_scratch: ", from_scratch, " and quantize: ", quantize)
    time_start = time.time()
    if task_name == "ss":
        run_task("ss", batch_size, epochs, lr, wd, from_scratch, quantize)
        
    elif task_name == "ke":
        run_task("ke", batch_size, epochs, lr, wd, from_scratch,quantize)
    else:
        raise ValueError("Unknown task name")
    time_end = time.time() 
    print('Task completed in {} seconds'.format(time_end - time_start))
    
if __name__ == "__main__":
    main()


### To run the script:

# python run_task.py --task ss --batch_size 16 --epochs 1 --lr 3e-5 --wd 0. --from_scratch False --quantize False

# --task: task name, ss for sentence similarity and ke for keyphrase extraction
# --batch_size: batch size
# --epochs: number of epochs
# --lr: learning rate
# --wd: weight decay
# --from_scratch: whether to train from scratch or not
# --quantize: whether to quantize the model or not



