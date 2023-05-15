import torch
import lightning.pytorch as pl
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pl.trainer.seed_everything(42)

from sentence_similiarity import dataloader as ss_dataloader
from sentence_similiarity import model as ss_model

from keyword_extraction import dataloader as ke_dataloader
from keyword_extraction import model as ke_model

def run_task(task_name, batch_size, epochs, lr, weight_decay, from_scratch=False, quantize=False):
    if task_name == "ss":
        
        train_dataloader, val_dataloader, test_dataloader = ss_dataloader.load_ss(batch_size=batch_size)
        model, trainer = ss_model.run_trainer(train_dataloader, val_dataloader, test_dataloader, epochs, lr, weight_decay, from_scratch)
        ## save model
        model.save_pretrained("../models/ss")

        ## save trainer 
        trainer.save_checkpoint("../models/ss/trainer.ckpt")

        ## torch save pt
        torch.save(model.state_dict(), "../models/ss/model.pt")

    elif task_name == "ke":
        train_dataloader, val_dataloader, test_dataloader = ke_dataloader.load_ke(batch_size=4)
        model, trainer = ke_model.run_trainer(train_dataloader, val_dataloader, test_dataloader, epochs=3, lr=2e-5, max_length=256)
        ## save model
        model.save_pretrained("../models/ke")

        ## save trainer 
        trainer.save_checkpoint("../models/ke/trainer.ckpt")

        ## torch save pt
        torch.save(model.state_dict(), "../models/ke/model.pt")
    else:
        raise ValueError("Unknown task name")
    

### main with arguments

import argparse
import time 
def main():
    # task_name, batch_size, epochs, lr, weight_decay

    # task_name, batch_size, epochs, lr, max_length

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



