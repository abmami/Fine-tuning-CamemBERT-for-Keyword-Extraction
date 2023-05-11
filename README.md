# French Keywords Extraction

This repository contains code for fine-tuning CamemBERT for French keywords extraction

## Datasets 

The datasets are available in the `data` folder. 

- `final-keys-dataset.csv`: This dataset contains the final dataset used for training and validation. It contains 2 columns: `text`, and `keywords`. The dataset collected from transcribed YouTube videos is merged with WikiNews french keywords dataset for data augmentation. The dataset contains ~340 documents.
- `PAWS-C-FR`: This folder contains the PAWS-X french dataset used for pre-fine-tuning CamemBERT on Sentences similiary task. The dataset contains 3 files: `translated_train.tsv`, `test_2k.tsv`, and `dev_2k`. Each file contains 3 columns: `id`, `sentence1`, and `sentence2`. The `sentence1` column contains the first sentence, the `sentence2` column contains the second sentence, and the `label` column contains the label (0 or 1). 


## Notebooks 

This repository contains the following notebooks:

- `finetuning.ipynb`: This notebook contains code for fine-tuning CamemBERT for French keywords extraction.
- `prefinetuning.ipynb`: This notebook contains code for pre-fine-tuning CamemBERT on Sentences Similarity task using PAWS-C french dataset.

## Results

- Pre-fine-tuning CamemBERT on Sentences Similarity task using PAWS-C french dataset. 

Epochs: 1 , Learning Rate: 3e-5, Batch Size: 16, Optimizer: AdamW

| Model | Validation F1-score | Validation | Accuracy | Params | Size(Mb) |
| --- | --- | --- | --- | --- | --- |
| Prefinetuned CamemBERT | 0.9056935985015315 | 0.906 | 0.906 | 110M | 442 Mb |
| + Dynamic Quantization | 0.3253368690450495 | 0.453 | 110M | 186 Mb |

- Fine-tuning CamemBERT for French keywords extraction.


Epochs: 20, Learning Rate: 5e-5, Batch Size: 8, Optimizer: AdamW

| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Params | Size(Mb) |
| --- | --- | --- | --- | --- | --- | --- |
| Finetuned CamemBERT | 0.001669629953348225 | 0.9996 | 0.09359 | 0.9859 | 110M | 419 Mb |
| + Dynamic Quantization | - | - | 0.2880 | 0.9240 | 110M | 176 Mb |


- Fine-tuning on French keywords extraction using pre-fine-tuned CamemBERT on Sentences Similarity task.

| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Params | Size(Mb) |
| --- | --- | --- | --- | --- | --- | --- |
| Finetuned Prefinetuned CamemBERT | - | - | - | - | 110M | - |
| + Dynamic Quantization | - | - | - | - | 110M | - |






