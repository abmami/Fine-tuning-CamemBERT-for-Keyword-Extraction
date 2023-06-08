# Fine-tuning CamemBERT for Keyword Extraction

This repository is dedicated to fine-tuning CamemBERT for keyword extraction from French text. It provides code for two distinct approaches with different training sequences. 

## Approaches
### Approach 1: Pre-finetuning on Sentence Similarity, followed by Fine-tuning for Keyword Extraction
In this approach, we first pre-finetune CamemBERT on the task of sentence similarity. By training the model to understand the semantic relationships between sentences, it gains a deeper understanding of contextual information. Subsequently, we fine-tune the pre-finetuned model specifically for keyword extraction. This two-step process allows CamemBERT to leverage its enhanced comprehension of language semantics to accurately identify and extract keywords.

### Approach 2: Direct Fine-tuning for Keyword Extraction
In this streamlined approach, we directly fine-tune CamemBERT for the keyword extraction task without pre-finetuning on sentence similarity. By focusing solely on the target task, we optimize the fine-tuning process and enable CamemBERT to learn directly from the keyword extraction data, and just benefit from its language representation capabilities.

## Datasets

## Training

Table summarizing the training parameters for each approach.

| Approach | Pre-training | Pre-finetuning Task | Pre-finetuning Params | Fine-tuning Task | Fine-tuning Params 
| --- | --- | --- | --- | --- | --- |
| Approach 1 | camembert-base | SST | "epochs": 3, "lr": 3e-5, "batch_size": 4, "accumulate_grad_batches": 8, "max_length": 128 | KWE | "epochs": 3, "lr": 2e-5, "batch_size": 4, "accumulate_grad_batches": 8, "max_length": 256, "eps": 1e-8, "betas": "(0.9, 0.999)" |
| Approach 2 | camembert-base | None | None | KWE | "epochs": 3, "lr": 2e-5, "batch_size": 4, "accumulate_grad_batches": 8, "max_length": 256, "eps": 1e-8, "betas": "(0.9, 0.999)"  |

## Results

Table summarizing the results for each approach.

| Approach | Training Sequence | Quantized | Test Accuracy | Test F1 Score |
| --- | --- | --- | --- | --- |
| Approach 1 | SST -> KWE | No | - | - |
| Approach 1 | SST -> KWE | Yes | - | - |
| Approach 2 | KWE | No | - | - |
| Approach 2 | KWE | Yes | - | - |

Note : 

This project aims to showcase the effectiveness of pre-finetuning CamemBERT on a sentence similarity task before fine-tuning it for keyword extraction. Please note that several factors, such as the choice of datasets, fine-tuning parameters, and computational resources, can impact the model's performance. The presented results serve as a general demonstration of the proposed approaches rather than definitive conclusions. To enhance the model's performance, one can conduct further experiments with different configurations. Due to limited computational resources, we were unable to conduct extensive optimization experiments. The project ensures reproducibility and provides automated scripts for running the experiments.

## Run Locally

### Requirements

The code is tested on an environment with :
- Python 3.10.6
- CUDA-enabled GPU (NVIDIA GeForce GTX 1650 Ti 4GB)
- CUDA 11.7
- Torch 2.0.1

To install the required packages, run the following command:
```
pip install -r requirements.txt
```

### Usage

#### Training
We provide two notebooks to run different approaches to fine-tuning CamemBERT for keyword extraction. The notebooks are self-contained and can be run directly from Google Colab. The notebooks are also available in the `notebooks` folder of this repository.

To run pipeline for Approach 1, run the following commands:
```shell
cd src
python preprocess.py
python run_task.py --task ss
python run_task.py --task ss-ke
```
The `preprocess.py` script will generate the data for the sentence similarity task. It's output will be saved in the `data` folder, and it's only required to run once.

To run pipeline for Approach 2, run the following commands:
```shell
cd src
python run_task.py --task ke
```

For both approaches, the models are saved in the `models` folder. 

#### Inference

For inference, we provide a demo app that can be run locally. To run the demo app, run the following commands:
```shell
cd src
streamlit run demo.py
```

We also provide a script to run inference on a single sample. To run inference on a single sample, run the following commands:
```shell
cd src
python inference.py --text "text to extract keywords from"
```

