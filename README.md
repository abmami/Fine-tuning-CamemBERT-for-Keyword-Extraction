# Fine-tuning CamemBERT for Keyword Extraction

This repository is dedicated to fine-tuning CamemBERT for keyword extraction from French text. It provides code for two distinct approaches with different training sequences. 

## Approaches
### Approach 1: Pre-finetuning on Sentence Similarity, followed by Fine-tuning for Keyword Extraction
In this approach, we first pre-finetune CamemBERT on the task of sentence similarity. By training the model to understand the semantic relationships between sentences, it gains a deeper understanding of contextual information. Subsequently, we fine-tune the pre-finetuned model specifically for keyword extraction. This two-step process allows CamemBERT to leverage its enhanced comprehension of language semantics to accurately identify and extract keywords.

### Approach 2: Direct Fine-tuning for Keyword Extraction
In this streamlined approach, we directly fine-tune CamemBERT for the keyword extraction task without pre-finetuning on sentence similarity. By focusing solely on the target task, we optimize the fine-tuning process and enable CamemBERT to learn directly from the keyword extraction data, and just benefit from its language representation capabilities.

## Datasets

The datasets are available in the `data` folder for each approach.

- `KEYS-DATASET`: This folder contains the final dataset used for training and validation. It was collected from transcribed YouTube videos and merged with WikiNews french keywords dataset for data augmentation. The dataset contains ~340 documents.
- `PAWS-C-FR`: This folder contains the PAWS-X french dataset used for pre-fine-tuning CamemBERT on Sentences similiary task. The dataset contains 3 files: `translated_train.tsv`, `test_2k.tsv`, and `dev_2k`. Each file contains 3 columns: `id`, `sentence1`, and `sentence2`. The `sentence1` column contains the first sentence, the `sentence2` column contains the second sentence, and the `label` column contains the label (0 or 1). 

## Results

Table summarizing the results for each approach.
| Approach | Training Sequence | Quantized | Test Accuracy | Test F1 Score |
| --- | --- | --- | --- | --- |
| Approach 1 | SS -> KE | No | 0.98 | - |
| Approach 1 | SS -> KE | Yes | - | - |
| Approach 2 | KE | No | - | - |
| Approach 2 | KE | Yes | - | - |

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
### Training
Run the script with the desired approach and task:

```shell
python train.py --task <task-name>
```
Replace <task-name> with "ss" for sentence similarity, "ke" for keyword extraction, or "ss-ke" to fine-tune the pre-finetuned CamemBERT model on keyword extraction.
The script will load the necessary configurations and data for the specified task and start the fine-tuning process. After fine-tuning, the model will be saved in the respective "models" directory.

### Quantization

Run the quantization script:

```shell
python qunatize.py [--model "model-name"]
```
Optional: Specify the "--model" argument to choose the model to quantize. Available options are "camembert-ss-ke" and "camembert-ke". By default, "camembert-ke" is used. 

The script will load the specified pre-trained model, perform dynamic quantization, and save the quantized model. The quantized model will be saved in the respective model directory with the name "quantized_model.pt".

### Inference
**Using demo app**

We provide a demo app that can be run locally. To launch the demo app, run the following commands:
```shell
streamlit run demo.py
```

**Using script**

We also provide a script to run inference on a single sample. To run inference on a single sample, run the following commands:
```shell
python inference.py --text "input-text" [--model "model-name"]
```
Replace "input-text" with the text for which you want to extract keywords.
Optional: Specify the "--model" argument to choose the model for inference. Available options are "camembert-ss-ke" and "camembert-ke". By default, "camembert-ke" is used.

The script will perform keyword extraction on the provided text using the specified model and display the extracted keywords.