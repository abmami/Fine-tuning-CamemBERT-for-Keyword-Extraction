# Fine-tuning CamemBERT for Keyword Extraction

This repository is dedicated to fine-tuning CamemBERT for keyword extraction. It provides code and resources for two distinct approaches to enhance the performance of CamemBERT in extracting keywords from French text.

## Approaches
### Approach 1: Pre-finetuning on Sentence Similarity, followed by Fine-tuning for Keyword Extraction
In this approach, we first pre-finetune CamemBERT on the task of sentence similarity. By training the model to understand the semantic relationships between sentences, it gains a deeper understanding of contextual information. Subsequently, we fine-tune the pre-finetuned model specifically for keyword extraction. This two-step process allows CamemBERT to leverage its enhanced comprehension of language semantics to accurately identify and extract keywords.

### Approach 2: Direct Fine-tuning for Keyword Extraction
In this streamlined approach, we directly fine-tune CamemBERT for the keyword extraction task without pre-finetuning on sentence similarity. By focusing solely on the target task, we optimize the fine-tuning process and enable CamemBERT to learn directly from the keyword extraction data, and just benefit from its language representation capabilities.

## Datasets

## Training

## Results

## Run Locally

### Requirements
### Usage

