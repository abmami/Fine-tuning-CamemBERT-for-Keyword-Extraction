

data = {
    "data_folder":"../data/",
    "PAWS-C-FR":{
    "task":"ss",
    "train":"PAWS-C-FR/translated_train.tsv",
    "dev": "PAWS-C-FR/dev_2k.tsv",
    "test": "PAWS-C-FR/test_2k.tsv"
    },
    "KEYS-DATASET":{
    "task":"ke",
    "train": "KEYS-DATASET/train.csv",
    "dev": "KEYS-DATASET/dev.csv",
    "test": "KEYS-DATASET/test.csv"
    }
}

models = { 
    "models_folder":"../models/",
    "ss":{
        "model_name":"camembert-base",
        "epochs":1,
        "lr":  3e-5,
        "batch_size": 16,
        "accumulate_grad_batches":8,
        "max_length": 128,      
        "weight_decay": 0.,
    },

    "ke": {
        "model_name":"camembert-base",
        "epochs":3,
        "lr":  2e-5,
        "batch_size": 4,
        "accumulate_grad_batches":8,
        "max_length": 256,
        "eps": 1e-08, 
        "betas": (0.9, 0.999),
    }
}