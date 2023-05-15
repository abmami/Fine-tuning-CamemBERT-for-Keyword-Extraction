import warnings
warnings.filterwarnings("ignore")



data_folder = "../data/"

data = {
    "PAWS-C-FR":{
    "task":"ss",
    "train": data_folder + "PAWS-C-FR/translated_train.tsv",
    "dev": data_folder + "PAWS-C-FR/dev_2k.tsv",
    "test": data_folder + "PAWS-C-FR/test_2k.tsv"
    },
    "KEYS-DATASET":{
    "task":"ke",
    "train": data_folder + "KEYS-DATASET/train.csv",
    "dev": data_folder + "KEYS-DATASET/dev.csv",
    "test": data_folder + "KEYS-DATASET/test.csv"
    }
}


models_folder = "../models/"
