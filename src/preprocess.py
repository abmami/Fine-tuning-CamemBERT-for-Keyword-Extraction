import json 
import os
import shutil
import spacy
import pandas as pd
import numpy as np
import string
from keybert import KeyBERT
import argparse
import time


def generate_files(data, dataset_path):
    for playlist_id, playlist_data in data.items():
        for video_id, video_data in playlist_data.items():
            transcript_sentences = video_data['transcript']
            
            # concat all sentences in one string
            transcript = ' '.join([sentence['text'] for sentence in transcript_sentences.values()])
            # get keywords
            keywords = video_data['tags']
            categories = video_data['categories']
            chapters = video_data['chapters']
            
            # get chapter titles, concat all titles in one string seperated by new line each
            if chapters:
                chapter_titles = '\n'.join([chapter['title'] for chapter in chapters])
            else:
                chapter_titles = ''

            title = video_data['title'].split(' ')

            # add title words to keywords
            keywords.extend(title)
            
            # concat all keywords and categories in one string seperated by new line each 
            keywords = '\n'.join(keywords)
            categories = '\n'.join(categories)

            # combine keywords, chapter_titles, categories in one string
            keywords_categories = keywords + '\n' + categories + '\n' + chapter_titles
            video_d = {'transcript': transcript, 'keywords': keywords_categories}

            # save transcript and keywords in text and keys folders
            with open('{}/docsutf8/{}.txt'.format(dataset_path, video_id), 'w') as f:
                f.write(video_d['transcript'])
            
            with open('{}/keys/{}.key'.format(dataset_path, video_id), 'w') as f:
                f.write(video_d['keywords'])

    print('Finished generating files')


def save_sets(path, use_keybert=False,split=True):

    # Load the txt files
    txt_files = sorted(os.listdir(path + "/docsutf8"))
    txt_files = [file for file in txt_files if file.endswith(".txt")]
    dataset = {"text": [], "keywords": []}
    # download the spacy fr model using the command "python -m spacy download fr_core_news_sm"
    nlp = spacy.load("fr_core_news_sm")

    for txt_file in txt_files:
        # load the text into one string
        text = open(path + "/docsutf8/" + txt_file, "r", encoding="utf-8").read()

        # preprocess the text
        text = text.replace("\n", "").replace("\t", "").replace("  ", " ").strip()

        # Chunk the text into 256 characters chunks for keyBERT
        if use_keybert:
            chunk_size = 256
            if len(text) > chunk_size:
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                for chunk in chunks:
                    dataset["text"].append(chunk)
                    dataset["keywords"].append([])
            else:
                dataset["text"].append(text)
                dataset["keywords"].append([])
        else:
            dataset["text"].append(text)
        
        if not use_keybert:
            # Load the keywords files
            keys_files = sorted(os.listdir(path + "/keys"))
            keys_files = [file for file in keys_files if file.endswith(".key")]
            keywords = open(path + "/keys/" + txt_file[:-4] + ".key", "r", encoding="utf-8").read().split("\n")
            # preprocess the keywords
            # remove the empty keywords
            keywords = [keyword for keyword in keywords if keyword != ""]
            # remove the extra spaces
            keywords = [keyword.strip() for keyword in keywords]
            # remove the duplicates
            keywords = list(set(keywords))
            # load spacy fr model and remove the stopwords
            keywords = [keyword for keyword in keywords if keyword not in nlp.Defaults.stop_words]
            # remove punctuation using string.punctuation
            keywords = [keyword for keyword in keywords if keyword not in string.punctuation]
            
            dataset["keywords"].append(keywords)

    dataset = pd.DataFrame(dataset)

    # Sometimes the keywords are not relevant, so we use keyBERT to extract the keywords
    if use_keybert:

        # extract the keywords for french text, get the top 5 keywords
        model = KeyBERT("Geotrend/distilbert-base-en-fr-cased")
        pre_dataset = dataset.copy()

        # remove stop words and punctuation before extracting the keywords
        pre_dataset["text"] = pre_dataset["text"].apply(lambda x: " ".join([word for word in x.split() if word not in nlp.Defaults.stop_words]))
        pre_dataset["text"] = pre_dataset["text"].apply(lambda x: " ".join([word for word in x.split() if word not in string.punctuation]))
        pre_dataset["keywords"] = pre_dataset["text"].apply(lambda x: [x[0] for x in model.extract_keywords(x, keyphrase_ngram_range=(1, 1), 
                                  top_n=5, min_df=1, diversity=0.9, stop_words=None)])
        dataset = pre_dataset.copy()
    
    dataset.to_csv(os.path.join(path, "full-dataset.csv"), index=False)

    # Generate the train, validation and test sets
    if split:
        train_dataset, val_dataset, test_dataset = np.split(dataset.sample(frac=1, random_state=42), [int(.8*len(dataset)), int(.9*len(dataset))])
        pd.DataFrame(train_dataset).to_csv(os.path.join(path, "train.csv"), index=False)
        pd.DataFrame(val_dataset).to_csv(os.path.join(path, "val.csv"), index=False)
        pd.DataFrame(test_dataset).to_csv(os.path.join(path, "test.csv"), index=False)

        print("Train set: {} samples".format(len(train_dataset)))
        print("Validation set: {} samples".format(len(val_dataset)))
        print("Test set: {} samples".format(len(test_dataset)))


def main():

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--use_keybert', type=bool, default=False, help='use keyBERT to extract keywords')
    args = parser.parse_args()
    use_keybert = args.use_keybert
    start = time.time()
    print("Generating datasets...")

    # Change directory to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    # Load config file
    with open('config.json') as f:
        config = json.load(f)

    # Load config params
    data_path = config['data']['data_folder']
    KE_dataset_name = config['data']['KE_DATASET']['name']

    # Initialize the directories
    dataset_path = os.path.join(data_path, KE_dataset_name)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'docsutf8'))
    os.makedirs(os.path.join(dataset_path, 'keys'))
    print("The dataset will be generated in {}".format(dataset_path))
          
    # Set this to True if you want to use the WKC dataset
    use_WKC = True 
    if use_WKC:
        # Copy files in docsutf8 of WKC to docsutf8 of dataset_path
        for file in os.listdir(os.path.join(data_path, 'WKC/docsutf8')):
            shutil.copy(os.path.join(data_path, 'WKC/docsutf8', file), os.path.join(dataset_path, 'docsutf8', file))
        
        # Copy files in keys of WKC to keys of dataset_path
        for file in os.listdir(os.path.join(data_path, 'WKC/keys')):
            shutil.copy(os.path.join(data_path, 'WKC/keys', file), os.path.join(dataset_path, 'keys', file))
        
        print("Copied files from WKC to", dataset_path)

    # Load our custom dataset
    with open(os.path.join(data_path, 'final_data.json')) as f:
        data = json.load(f)
    
    # Generate the data
    generate_files(data, dataset_path)

    # Preprocess and save the data
    save_sets(dataset_path, use_keybert=use_keybert)

    print("Finished generating dataset at", dataset_path, "in", str(time.time() - start)[:5], "seconds")


if __name__ == '__main__':
    main()

  