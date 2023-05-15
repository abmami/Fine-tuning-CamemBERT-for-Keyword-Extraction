import json 
import os
import shutil
import spacy
import pandas as pd
import numpy as np
import string, re
from keybert import KeyBERT
import argparse
import time

def init_dirs(temp_dataset,final_dataset):
    if os.path.exists('data/{}'.format(temp_dataset)):
        shutil.rmtree('data/{}'.format(temp_dataset)) 
    os.makedirs('data/{}/docsutf8'.format(temp_dataset))
    os.makedirs('data/{}/keys'.format(temp_dataset))

    if os.path.exists(f'data/{final_dataset}'):
        shutil.rmtree(f'data/{final_dataset}')
    os.makedirs(f'data/{final_dataset}/docsutf8')
    os.makedirs(f'data/{final_dataset}/keys')


def generate_data_files(data, dataset, final_dataset):
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

            with open('data/{}/docsutf8/{}.txt'.format(dataset, video_id), 'w') as f:
                f.write(video_d['transcript'])
            
            with open('data/{}/keys/{}.key'.format(dataset, video_id), 'w') as f:
                f.write(video_d['keywords'])

    # copy the files to the final dataset folder
    for filename in os.listdir('data/WKC/keys'):
        shutil.copy('data/WKC/keys/{}'.format(filename), 'data/{}/keys/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/WKC/docsutf8'.format(dataset)):
        shutil.copy('data/WKC/docsutf8/{}'.format(filename), 'data/{}/docsutf8/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/{}/docsutf8'.format(dataset)):
        shutil.copy('data/{}/docsutf8/{}'.format(dataset, filename), 'data/{}/docsutf8/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/{}/keys'.format(dataset)):
        shutil.copy('data/{}/keys/{}'.format(dataset, filename), 'data/{}/keys/{}'.format(final_dataset,filename))



def save_sets(path, use_keybert=False):

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
    print(len(dataset))

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
        
    pre_dataset.to_csv("data/KEYS-DATASET/full-dataset.csv", index=False)
    train_dataset, val_dataset, test_dataset = np.split(pre_dataset.sample(frac=1, random_state=42), [int(.8*len(pre_dataset)), int(.9*len(pre_dataset))])

    pd.DataFrame(train_dataset).to_csv("data/KEYS-DATASET/train.csv", index=False)
    pd.DataFrame(val_dataset).to_csv("data/KEYS-DATASET/dev.csv", index=False)
    pd.DataFrame(test_dataset).to_csv("data/KEYS-DATASET/test.csv", index=False)


def main():
    start = time.time()
    print("Generating dataset...")
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--use_keybert', type=bool, default=False, help='use keyBERT to extract keywords')
    args = parser.parse_args()
    use_keybert = args.use_keybert

    # Load the data
    with open('data/final_data.json') as f:
        data = json.load(f)
    
    temp_dataset = 'temp-keys-dataset'
    final_dataset = 'KEYS-DATASET'
    
    # Initialize the directories
    init_dirs(temp_dataset,final_dataset)

    # Generate the data
    generate_data_files(data, temp_dataset, final_dataset)

    # Preprocess and save the data
    save_sets('data/{}'.format(final_dataset), use_keybert=use_keybert)

    print(f"Finished generating dataset at data/{final_dataset}")
    print(f"Time taken: {time.time() - start} seconds")


if __name__ == '__main__':
    main()

  