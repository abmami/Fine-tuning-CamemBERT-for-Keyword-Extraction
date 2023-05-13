import json 
import os
import shutil
import spacy
import pandas as pd
import numpy as np
import string, re


def init_dirs():
    # Remove folder even if it has files in it

    if os.path.exists('data/{}'.format(dataset)):
        shutil.rmtree('data/{}'.format(dataset)) 
    os.makedirs('data/{}/docsutf8'.format(dataset))
    os.makedirs('data/{}/keys'.format(dataset))

    if os.path.exists(f'data/{final_dataset}'):
        shutil.rmtree(f'data/{final_dataset}')
    os.makedirs(f'data/{final_dataset}/docsutf8')
    os.makedirs(f'data/{final_dataset}/keys')



def generate_data():
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


def move_data():
    for filename in os.listdir('data/WKC/keys'):
        shutil.copy('data/WKC/keys/{}'.format(filename), 'data/{}/keys/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/WKC/docsutf8'.format(dataset)):
        shutil.copy('data/WKC/docsutf8/{}'.format(filename), 'data/{}/docsutf8/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/{}/docsutf8'.format(dataset)):
        shutil.copy('data/{}/docsutf8/{}'.format(dataset, filename), 'data/{}/docsutf8/{}'.format(final_dataset,filename))

    for filename in os.listdir('data/{}/keys'.format(dataset)):
        shutil.copy('data/{}/keys/{}'.format(dataset, filename), 'data/{}/keys/{}'.format(final_dataset,filename))

def save_sets(path):
    """
    Load txt files and their corresponding keywords into a dict then split it into train, validation and test sets
    """
    
    # Load the txt files
    txt_files = sorted(os.listdir(path + "/docsutf8"))
    txt_files = [file for file in txt_files if file.endswith(".txt")]
                 
    # Load the keywords files
    keys_files = sorted(os.listdir(path + "/keys"))
    keys_files = [file for file in keys_files if file.endswith(".key")]

    dataset = {"text": [], "keywords": []}
    # download the spacy fr model using the command "python -m spacy download fr_core_news_sm"
    nlp = spacy.load("fr_core_news_sm")

    for txt_file in txt_files:
        # load the text into one string
        text = open(path + "/docsutf8/" + txt_file, "r", encoding="utf-8").read()

        # preprocess the text
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        text = text.replace("  ", " ")
        # remove the extra spaces
        text = text.strip()
        #text = re.sub(r"['\"]", "", text) # remove single and double quotes

        dataset["text"].append(text)
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
    

    # save the dataset into a csv file
    dataset = pd.DataFrame(dataset)
    dataset.to_csv("data/KEYS-DATASET/full-dataset.csv", index=False)

    # split dataframe into 80% train, 10% validation and 10% test
    train_dataset, val_dataset, test_dataset = np.split(dataset.sample(frac=1, random_state=42), [int(.8*len(dataset)), int(.9*len(dataset))])


    
    train_dataset = pd.DataFrame(train_dataset)
    train_dataset.to_csv("data/KEYS-DATASET/train.csv", index=False)

    val_dataset = pd.DataFrame(val_dataset)
    val_dataset.to_csv("data/KEYS-DATASET/dev.csv", index=False)

    test_dataset = pd.DataFrame(test_dataset)
    test_dataset.to_csv("data/KEYS-DATASET/test.csv", index=False)


if __name__ == '__main__':
    with open('data/final_data.json') as f:
        data = json.load(f)
    dataset = 'temp-keys-dataset'
    final_dataset = 'KEYS-DATASET'
    init_dirs()
    generate_data()
    move_data()
    save_sets('data/{}'.format(final_dataset))

    print(f"Finished generating dataset at data/{final_dataset}")