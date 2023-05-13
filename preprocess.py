import json 
import os
import shutil


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


if __name__ == '__main__':
    with open('data/final_data.json') as f:
        data = json.load(f, ensure_ascii=False)
    dataset = 'temp-keys-dataset'
    final_dataset = 'final-keys-dataset'
    init_dirs()
    generate_data()
    move_data()
    print(f"Finished generating dataset at data/{final_dataset}")