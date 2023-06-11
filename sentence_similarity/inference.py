import argparse
import argparse
import os, json
from model import SSLightningModel

def run_inference(sentences):
    cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    config['model_name'] = "camembert-ss"
    from_scratch = False
    model = SSLightningModel(config, from_scratch=from_scratch)
    model.to('cuda')
    model.eval()
    return model.inference(sentence_1=sentences[0], sentence_2=sentences[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', type=str, default="Le chat est sur la table.", help='sentence 1 to run inference on')
    parser.add_argument('--s2', type=str, default="Le chien est sur le tapis.", help='sentence 2 to run inference on')
    args = parser.parse_args()
    sentences = [args.s1, args.s2]
    result = run_inference(sentences=sentences)
    output = "Sentence 1: {}\nSentence 2: {}\nSimilarity: {}".format(sentences[0], sentences[1], result)
    print(output)
