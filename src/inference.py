import argparse
import argparse
import os, json
from keyword_extraction import model as ke_model
from sentence_similiarity import model as ss_model

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_dir)

def run_inference_ss(sentences):

    with open("config.json") as f:
        config = json.load(f)
    cfg = config['models']['ss']
    cfg.update({"models_folder": config['models']['models_folder'], "num_labels": 2, "model_name": "camembert-ss"})
    from_scratch = False # for inference we load local model
    model = ss_model.SSLightningModel(cfg, from_scratch=from_scratch)
    model.to('cuda')
    model.eval()
    return model.inference(sentence_1=sentences[0], sentence_2=sentences[1])

def run_inference_ke(text):

    with open("config.json") as f:
        config = json.load(f)

    from_scratch = False # for inference we load local model
    cfg = config['models']['ke']
    cfg.update({"models_folder": config['models']['models_folder'], "num_labels": 3, "model_name": "camembert-ke"})
    model = ke_model.KELightningModel(cfg, from_scratch)
    model.eval()
    return extract_keywords(text, model)

def extract_keywords(text, model):
    # use infer method to get the labels
    labels = model.infer(text)[0]
    # get the tokens from the text
    print(model.config.max_length)
    tokens = model.tokenizer.tokenize(text,truncation=True, padding=True, max_length=256)
    # get the keywords from the tokens and labels
    keywords = []
    for i in range(len(tokens)):
        if labels[i] == 1:
            keyword = tokens[i]
            for j in range(i+1, len(tokens)):
                if labels[j] == 2:
                    keyword += " " + tokens[j]
                else:
                    break
            # convert the keyword to the original string
            keyword = model.tokenizer.convert_tokens_to_string(model.tokenizer.tokenize(keyword))
            keywords.append(keyword)

    return keywords


#text2 = "Je vous avais préparé encore un autre exercice pour s'habituer un petit peu avec les techniques d'enregistrement comptable des opérations commerciales. Vous avez ici un bilan initial, construction 100 mèles euros, client 15 mèles euros, banque 13 mèles euros, caisse 2 mèles euros, au total 130 mèles euros, capital 100 mèles euros, fournisseur 30 mèles euros, total la même chose. Voilà, on a ici un bilan initial, tout simple, et on va essayer de voir un petit peu ce qui est demandé dans l'exercice. Donc ici vous avez les opérations, les opérations que l'entreprise a effectuées, et tout en bas on a un travail à faire. Le travail à faire c'est quoi ? On va effectuer l'ouverture des comptes, on va enregistrer les opérations dans les comptes en T, et puis on va calculer le résultat de l'entreprise, on va établir la balance définitive, et au final établir le bilan final. J'espère bien arriver à un bilan final équilibré. J'essaierai de me concentrer pour que je ne fasse pas d'erreurs, parce que dans ce genre d'exercice, une simple erreur quelque part peut aboutir à un bilan final déséquilibré, c'est-à-dire total actif et différent de total passif."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a model on a task')
    parser.add_argument('--task', type=str, default="ke",help='task name: ss or ke')
    parser.add_argument('--text', type=str, default="Hello",help='text to extract keywords from')
    parser.add_argument('--sentence1', type=str, default="Le chat est sur le tapis.", help='sentence 1 to run inference on')
    parser.add_argument('--sentence2', type=str, default="Le chien est sur le canapé.", help='sentence 2 to run inference on')
        
    args = parser.parse_args()
    task = args.task

    if task == "ss":
        sentences = [args.sentence1, args.sentence2]
        print(run_inference_ss(sentences=sentences))
    elif task == "ke":
        text = args.text
        print(run_inference_ke(text=text))



