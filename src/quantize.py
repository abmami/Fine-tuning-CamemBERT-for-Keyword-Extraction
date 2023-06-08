import torch, os, json
from keyword_extraction import model as ke_model
import argparse
import os


def quantize(config, task):

    # Load the saved model
    model_name = 'camembert-' + task
    path_to_model = config['models']["models_folder"] + model_name
    device = torch.device("cuda")
    cfg = config['models']['ke']
    cfg.update({"models_folder": config['models']['models_folder'], "num_labels": 3, "model_name": model_name})
    loaded_model = ke_model.KELightningModel(cfg, from_scratch=False)
    loaded_state_dict = torch.load(path_to_model + "/model.pt",map_location=device)
    loaded_model.load_state_dict(loaded_state_dict)
    loaded_model.to('cuda')
    
    # Quantize model using dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        loaded_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    save_to = path_to_model + "/quantized_model.pt"
    torch.save(quantized_model.state_dict(), save_to)
    
    print('Quantized model saved to', save_to)
    print('Original | Size (MB):', os.path.getsize(path_to_model + "/model.pt")/1e6)
    print('Quantized | Size (MB):', os.path.getsize(save_to)/1e6)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ke", help="Task to quantize")
    args = parser.parse_args()
    task = args.task

    if task not in ["ss-ke", "ke"]:
        raise ValueError("Unknown task name")
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.chdir(root_dir)
    
    with open("config.json") as f:
        config = json.load(f)
    
    quantize(config,task)