import torch, os, json
from keyword_extraction import model as ke_model
import argparse
import os


def quantize(model_name):

    # Load the saved model
    wdir = "keyword_extraction"
    with open(os.path.join(wdir, "config.json")) as f:
        config = json.load(f)

    model_dir = os.path.join(wdir, "models", model_name + "/" )
    device = torch.device("cuda")
    config['model_name'] = model_name
    loaded_model = ke_model.KELightningModel(config, from_scratch=False, inference=True)
    loaded_state_dict = torch.load(model_dir + "/model.pt",map_location=device)
    loaded_model.load_state_dict(loaded_state_dict)
    loaded_model.to('cuda')
    
    # Quantize model using dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        loaded_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    save_to = model_dir + "/quantized_model.pt"
    torch.save(quantized_model.state_dict(), save_to)
    
    print('Quantized model saved to', save_to)
    print('Original | Size (MB):', os.path.getsize(model_dir + "/model.pt")/1e6)
    print('Quantized | Size (MB):', os.path.getsize(save_to)/1e6)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="camembert-ke", help="model to quantize")
    args = parser.parse_args()
    model = args.model

    if model not in ["camembert-ss-ke", "camembert-ke"]:
        raise ValueError("Unknown model name. Choose from camembert-ss-ke and camembert-ke")    
    
    quantize(model)