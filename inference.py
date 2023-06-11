import argparse
from keyword_extraction import inference
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='text to extract keywords from')
    parser.add_argument('--model', type=str, required=False, default='camembert-ke', help='model name to use for inference')
    args = parser.parse_args()
    text = args.text
    model_name = args.model
    if model_name not in ["camembert-ss-ke", "camembert-ke"]:
        raise ValueError("Unknown model name. Choose from camembert-ss-ke and camembert-ke")
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.chdir(root_dir)
    print(inference.run_inference(text=text, model_name=model_name))