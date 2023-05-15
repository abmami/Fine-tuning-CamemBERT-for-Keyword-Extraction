"""
# Load the saved model
loaded_model = LightningModel("camembert-base", 2, lr=lr, weight_decay=weight_decay)
loaded_state_dict = torch.load("models/camembert-prefinetuned.pt")
loaded_model.load_state_dict(loaded_state_dict)
loaded_model.to('cuda')
loaded_model.eval()

# Quantize model using dynamic quantization

quantized_model = torch.quantization.quantize_dynamic(
    loaded_model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), 'models/camembert-prefinetuned-quantized.pt')


import os
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(loaded_model)
print_size_of_model(quantized_model)


# Load the saved model
device = torch.device("cuda")
loaded_quantized_model = LightningModel("camembert-base", 2, lr=lr, weight_decay=weight_decay)
loaded_quantized_state_dict = torch.load("models/camembert-prefinetuned-quantized.pt",map_location=device)
loaded_quantized_model.load_state_dict(loaded_quantized_state_dict, strict=False)
loaded_quantized_model.to('cuda')


# Set the model to evaluation mode
loaded_quantized_model.eval()
eval(loaded_quantized_model,camembert_trainer,test_dataloader)

"""