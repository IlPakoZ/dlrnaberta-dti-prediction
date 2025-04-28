import torch
import train

def tokenize_inputs(target, drug, target_tokenizer, drug_tokenizer, only_drug = False, only_target = False):
    if only_drug:
        target = None
    else:
        target = target.strip()
        target = target_tokenizer(target, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    
    if only_target:
        drug = None
    else:
        drug = drug.strip()
        drug = drug_tokenizer(drug, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    
    return target, drug


def predict(model, targets, drugs, device="cpu"):
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for target, drug in zip(targets, drugs):
            target = target.to(device)
            
            drug = drug.to(device)
            output = model(target, drug)
            output = model.unscale(output)
            yield output.cpu().numpy()
