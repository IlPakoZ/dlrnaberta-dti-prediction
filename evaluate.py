import torch

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
    """
    Runs prediction for a batch of target-drug pairs using the given model.

    Args:
        model (nn.Module): The trained interaction model used for prediction.
        targets (Iterable[dict]): An iterable of dictionaries containing inputs for the target encoder.
        drugs (Iterable[dict]): An iterable of dictionaries containing inputs for the drug encoder.
        device (str): The device to perform computation on.
            Default: ``cpu``


    Yields:
        float: The predicted binding affinity (unscaled) for each target-drug pair.
    """
    with torch.no_grad():
        for target, drug in zip(targets, drugs):
            target = target.to(device)
            
            drug = drug.to(device)
            output = model(target, drug)
            output = model.unscale(output)
            yield output.cpu().numpy()[0][0]

def interp_predict(model, targets, drugs, device="cpu"):
    """
    Runs prediction in interpretation mode, returning the final summation layer value.

    Args:
        model (nn.Module): The interaction model set in interpretation mode (`INTERPR_MODE = True`).
        targets (Iterable[dict]): An iterable of dictionaries containing inputs for the target encoder.
        drugs (Iterable[dict]): An iterable of dictionaries containing inputs for the drug encoder.
        device (str): The device to perform computation on.
            Default: ``cpu``

    Yields:
        Tuple[dict, dict, torch.Tensor, torch.Tensor]: A tuple containing:
            - The target input dictionary (moved to CPU),
            - The drug input dictionary (moved to CPU),
            - The predicted binding affinity (unscaled, shape (1,)),
            - The values of the final summation layer before summation (for interpretability).
    """
    with torch.no_grad():
        for target, drug in zip(targets, drugs):
            target = target.to(device)
            
            drug = drug.to(device)
            output = model(target, drug)
            last_layer_values = model.model.presum_layer

            output = model.unscale(output)
            yield (target.to("cpu"), drug.to("cpu"), output.cpu(), last_layer_values.cpu())
