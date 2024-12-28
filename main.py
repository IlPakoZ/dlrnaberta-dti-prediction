import train
import pandas as pd
import model 
from transformers import AutoTokenizer, RobertaTokenizerFast
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Next thing to do: 
    # StratifiedShuffleSplit (stratify by category)
    # During the final training, only then select sets such as sequences in training and validation sets don't overlap
    train_parameters = {"train_batch_size": 4,
                        "device": "cuda",
                        "learning_rate": 3e-4,
                        "adam_epsilon": 1e-6,
                        "gradient_accumulation_steps": 16,
                        #"num_training_steps":3000,
                        "num_epochs":4,
                        "log_performance_every":5,
                        "weight_decay": 0.01,
                        "model_dropout": 0.2,
                        "lora_r": 16,
                        "lora_alpha":16,
                        "lora_dropout":0.1,
                        "max_norm":1,
                        #plot_grads":True,
                        "validate_while_training":True
                        }
    
    drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    target_tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer')

    accelerator, finetune_model, target_tokenizer, drug_tokenizer = train.create_model(train_parameters,  6)

    # Try with smaller models
    inters = pd.read_csv("processed/interactions/all.csv")
    X = inters[["SMILES", "Target_RNA_sequence"]]
    y = inters['pKd'].values
    classes = inters["Category"].values

    skf = StratifiedKFold(n_splits=8)
    for i, (train_index, val_index) in enumerate(skf.split(X, classes)):
        print(f"Fold {i}:")
        train_X = inters.iloc[train_index]
        val_X = inters.iloc[val_index]
        train_y = y[train_index]
        val_y = y[val_index]

        smiles = drug_tokenizer(train_X["SMILES"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
        targets = target_tokenizer(train_X["Target_RNA_sequence"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
        
        scaler = StandardScaler()

        train_pkd = scaler.fit_transform(train_y.reshape(-1,1)).astype(np.float32)
        train_dataset = model.InterDataset(targets, smiles, train_pkd)

        smiles = drug_tokenizer(val_X["SMILES"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
    
        targets = target_tokenizer(val_X["Target_RNA_sequence"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")

        val_pkd = scaler.transform(val_y.reshape(-1,1)).astype(np.float32)
        val_dataset = model.InterDataset(targets, smiles, val_pkd)

        print(len(val_dataset))

        # Define a common x-axis range based on your data
        common_range = (-4, 4)

        # Plotting overlapping histograms with proportions
        plt.figure(figsize=(8, 6))
        plt.hist(train_pkd, bins=30, range=common_range, color='blue', alpha=0.5, edgecolor='black', label='train_pkd', density=True)
        plt.hist(val_pkd, bins=30, range=common_range, color='red', alpha=0.5, edgecolor='black', label='val_pkd', density=True)

        # Adding titles and labels
        plt.title('Overlapping Distributions of train_pkd and val_pkd')
        plt.xlabel('Scaled Values')
        plt.ylabel('Proportion')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()


        # Scaler should be passed to the methods and used to rescale the values back to normal during metric calculation
        # Sklearn scaler is not compatible with cuda, implement it by yourself
        # TODO: Implement the scalar
        train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, None)