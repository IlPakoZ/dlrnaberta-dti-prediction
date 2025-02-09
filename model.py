import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, Trainer, get_cosine_schedule_with_warmup
from mup import MuAdamW
from dataclasses import dataclass, field
from transformers.trainer import IS_SAGEMAKER_MP_POST_1_10

# The key is the drug because it binds to the target (like the key of a lock)
class InteractionModelATTNConfig(PretrainedConfig):
    def __init__(self, dropout = 0.2, num_heads = 1, **kwargs,):
        self.num_heads = num_heads
        self.dropout = dropout
        super().__init__(**kwargs)

class InteractionModelATTNForRegression(PreTrainedModel):
    config_class = InteractionModelATTNConfig

    def __init__(self, config, target_encoder, drug_encoder):
        super().__init__(config)
        self.model = InteractionModelATTN(target_encoder, 
                                          drug_encoder, 
                                          config.dropout, 
                                          config.num_heads)

    def forward(self, x1, x2):
        return self.model(x1, x2)

class InteractionModelATTN(nn.Module):
    def __init__(self, target_encoder, drug_encoder, dropout, num_heads=1):
        super().__init__()
        self.target_encoder = target_encoder
        self.drug_encoder = drug_encoder
        self.lin_map = nn.Linear(512, 384)
        self.dropout_map = nn.Dropout(dropout)

        self.multihead_attention = nn.MultiheadAttention(384, num_heads, dropout = dropout, batch_first=True)
        self.conv = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(384)                          # For attention output
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(1)
        self.process = nn.Linear(384, 384)

        self.output = nn.Linear(384, 1)

        
    def forward(self, x1, x2):
        y1 = self.target_encoder(**x1).hidden_states[-1]       # The target
        y2 = self.drug_encoder(**x2).hidden_states[-1]         # The drug

        y1 = self.lin_map(y1)
        y1 = self.dropout_map(y1)

        key_padding_mask = torch.squeeze(x2["attention_mask"], axis=1).bool() # This doesn't work in multihead_attention
        out, _ = self.multihead_attention(y1, y2, y1, key_padding_mask=torch.squeeze(x2["attention_mask"].float(), axis=1))
        out = self.norm1(self.dropout(out))

        out = out.masked_fill(key_padding_mask.unsqueeze(2), 0) # Padding elements values should contribute zero to the sum
        
        scales = key_padding_mask.sum(dim=-1).unsqueeze(1)      # Number of non-padded elements
        out = self.conv(out).squeeze(1)
        out = out/scales                                        # Divide the output by the number of non-padded elements

        out = self.batch_norm(out.unsqueeze(1)).squeeze(1)
        out = self.gelu(out)
        out = self.gelu(self.process(out))
        out = self.output(out)

        return out
    
class MuTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = MuAdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
                
        #smp.state.cfg.fp16
        if IS_SAGEMAKER_MP_POST_1_10 and False:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio*num_training_steps, num_training_steps=num_training_steps, num_cycles=0.41)

class InterDataset(Dataset):
    def __init__(self, targets, smiles, pkd=None):
        self.targets = targets
        self.smiles = smiles
        self.pkd = pkd

    def __len__(self):
        return len(self.targets["input_ids"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input1, input2 = self.targets[idx], self.smiles[idx]

        input1 = {"input_ids":self.targets["input_ids"][idx], "attention_mask": self.targets["attention_mask"][idx]}
        input2 = {"input_ids":self.smiles["input_ids"][idx], "attention_mask": self.smiles["attention_mask"][idx]}
        
        target = None
        if not self.pkd is None:
            target = self.pkd[idx]

        return ((input1, input2), target)

    def save(self, directory):
        with open(directory+'dataset.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(directory):
        with open(directory+'dataset.pkl', 'rb') as f:
            return pickle.load(f)

class StdScaler():
    def fit(self, X):
        self.mean_ = torch.mean(X).item()
        self.std_ = torch.std(X, correction=0).item()

    def fit_transform(self, X):
        self.mean_ = torch.mean(X).item()
        self.std_ = torch.std(X, correction=0).item()

        return (X-self.mean_)/self.std_
    
    def transform(self, X):
        return (X-self.mean_)/self.std_

    def inverse_transform(self, X):
        return (X*self.std_)+self.mean_
    
    def save(self, directory):
        with open(directory+"/scaler.config", "w") as f:
            f.write(str(self.mean_)+"\n")
            f.write(str(self.std_)+"\n")

    def load(self, directory):
        with open(directory+"/scaler.config", "r") as f:
            self.mean_ = float(f.readline())
            self.std_ = float(f.readline())

@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

