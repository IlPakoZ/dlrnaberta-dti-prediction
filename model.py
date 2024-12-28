import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# The key is the drug because it binds to the target (like the key of a lock)

class InteractionModelATTN(nn.Module):
    def __init__(self, target_encoder, drug_encoder, dropout, num_heads=1):
        super().__init__()
        self.target_encoder = target_encoder
        self.drug_encoder = drug_encoder
        self.multihead_attention = nn.MultiheadAttention(768, num_heads, dropout = dropout, batch_first=True)
        self.conv = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(768)                          # For attention output
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(1)
        self.process = nn.Linear(768, 768)
        self.output = nn.Linear(768, 1)

        
    def forward(self, x1, x2):
        y1 = self.target_encoder(**x1).hidden_states[-1]       # The target
        y2 = self.drug_encoder(**x2).hidden_states[-1]         # The drug

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
        #input1["input_ids"] = input1["input_ids"].view(-1)
        #input2["input_ids"] = input2["input_ids"].view(-1)
        input1 = {"input_ids":self.targets["input_ids"][idx], "attention_mask": self.targets["attention_mask"][idx]}
        input2 = {"input_ids":self.smiles["input_ids"][idx], "attention_mask": self.smiles["attention_mask"][idx]}
        
        target = None
        if not self.pkd is None:
            target = self.pkd[idx]

        return ((input1, input2), target)

