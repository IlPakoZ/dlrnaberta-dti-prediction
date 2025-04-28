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
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import train, math
from math import sqrt
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding

class ChembertaTokenizer:
    def __init__(self, vocab_file):
        self.tokenizer = Tokenizer(
            WordLevel.from_file(
                vocab_file, 
                unk_token='[UNK]'
        ))
        self.tokenizer.pre_tokenizer = Split(
            pattern=Regex(r"\[(.*?)\]|Cl|Br|>>|\\|.*?"),
            behavior='isolated'
        )
        # Disable padding
        
        self.tokenizer.encode_special_tokens = True
        self.special_token_ids = {
            self.tokenizer.token_to_id('[CLS]'),
            self.tokenizer.token_to_id('[SEP]'),
            self.tokenizer.token_to_id('[PAD]'),
            self.tokenizer.token_to_id('[UNK]')  
        }

        self.tokenizer.post_processor = TemplateProcessing(
            single='[CLS] $A [SEP]',
            pair='[CLS] $A [SEP] $B:1 [SEP]:1',
            special_tokens=[
                ('[CLS]', self.tokenizer.token_to_id('[CLS]')),
                ('[SEP]', self.tokenizer.token_to_id('[SEP]'))
            ]
        )
    def __call__(self, inputs, padding=None, truncation=False,
                 max_length=None, return_tensors=None):
        # Configure padding/truncation
        if padding:
            self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('[PAD]'),
                                          pad_token='[PAD]', length=max_length)
        else:
            self.tokenizer.no_padding()

        if truncation:
            self.tokenizer.enable_truncation(max_length=max_length)
        else:
            self.tokenizer.no_truncation()
        if return_tensors == 'pt':
            tensor_type = 'pt'
        else:
            tensor_type = None
        # Handle batch or single input
        if isinstance(inputs, list):
            enc = self.tokenizer.encode_batch(inputs)
            data = {
                "input_ids": [e.ids for e in enc],
                "attention_mask": [e.attention_mask for e in enc]
            }
            return BatchEncoding(data=data, encoding=enc, tensor_type=tensor_type)

        else:
            # Single sequence: wrap into batch of size 1
            enc = [self.tokenizer.encode(inputs)]
            data = {
                "input_ids": [e.ids for e in enc],
                "attention_mask": [e.attention_mask for e in enc]
            }
            return BatchEncoding(data=data, encoding=enc, tensor_type=tensor_type)

        
    def decode(self, ids, skip_special_tokens=False):
        def _decode_sequence(seq):
            if skip_special_tokens:
                seq = [idx for idx in seq if idx not in self.special_token_ids]
            return ''.join(self.tokenizer.id_to_token(idx) for idx in seq)

        # 1) batch: list of lists or torch tensor
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            if len(ids) == 1:
                ids = ids[0]
            
        if isinstance(ids, (list)) and len(ids) > 0 and isinstance(ids[0], (list)):
            return [_decode_sequence(seq) for seq in ids]

        # 2) single sequence: list of ints or torch tensor
        if isinstance(ids, (list)):
            return _decode_sequence(ids)
        
        # 3) single int
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        
# The key is the drug because it binds to the target (like the key of a lock)
class InteractionModelATTNConfig(PretrainedConfig):
    def __init__(self, attention_dropout = 0.2, hidden_dropout = 0.2, num_heads = 1, **kwargs,):
        self.num_heads = num_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        super().__init__(**kwargs)

class InteractionModelATTNForRegression(PreTrainedModel):
    config_class = InteractionModelATTNConfig

    def __init__(self, config, target_encoder, drug_encoder, scaler):
        super().__init__(config)
        self.model = InteractionModelATTN(target_encoder, 
                                          drug_encoder,
                                          scaler,
                                          config.attention_dropout,
                                          config.hidden_dropout,
                                          config.num_heads)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def unscale(self, x):
        return self.model.unscale(x)
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, hidden_dropout=0.0, add_bias_kv=False, **factory_kwargs):
        """
        Initializes the CrossAttention layer.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim ** -0.5

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        # Linear projections for query, key, and value.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)

        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        constant_(self.q_proj.bias, 0.)
        constant_(self.k_proj.bias, 0.)
        constant_(self.v_proj.bias, 0.)

        # Output projection.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        constant_(self.out_proj.bias, 0)

        self.drop_out = nn.Dropout(hidden_dropout)
     
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for cross attention.

        Args:
            query (Tensor): Query embeddings of shape (batch_size, query_len, embed_dim).
            key (Tensor): Key embeddings of shape (batch_size, key_len, embed_dim).
            value (Tensor): Value embeddings of shape (batch_size, key_len, embed_dim).
            attn_mask (Tensor, optional): Attention mask of shape (batch_size, num_heads, query_len, key_len).

        Returns:
            output (Tensor): The attended output of shape (batch_size, query_len, embed_dim).
            attn_weights (Tensor): The attention weights of shape (batch_size, num_heads, query_len, key_len).
        """

        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(batch_size, self.num_heads, query_len, self.head_dim)
        K = K.view(batch_size, self.num_heads, key_len, self.head_dim)
        V = V.view(batch_size, self.num_heads, key_len, self.head_dim)
        
            # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, query_len, key_len)

        if key_padding_mask is not None:
            # Convert boolean mask (False -> -inf, True -> 0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, key_len) for broadcasting
            scores = scores.masked_fill(key_padding_mask, float('-inf'))  # Set masked positions to -inf

        # Compute attention weights using softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (batch_size, num_heads, query_len, key_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # Shape: (batch_size, 1, query_len, key_len)
            attn_weights = attn_weights.masked_fill(attn_mask, 0)  # Set masked positions to 0

        # Optionally apply dropout to the attention weights if self.dropout is defined
        attn_weights = self.attn_dropout(attn_weights)
        # Compute the weighted sum of the values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, query_len, head_dim)
        # Recombine heads: transpose and reshape back to (batch_size, query_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)

        # Final linear projection and dropout
        output = self.out_proj(attn_output)
        output = self.drop_out(output)

        return output, attn_weights

class InteractionModelATTN(nn.Module):
    def __init__(self, target_encoder, drug_encoder, scaler, attention_dropout, hidden_dropout, num_heads=1, kernel_size=1):
        super().__init__()
        self.scaler = scaler
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.target_encoder = target_encoder
        self.drug_encoder = drug_encoder
        self.kernel_size = kernel_size
        self.lin_map_target = nn.Linear(512, 384)
        self.dropout_map_target = nn.Dropout(hidden_dropout)

        self.lin_map_drug = nn.Linear(384, 384)
        self.dropout_map_drug = nn.Dropout(hidden_dropout)

        self.crossattention = CrossAttention(384, num_heads, attention_dropout, hidden_dropout)
        self.norm1 = nn.LayerNorm(384)                          # For attention output
        self.summary1 = nn.Linear(384, 384)
        self.summary2 = nn.Linear(384, 1)
        self.dropout_summary = nn.Dropout(hidden_dropout)

        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(512) 

        self.w = Parameter(torch.ones(1))
        self.b = Parameter(torch.zeros(1))

    def forward(self, x1, x2):     
        x1["attention_mask"] = x1["attention_mask"].bool()   # Fix dropout model issue: https://github.com/pytorch/pytorch/issues/86120
        y1 = self.target_encoder(**x1).last_hidden_state     # The target

        query_mask = x1["attention_mask"].unsqueeze(-1).to(y1.dtype)
        y1 = y1 * query_mask

        x2["attention_mask"] = x2["attention_mask"].bool()   # Fix dropout model issue: https://github.com/pytorch/pytorch/issues/86120
        y2 = self.drug_encoder(**x2).last_hidden_state       # The drug
        key_mask = x2["attention_mask"].unsqueeze(-1).to(y2.dtype)
        y2 = y2 * key_mask
        
        y1 = self.lin_map_target(y1)
        y1 = self.gelu(y1) 
        y1 = self.dropout_map_target(y1)

        y2 = self.lin_map_drug(y2)
        y2 = self.gelu(y2) 
        y2 = self.dropout_map_drug(y2)

        key_padding_mask=(x2["attention_mask"] == 0) # S

        out, _ = self.crossattention(y1, y2, y2, key_padding_mask=key_padding_mask, attn_mask=None)

        out = self.summary1(out * query_mask)
        out = self.gelu(out)
        out = self.dropout_summary(out)
        out = self.summary2(out).squeeze(-1)

        out = self.layer_norm(out)

        out = out.sum(dim=1, keepdim=True)*self.w + self.b
        return out

    def train(self, mode = True): 
        super().train(mode)
        self.target_encoder.train(mode)
        self.drug_encoder.train(mode)
        return self
    
    def eval(self): 
        super().eval()
        self.target_encoder.eval()
        self.drug_encoder.eval()
        return self
    """
        Unscales the labels using a scaler. If the scaler is not specified, don't do anything.

        Parameters:
            target_value: the target values to be unscaled
    """
    def unscale(self, x):
        with torch.no_grad():
            if self.scaler is None:
                return x
            unscaled = self.scaler.inverse_transform(x)
        return unscaled

class MuTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = MuAdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, eps=self.args.adam_epsilon)
                
        #smp.state.cfg.fp16
        if IS_SAGEMAKER_MP_POST_1_10 and False:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio*num_training_steps, num_training_steps=num_training_steps, num_cycles=0.41)

class InterDataset(Dataset):
    def __init__(self, targets, smiles, pkd=None, weights=None):
        self.targets = targets
        self.smiles = smiles
        self.pkd = pkd
        self.weights = weights

    def __len__(self):
        return len(self.targets["input_ids"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input1, input2 = self.targets[idx], self.smiles[idx]
        class_int = self.weights[idx] if not self.weights is None else 1
        input1 = {"input_ids":self.targets["input_ids"][idx], "attention_mask": self.targets["attention_mask"][idx]}
        input2 = {"input_ids":self.smiles["input_ids"][idx], "attention_mask": self.smiles["attention_mask"][idx]}
        
        target = None
        if not self.pkd is None:
            target = self.pkd[idx]

        return ((input1, input2, class_int), target)

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

