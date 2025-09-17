# DLRNA-BERTa: A transformer approach for RNA-drug interaction prediction
This is a repository for our paper "DLRNA-BERTa: A transformer approach for RNA-drug binding affinity prediction[^1]" as a continuation of my Master's Thesis project.<br>

## Architecture 

A RoBERTa[^2] model is pretrained on RNA-sequences through a MLM task and used in conjuction to ChemBERTa-2[^3] to predict the binding of a drug to a RNA-based target.
The model is composed of three parts:
- a RoBERTa model as encoder for RNA-targets, taking RNA sequences as input; 
- a ChemBERTa-2 model as encoder for drug molecules, taking SMILES sequences as input;
- an interaction head, taking the outputs of the two encoders as input and predicting binding affinity.


![Overall DLRNA-BERTa model architecture.](https://github.com/IlPakoZ/dlrnaberta-dti-prediction/blob/main/imgs/model%20architecture.jpg)
<br><br>
<b>Figure 1</b>: The overall DLRNA-BERTa architecture is presented here.

### Technical specifications

- **Model size**: Combines RNA-BERTa (target encoder) + ChemBERTa-77M-MTR (drug encoder)
- **Cross-attention**: Single-head attention with 384-dimensional embeddings
- **Maximum sequence length**: 512 tokens for both target and drug inputs
- **Output**: Continuous binding affinity prediction (pKd values)
- **Dropout**: Configurable attention dropout and hidden dropout for regularization
- **Layer normalization**: Applied for training stability

## How to use

```python
from modeling_dlmberta import InteractionModelATTNForRegression, StdScaler
from configuration_dlmberta import InteractionModelATTNConfig
from transformers import AutoModel, RobertaModel, AutoConfig
from chemberta import ChembertaTokenizer

# Load model components
config = InteractionModelATTNConfig.from_pretrained("path/to/model")
# Load encoders
target_encoder = AutoModel.from_pretrained("IlPakoZ/RNA-BERTa9700")

drug_encoder_config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
drug_encoder_config.pooler = None
drug_encoder = RobertaModel(config=drug_encoder_config, add_pooling_layer=False)

# Load scaler (if available)
scaler = StdScaler()
scaler.load("path/to/model")

# Initialize model
model = InteractionModelATTNForRegression.from_pretrained(
    "path/to/model",
    config=config,
    target_encoder=target_encoder,
    drug_encoder=drug_encoder,
    scaler=scaler
)

# Make predictions
target_sequence = "AUGCGAUCGACGUACGUUAGCCGUAGCGUAGCUAGUGUAGCUAGUAGCU"
drug_smiles = "C1=CC=C(C=C1)NC(=O)C2=CC=CC=N2"

# Tokenize inputs
target_inputs = target_tokenizer(target_sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
drug_inputs = drug_tokenizer(drug_smiles, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Predict
with torch.no_grad():
    prediction = model(target_inputs, drug_inputs)
    if model.scaler:
        prediction = model.unscale(prediction)
```
        
## Datasets

### Fine-tuning Dataset (Training)
        
The model was trained on a dataset comprising **1,439 RNA–drug interaction pairs**, including:
- **759 unique compounds** (SMILES representations)
- **294 unique RNA sequences**
- Dissociation constants (pKd values) for binding affinity prediction
        
**RNA Sequence Distribution by Type:**
        
| RNA Sequence Type | Number of Interactions |
|-------------------|------------------------|
| Aptamers          | 520                    |
| Ribosomal         | 295                    |
| Viral RNAs        | 281                    |
| miRNAs            | 146                    |
| Riboswitches      | 100                    |
| Repeats           | 97                     |
| **Total**         | **1,439**              |
        
### External Evaluation Dataset (Test)
        
Model validation was performed using external ROBIN classification datasets containing **5,534 RNA–drug pairs**:
- **2,991 positive interactions**
- **2,538 negative interactions**
        
**Test Dataset Composition:**
- **1,617 aptamer pairs** (5 unique RNA sequences)
- **1,828 viral RNA pairs** (6 unique RNA sequences)  
- **1,459 riboswitch pairs** (5 unique RNA sequences)
- **630 miRNA pairs** (3 unique RNA sequences)

### Dataset Downloads
- [Training Dataset Download](https://huggingface.co/spaces/IlPakoZ/DLRNA-BERTa/resolve/main/datasets/training_data.csv?download=true)
- [Test Dataset Download](https://huggingface.co/spaces/IlPakoZ/DLRNA-BERTa/resolve/main/datasets/test_data.csv?download=true)

## Model evaluation

The model was evaluated on external ROBIN test datasets [^4] across different RNA classes:

| Dataset | Precision | Specificity | Recall | AUROC | F1 Score |
|---------|-----------|-------------|---------|-------|----------|
| Aptamers | 0.648 | 0.002 | 1.000 | 0.571 | 0.787 |
| Riboswitch | 0.519 | 0.035 | 0.972 | 0.577 | 0.677 |
| Viral RNA | 0.562 | 0.095 | 0.943 | 0.579 | 0.704 |
| miRNA | 0.373 | 0.028 | 0.991 | 0.596 | 0.542 |

![Test set performance comparison](https://github.com/IlPakoZ/dlrnaberta-dti-prediction/blob/main/imgs/combined_performance_comparison.jpg)
<b>Figure 2</b>: Performance comparison of RNA–drug interaction prediction models across four RNA classes. Bar plots show (A) F1-score and (B) AUROC for DLRNA-BERTA, RSAPred, DeepRNA-DTI, and DeepRSMA models on aptamers, riboswitches, miRNAs, and viral RNAs. DeepRNA-DTI (diagonal hatching, labeled "Classification") uses a classification approach and achieves higher performance due to greater data availability and different objective function. Regression models used pKd ≥ 4 threshold for active compounds. *Classification model threshold was selected to maximize F1-score.

## Interpretability features

The model includes advanced interpretability capabilities:

- **Cross-attention visualization**: Heatmaps showing interaction patterns between drug and target tokens
- **Unnormalized vs. normalized token-level contributions**: Visualization of individual token contributions (normalized and unnormalized) to the final prediction
- **Interpretation mode**: Special mode for extracting attention weights and intermediate values

### Enabling interpretation mode

```python
# Enable interpretation mode (evaluation only)
model.INTERPR_ENABLE_MODE()
# Make prediction with interpretation data
with torch.no_grad():
# Unscale if scaler exists
    prediction = model(target_inputs, drug_inputs)

    if self.model.scaler is not None:
        prediction = self.model.unscale(prediction)
                
    prediction_value = prediction.cpu().numpy()[0][0]
                
    # Access interpretation data
    cross_attention_weights = model.model.crossattention_weights
    presum_contributions = model.model.presum_layer
    attention_scores = model.model.scores

# Disable interpretation mode
model.INTERPR_DISABLE_MODE()
```

## Other links

[Try the model without set-up here.](https://huggingface.co/spaces/IlPakoZ/DLRNA-BERTa)<br>
[Model files and example implementation of DLRNA-BERTa can be found here.](https://huggingface.co/spaces/IlPakoZ/DLRNA-BERTa/tree/main)

## Citations

[^1]: [Lobascio, Pasquale, et al. "DLRNA-BERTa: A transformer approach for RNA-drug binding affinity prediction." bioRxiv (2025): 2025-09.](https://www.biorxiv.org/content/10.1101/2025.09.05.674445v1)
[^2]: [LIU, Yinhan. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019, 364.](https://arxiv.org/pdf/1907.11692)
[^3]: [AHMAD, Walid, et al. Chemberta-2: Towards chemical foundation models. arXiv preprint arXiv:2209.01712, 2022.](https://arxiv.org/abs/2209.01712)
[^4]: [Krishnan, Sowmya R., Arijit Roy, and M. Michael Gromiha. "Reliable method for predicting the binding affinity of RNA-small molecule interactions using machine learning." Briefings in Bioinformatics 25.2 (2024): bbae002.](https://academic.oup.com/bib/article/25/2/bbae002/7584787)
