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

## Requirements

Use a Python version <b>3.11</b> and install the requirements.txt file in this folder.
Separate CUDA installation is required to run the model on a GPU.

If you plan to try to run the model without using the requirements.txt file, make sure you install mup through the following command:<br>
```
pip install mup
```

Pretrained models and datasets are not uploaded in the repository directly. Instead, they are released as .zip and .tar.gz package in the "Releases" section of this repository. Move the extracted folders in the cloned repository folder for immediate use.

## How to use

### Use through your own script

After you satisfied the requirements, you can use the model in your own python scripts by importing the `model` library. Below a code snapshot to run model prediction on a single target and molecule.

```python
from transformers import AutoModel, RobertaModel, AutoConfig, AutoTokenizer
from model import ChembertaTokenizer
from model import InteractionModelATTNConfig, InteractionModelATTNForRegression, StdScaler
import torch

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
model_path = "saves"
# Load model components
config = InteractionModelATTNConfig.from_pretrained(model_path)

# Load encoders
target_encoder = AutoModel.from_pretrained("IlPakoZ/RNA-BERTa9700")
target_tokenizer = AutoTokenizer.from_pretrained("IlPakoZ/RNA-BERTa9700")
drug_tokenizer = ChembertaTokenizer(f"chemberta/vocab.json")

drug_encoder_config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
drug_encoder_config.pooler = None
drug_encoder = RobertaModel(config=drug_encoder_config, add_pooling_layer=False)

# Load scaler (if available)
scaler = StdScaler()
scaler.load(f"{model_path}")

# Initialize model
model = InteractionModelATTNForRegression.from_pretrained(
    model_path,
    config=config,
    target_encoder=target_encoder,
    drug_encoder=drug_encoder,
    scaler=scaler
).to(device)


# Make predictions
target_sequence = "AUGCGAUCGACGUACGUUAGCCGUAGCGUAGCUAGUGUAGCUAGUAGCU"
drug_smiles = "C1=CC=C(C=C1)NC(=O)C2=CC=CC=N2"

# Tokenize inputs
target_inputs = target_tokenizer(target_sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
drug_inputs = drug_tokenizer(drug_smiles, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

model.eval()

# Predict
with torch.no_grad():
    # Move inputs to the correct device
    target_inputs.to(device)
    drug_inputs.to(device)

    prediction = model(target_inputs, drug_inputs).cpu()
    if model.model.scaler:
        prediction = model.unscale(prediction)

print(f"Predicted pKd: {prediction[0][0]:.2f}")
```
        
You can run the script `test-author-model.py` to evaluate Krishnan et. al.[^4] model performance.
You can run the script `prediction-test.py` to evaluate the model's test performance (file produced through task 7).

For examples on how to interact with the model through the various pre-implemented tasks in the `main.py` file, refer to folder `commands`. Here you can find the commands we used during the development of the project.

### Use through Google Colab
Alternatively, to avoid the setup process, you can run the model through [this Google Colab notebook](https://colab.research.google.com/drive/1uHXBQ6XXNsRbc2sDnFM-y2iNwG7h3ZxN?usp=sharing). To use it, create a copy of the notebook in your Google Drive and execute it with your own input sequences and SMILES. The notebook includes examples demonstrating how to run the model in evaluation mode within Python. It is fully compatible with both CPU and CUDA environments.

### Use through Docker
A Dockerfile based on linux kernel is available in the repository and allows you to build a Docker container with everything pre-installed and configured to run the model. Make sure you have at least 15G of free space on you disk and are on a Linux based system. Alternatively, you can use WSL or Docker Desktop on Windows with Linux virtualization.<br>
First, make sure to clone the repository and open a terminal in the directory. You can do it on GitHub or using the command:
```
git clone https://github.com/IlPakoZ/dlrnaberta-dti-prediction.git && cd dlrnaberta-dti-prediction
```

Then, download the general model and dataset .zip/.tar package and decompress it in the repository. You can use the command:
```
wget https://github.com/IlPakoZ/dlrnaberta-dti-prediction/releases/download/general/general-model.tar.gz
tar -xzf general-model.tar.gz
```

Finally, build the Docker image from the current directory, create a container, and run it in interactive mode using:
```
docker build -t dlrnaberta-dti .
docker run -it dlrnaberta-dti
```

After running the container, you will have access to all files and can run the model through `main.py` or write your own scripts to interact with the model.

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
