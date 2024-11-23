# RNABert: RNA-based DTI prediction
This is a repository for my Master's Thesis project "RNABert: Pretraining of RoBERTa on RNA-based targets for Drug-Target Interaction (DTI) prediction". The name is temporary.<br>

A RoBERTa[^1] model is pretrained on RNA-sequences through a MLM task and used in conjuction to ChemBERTa-2[^2] to predict the binding of a drug to a RNA-based target.
The model is composed of three parts:
- a RoBERTa model as encoder for RNA-targets, taking RNA sequences as input; 
- a ChemBERTa-2 model as encoder for drug molecules, taking SMILES sequences as input;
- an interaction head, taking the outputs of the two encoders as input and predicting binding affinity.
<br>
Implementation of the prototype can be found in the "prototype" directory.<br><br>


[^1]: [LIU, Yinhan. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019, 364.](https://arxiv.org/pdf/1907.11692)
[^2]: [AHMAD, Walid, et al. Chemberta-2: Towards chemical foundation models. arXiv preprint arXiv:2209.01712, 2022.](https://arxiv.org/abs/2209.01712)
