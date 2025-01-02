import train
import pandas as pd
import model 
from transformers import AutoTokenizer, RobertaTokenizerFast
from sklearn.model_selection import train_test_split
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments, HfArgumentParser
from model import ModelArgs

if __name__ == "__main__":

    # During the final training, only then select sets such as sequences in training and validation sets don't overlap
    train_parameters = {"train_batch_size": 4,
                        "device": "cuda",
                        "learning_rate": 3e-4,
                        "adam_epsilon": 1e-6,
                        "gradient_accumulation_steps": 16,
                        "layers_to_remove":6,
                        #"num_training_steps":3000,
                        "num_epochs":1,
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

    parser = HfArgumentParser((TrainingArguments, ModelArgs,))
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--learning_rate', '2e-4',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '5000',
        '--logging_steps', '100',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_device_eval_batch_size', '8',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '32',
        '--do_train',
        '--do_eval',
        #'--num_train_epochs', '1',
        #'--tpu_num_cores', '8',                      # Number of TPU cores (typically 8)
        '--lr_scheduler_type', 'cosine',
        '--warmup_ratio', '0.1',
        # This cosine scheduler drops to min lr rate of zero, not of 10x less the initial lr like in the paper

        # This drops approximately 10x  
        '--lr_scheduler_kwargs', '{"num_cycles": 0.41}',            
    ])

    training_args.train_datapath = './processed/dataset/train/train.txt'
    training_args.test_datapath = './processed/dataset/test/test.txt'

    training_args.prediction_loss_only = True
    print("Device:", training_args.device)

    target_encoder = train.load_RNABERTa(train_parameters["layers_to_remove"])

    train.pretrain_and_evaluate(training_args, target_encoder, target_tokenizer, True, None)
    
    # Try with smaller models
    inters = pd.read_csv("processed/interactions/all.csv")
    X = inters[["SMILES", "Target_RNA_sequence"]]
    y = inters['pKd'].values
    classes = inters["Category"].values
    
    scaler = model.StdScaler()
    #result = train.crossvalidate(X, y, 3, train_parameters, scaler, classes)
    #print(result)
    #finetune_model, accelerator, train_dataset, val_dataset, scaler = train.load_finetuned_model("lora_adapter/20250102_155624")

    #train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.875, stratify=classes, random_state=42)
    #train_dataset, val_dataset = train.__prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot=True)
    #accelerator, finetune_model = train.create_finetune_model(train_parameters)
    #scores, finetune_model = train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)
    #train.save_finetuned_model(finetune_model, train_parameters, train_dataset, val_dataset, scaler)
    #best_params = train.optimize(X, y, 10, scaler, classes, True)


    #accelerator, finetune_model = train.create_finetune_model(train_parameters)
    #scores, finetune_model = train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)

