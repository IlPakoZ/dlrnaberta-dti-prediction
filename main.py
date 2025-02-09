import train
import pandas as pd
import model 
from transformers import AutoTokenizer, RobertaTokenizerFast, AutoModel
from mutransformers import RobertaConfig, RobertaForMaskedLM

from sklearn.model_selection import train_test_split
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments, HfArgumentParser
from model import ModelArgs
import argparse
from argparse import RawTextHelpFormatter
from mup import set_base_shapes
import mup
import torch

if __name__ == "__main__":

    # During the final training, only then select sets such as sequences in training and validation sets don't overlap
    train_parameters = {"train_batch_size": 4,
                        "device": "cuda",
                        "learning_rate": 5e-5,
                        "adam_epsilon": 1e-6,
                        "gradient_accumulation_steps": 8,
                        #"num_training_steps":3000,
                        "num_epochs":10,
                        "log_performance_every":5,
                        "weight_decay": 0.01,
                        "model_dropout": 0.2,
                        "lora_r": 16,
                        "lora_alpha":16,
                        "lora_dropout":0.1,
                        "max_norm":1,
                        #plot_grads":True,
                        "validate_while_training":True,
                        "val_all": True,
                        "hidden_layers": 12,
                        "hidden_size": 512,
                        "num_attention_heads": 16,
                        }
    
    """

    parser = argparse.ArgumentParser(description="RNA-Based DTI", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--task', '-t', type=int, required=True, help="Task to perform:\n\t1 for pretraining RNABERTa;\n\t2 for finetuning RNABERTa;\n\t3 for finetuning with crossvalidation\n\t4 for hyperparameter optimization;\n\t5 for training the tokenizer.")
    parser.add_argument('--tok_file', '-tf', type=str, required=False, nargs='+', help="List of files for training the tokenizer")
    args = parser.parse_args()

    if args.task == 1:
        raise NotImplementedError("Pretrain RNABERTa is not implemented yet")
    elif args.task == 2:
        raise NotImplementedError("Finetune RNABERTa is not implemented yet")
    elif args.task == 3:
        raise NotImplementedError("Finetune with crossvalidation is not implemented yet")
    elif args.task == 4:
        raise NotImplementedError("Optimize hyperparameters is not implemented yet")
    elif args.task == 5:
        if not args.tok_file:
            print("Specify the files to train the tokenizer on using the -tf flag")
            exit(1)
        train.train_tokenizer(args.tok_file)
        exit(0)
    else:
        print("Invalid task")
        exit(1)
    
    argparser = HfArgumentParser((TrainingArguments, ModelArgs,))
    training_args, model_args = argparser.parse_args_into_dataclasses(look_for_args_file=False, args=[
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
    """
    """
    inters = pd.read_csv("processed/interactions/all.csv")
    X = inters[["SMILES", "Target_RNA_sequence"]]
    y = inters['pKd'].values

    
    scaler = model.StdScaler()
    #result = train.crossvalidate(X, y, 3, train_parameters, scaler, classes)
    #print(result)
    #finetune_model, accelerator, train_dataset, val_dataset, scaler = train.load_finetuned_model("saves/20250120_081621_best")
    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
    train_X, val_X, train_y, val_y = train.split(inters, X, y, train_size=0.9, random_state=42)
    train_dataset, val_dataset = train.__prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot=True)
    accelerator, finetune_model = train.create_finetune_model(train_parameters)
  

    scores, finetune_model = train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)
    #train.save_finetuned_model(finetune_model, train_parameters, train_dataset, val_dataset, scaler)
    #best_params = train.optimize(X, y, 10, scaler, classes, True)

    #accelerator, finetune_model = train.create_finetune_model(train_parameters)
    #scores, finetune_model = train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)

    """
    """
    target_tokenizer, _ = train.__get_tokenizers__() 
    
    parser = HfArgumentParser((TrainingArguments, model.ModelArgs,))
    
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--logging_steps', '100',
        '--save_steps', '5000',
        '--max_grad_norm', '5.0',
        '--per_device_eval_batch_size', '32',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '64',
        '--do_train',
        '--do_eval',
        '--num_train_epochs', '1',
        #'--bf16', 'True'
      
    ])

    train_parameters = {
        "learning_rate": 1e-2,
        "weight_decay": 1e-2,
        "adam_epsilon": 1e-6,
        "warmup_ratio": 0.01,
    }
    training_args.train_datapath = 'processed/dataset/train/train.txt'
    training_args.val_datapath = 'processed/dataset/test/test.txt'
    target_config = RobertaConfig(vocab_size=len(target_tokenizer),
                                hidden_size=128,
                                num_hidden_layers=12,
                                num_attention_heads=16,  
                                intermediate_size=1024,
                                max_position_embeddings=514,
                                attn_mult=(32**0.5),
                                output_hidden_states=True)
    target_model = RobertaForMaskedLM(config=target_config)
    print(f"Number of parameters in target_model: {sum(p.numel() for p in target_model.parameters())}")
    set_base_shapes(target_model, "roberta512.bsh")

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of visible CUDA devices: {num_gpus}")
        # Print device names
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices are visible.")
    
    # If you obtain the model base_shapes using a base_model with shape equal to the model you want to use for finetuning,
    # width_mult() = 1, meaning that the MuReadOut() layer forward function will equal the Linear layer's one.
    # This means that the only thing left to do to transfer parameters from the pretrained model using mu initialization
    # to the parameters for the model you want to finetune is remove the attn_mult.
    # This is because the only two differences between mu initialization and sp initialization are the attn_mult and the
    # width_mult, for inference sake.
    # This can be checked by comparing the codes here:
    # https://github.com/microsoft/mutransformers/tree/main/mutransformers/models/roberta

    #for parameter in target_model.parameters():
    ##    print(parameter.infshape.width_mult())
    train.pretrain_and_evaluate(training_args, train_parameters, target_model, target_tokenizer, False, None, eval_first=False, save_result=False)[0]
    """
    train.pretrain_optimize()