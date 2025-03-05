import train
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
import torch
import os
import pandas as pd 
import model
from transformers import TrainingArguments, HfArgumentParser


if __name__ == "__main__":
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


    parser = argparse.ArgumentParser(description="RNA-Based DTI", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--task', '-t', type=int, required=True, help="Task to perform:\n\t1 for pretraining RNABERTa;\n\t2 for finetuning with crossvalidation;\n\t3 for pretraining optimization\n\t4 for hyperparameter optimization;\n\t5 for training the tokenizer.")
    parser.add_argument('--tok_file', '-tf', type=str, required=False, nargs='+', help="List of files for training the tokenizer (tokenizer training only)")
    parser.add_argument('--path', '-p', type=str, required=False, help="Path to the folder containing the datasets")
    parser.add_argument('--lr', '-lr', type=float, required=False, default=3e-4, help="Learning rate for the training")
    parser.add_argument('--warmup', '-wu', type=float, required=False, default=0.1, help="Warmup ratio for the training")
    parser.add_argument('--full', '-fll', action='store_true', help="Train the full encoder rather than the small version (pretraining only)")
    parser.add_argument('--freeze', '-frz', action='store_true', help="During finetuning, use the freezed version of the finetuning model")

    args = parser.parse_args()

    if args.task == 1:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        if not args.lr:
            print("Please, specify the learning rate using the -lr flag")
            exit(1)
        if not args.warmup:
            print("Please, specify the warmup ratio using the -warmup flag")
            exit(1)

        train_datapath = f'{args.path}/processed/dataset/train/train.txt'
        val_datapath = f'{args.path}//processed/dataset/val/val.txt'
        tokenizer, _ = train.__get_tokenizers__()
        train_parser = HfArgumentParser((TrainingArguments, model.ModelArgs,))
        now = datetime.now().strftime('%Y%m%d_%H%M%S')

        training_args, model_args = train_parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
            '--output_dir', f'{args.path}/saves/{now}',
            '--logging_steps', '2000',
            '--save_steps', '2000',
            '--save_strategy', 'steps',
            '--evaluation_strategy', 'steps',
            '--per_device_eval_batch_size', '64',
            '--per_device_train_batch_size', '32',
            '--gradient_accumulation_steps', str(8//int(os.environ["WORLD_SIZE"])),
            '--do_train',
            '--do_eval',
            '--num_train_epochs', '1',
            '--dataloader_pin_memory', 'True',
            '--ddp_find_unused_parameters', 'False'
        ])

        train_parameters = {
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-6,
            "warmup_ratio": args.warmup,
        }

        mu_model = train.pretraining_model_init(0, args.full)
        datasets = train.load_pretrain_data(args.path, train_datapath, val_datapath, tokenizer, eval_only=False)
        train.pretrain_and_evaluate(training_args, train_parameters, datasets, mu_model, tokenizer, False, None, False, True)

    elif args.task == 2:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        if not args.lr:
            print("Please, specify the learning rate using the -lr flag")
            exit(1)    
        
   
        tokenizer, _ = train.__get_tokenizers__()
        world_size = int(os.getenv("WORLD_SIZE", 1))

        
        train_parameters = {
            "train_batch_size": 32//world_size,
            "device": "cuda",
            "validate_while_training": True,
            "gradient_accumulation_steps": 1,
            "exit_if_train_loss_less_than_or_converges": 5e-2,
            "learning_rate": args.lr,
            "adam_epsilon": 1e-6,
            "num_epochs": 100,
            "log_performance_every": 40,
            "weight_decay": 0.01,
            "model_dropout": 0.2,
            "max_norm": 1,
            "num_cycles": 0.25,
            "override_scheduler_steps": 2025*2
        }
       
        inters = pd.read_csv(f"{args.path}/processed/interactions/all.csv")
        X = inters[["SMILES", "Target_RNA_sequence"]]
        y = inters['pKd'].values
        classes = inters["Category"].values

    
        scaler = model.StdScaler()
        pretrained_model_path = f"{args.path}/pretrained"
        has_local = "LOCAL_RANK" in os.environ 
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        if has_local:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)
            
        result = train.crossvalidate(X, y, 10, train_parameters, scaler, classes, path=pretrained_model_path)
        print(result)
        # raise NotImplementedError("Finetune RNABERTa is not implemented yet")
       
    
    elif args.task == 3:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        train.pretrain_optimize(args.path)

    elif args.task == 4:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        
        inters = pd.read_csv(f"{args.path}/processed/interactions/all.csv")
        pretrained_model_path = f"{args.path}/pretrained"
        X = inters[["SMILES", "Target_RNA_sequence"]]
        y = inters['pKd'].values
        scaler = model.StdScaler()
        classes = inters["Category"].values
        #classes = [f"{cat}_{int(pkd/2+0.5)}" for cat, pkd in zip(inters["Category"].values ,inters["pKd"].values)]

        train.finetune_optimize(X, y, 10, scaler, classes, True, pretrained_model_path)

    elif args.task == 5:
        if not args.tok_file:
            print("Please, specify the files to train the tokenizer on using the -tf flag")
            exit(1)
        train.train_tokenizer(args.tok_file)
        exit(0)        
    else:
        print("Invalid task")
        exit(1)
    
    # If you obtain the model base_shapes using a base_model with shape equal to the model you want to use for finetuning,
    # width_mult() = 1, meaning that the MuReadOut() layer forward function will equal the Linear layer's one.
    # This means that the only thing left to do to transfer parameters from the pretrained model using mu initialization
    # to the parameters for the model you want to finetune is remove the attn_mult.
    # This is because the only two differences between mu initialization and sp initialization are the attn_mult and the
    # width_mult, for inference sake.
    # This can be checked by comparing the codes here:
    # https://github.com/microsoft/mutransformers/tree/main/mutransformers/models/roberta
    
  