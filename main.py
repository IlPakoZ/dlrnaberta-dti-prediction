import train
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
import torch
import os
import pandas as pd 
import model
from transformers import TrainingArguments, HfArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
import evaluate
import numpy as np
from tqdm import tqdm
import sys

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
    parser.add_argument('--task', '-t', type=int, required=True, help="Task to perform:\n\t1 for pretraining RNABERTa;\n\t2 for finetuning with crossvalidation;\n\t3 for pretraining hyperparameter optimization\n\t4 for finetuning hyperparameter optimization;\n\t5 for training the tokenizer;\n\t6 for finetuning the model;\n\t7 for prediction.")
    parser.add_argument('--tok_file', '-k', type=str, required=False, nargs='+', help="List of files for training the tokenizer (tokenizer training only)")
    parser.add_argument('--path', '-p', type=str, required=False, help="Path to the folder containing the datasets")
    parser.add_argument('--lr', '-l', type=float, required=False, default=3e-4, help="Learning rate for the training")
    parser.add_argument('--warmup', '-w', type=float, required=False, default=0.1, help="Warmup ratio for the training")
    parser.add_argument('--full', '-u', action='store_true', help="Train the full encoder rather than the small version (pretraining only)")
    parser.add_argument("--hidden_do", "-n", type=float, required=False, default=0.0, help="Dropout rate for the model's hidden layers")
    parser.add_argument("--attention_do", "-a", type=float, required=False, default=0.0, help="Dropout rate for the model's attention layers")
    parser.add_argument("--weight_decay", "-y", type=float, required=False, default=0.0, help="Weight decay for the optimizer")
    parser.add_argument("--decay_ratio", "-r", type=float, required=False, default=0.0, help="Decay ratio for the learning rate scheduler")
    parser.add_argument("--finetune_data", "-f", type=str, required=False, default="all.csv", help="Finetune dataset file")
    parser.add_argument("--load_weights", "-g", type=str, required=False, default=None, help="Path to the weights to load for the finetuning model")
    parser.add_argument("--compute_weights", "-c", action='store_true', help="Uses a MSE loss weighted by class during finetuning")
    parser.add_argument("--continue_optimize", "-s", type=str, default=None, help="Continue the finetuning optimization process by using the SQLite database file specified")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input a .csv file to use for prediction. The file should contain target-drug pairs, one per line, with elements separated by a comma")
    parser.add_argument("--output", "-o", type=str, default=None, help="Outputs a .csv file containing predictions for each target-drug pair. If not specified, the predictions will be printed to the console")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use for prediction (task 7). Default is 'cuda'.")
    args = parser.parse_args()

    if args.task == 1:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        if not args.lr:
            print("Please, specify the learning rate using the -l flag")
            exit(1)
        if not args.warmup:
            print("Please, specify the warmup ratio using the -w flag")
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
            print("Please, specify the learning rate using the -l flag")
            exit(1)    
        
   
        tokenizer, _ = train.__get_tokenizers__()
        world_size = int(os.getenv("WORLD_SIZE", 1))

        train_parameters = {
            "train_batch_size": 32//world_size,
            "device": "cuda",
            "validate_while_training": True,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "adam_epsilon": 1e-6,
            "num_epochs": 300,
            "weight_decay": args.weight_decay,
            "hidden_dropout": args.hidden_do,
            "attention_dropout": args.attention_do,
            "decay_ratio": args.decay_ratio,
            "target_epochs":100,
        }
       
        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
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
            
        result = train.crossvalidate(X, y, 10, train_parameters, scaler, classes, path=pretrained_model_path, compute_weights=args.compute_weights)
        print(result)       
    
    elif args.task == 3:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        train.pretrain_optimize(args.path)

    elif args.task == 4:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        
        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
        pretrained_model_path = f"{args.path}/pretrained"
        X = inters[["SMILES", "Target_RNA_sequence"]]
        y = inters['pKd'].values
        scaler = model.StdScaler()
        classes = inters["Category"].values

        train.finetune_optimize(X, y, 10, scaler, classes, True, pretrained_model_path, compute_weights=args.compute_weights, optimize_path=args.continue_optimize)

    elif args.task == 5:
        if not args.tok_file:
            print("Please, specify the files to train the tokenizer on using the -k flag")
            exit(1)
        train.train_tokenizer(args.tok_file)
        exit(0)    
    elif args.task == 6:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        if not args.lr:
            print("Please, specify the learning rate using the -l flag")
            exit(1)  

        target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
        world_size = int(os.getenv("WORLD_SIZE", 1))

        train_parameters = {
            "train_batch_size": 32//world_size,
            "device": "cuda",
            "validate_while_training": False,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "adam_epsilon": 1e-6,
            "num_epochs": 300,
            "weight_decay": args.weight_decay,
            "hidden_dropout": args.hidden_do,
            "attention_dropout": args.attention_do,
            "decay_ratio": args.decay_ratio,
            "target_epochs":100
        } 
        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
        X = inters[["SMILES", "Target_RNA_sequence"]]
        y = inters['pKd'].values

        if args.compute_weights:
            classes = inters["Category"].values
        else:
            classes = None

        has_local = "LOCAL_RANK" in os.environ 
        local_rank = int(os.getenv("LOCAL_RANK", 0))

        if has_local:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)

        scaler = model.StdScaler()
        pretrained_model_path = f"{args.path}/pretrained"
        train_dataset, val_dataset = train.__prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, X, X, y, y, scaler, False, weights=classes)
        accelerator, finetune_model = train.create_finetune_model(train_parameters, pretrained_model_path, scaler)
        scores, finetune_model, best_scores, best_finetune_model, difference_validation_train = train.finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, True)
    elif args.task == 7:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)
        if not args.load_weights:
            print("Please, specify the path to the weights to load using the -g flag")
            exit(1)
        if not args.input:
            print("Please, specify the input file using the -i flag")
            exit(1)
        else:
            if not os.path.exists(args.input):
                print("Input file not found")
                exit(1)
        
        has_local = "LOCAL_RANK" in os.environ 
        local_rank = int(os.getenv("LOCAL_RANK", 0))

        if has_local:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)

        # Read the input CSV file where rows are drugs and columns are targets
        inters = pd.read_csv(args.input, index_col=0)
        drugs = inters.index
        targets = inters.columns

        target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
        pretrained_model_path = f"{args.path}/pretrained"
        finetune_model, _, train_dataset, val_dataset, scaler = train.load_finetuned_model(pretrained_model_path, args.load_weights)

        # Prepare the predictions dataframe
        predictions = pd.DataFrame(index=drugs, columns=targets)

        # Tokenize all drugs and targets once
        tokenized_drugs = {drug: evaluate.tokenize_inputs(None, drug, target_tokenizer, drug_tokenizer, only_drug=True)[1] for drug in tqdm(drugs, desc="Tokenizing drugs")}
        tokenized_targets = {target: evaluate.tokenize_inputs(target, None, target_tokenizer, drug_tokenizer, only_target=True)[0] for target in tqdm(targets, desc="Tokenizing targets")}
        all_target_vs = [tokenized_targets[target] for target in targets]
        print("Predicting drug-target interactions")    
        i=1

        if has_local:
            torch.distributed.barrier()

        for drug in drugs:
            start_time = datetime.now()
            drug_v = tokenized_drugs[drug]            
            res = list(evaluate.predict(finetune_model, all_target_vs, [drug_v]*len(all_target_vs), device=args.device)) 
            predictions.loc[drug] = res
            print(f"Drug {i}/{len(drugs)} completed in {(datetime.now() - start_time).total_seconds()} seconds")
            i += 1
            

        if args.output:
            # Save the predictions to a CSV file in the same format
            predictions.to_csv(args.output)
            print(f"Predictions saved to {args.output}")
        else:
            # Print the predictions to the console
            print(predictions)

    elif args.task == 8:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        if not args.load_weights:
            print("Please, specify the path to the weights to load using the -g flag")
            exit(1)

        inters = pd.read_csv(f"{args.path}/processed/interactions/")
        X = inters[["SMILES", "Target_RNA_sequence"]]
        y = inters['pKd'].values

        pretrained_model_path = f"{args.path}/pretrained"

        target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
        finetune_model, _, train_dataset, val_dataset, scaler = train.load_finetuned_model(pretrained_model_path, args.load_weights)

        targets = []
        drugs = []
        for drug, target in zip(X["SMILES"], X["Target_RNA_sequence"]):
            target_v, drug_v = evaluate.tokenize_inputs(target, drug, target_tokenizer, drug_tokenizer)
            targets.append(target_v)
            drugs.append(drug_v) 

        MAE_mean = 0
        i = 0
        differences = []
        for (res, yval) in zip(evaluate.predict(finetune_model, targets, drugs), y):
            train.print_if_0_rank("PREDICTED, REAL:", res, yval)
            train.print_if_0_rank("Difference:", res[0][0] - yval)
            difference = abs(res[0][0] - yval)
            differences.append(difference)
            MAE_mean += difference
            i += 1

        MAE_mean /= i
        train.print_if_0_rank("MAE:", MAE_mean)
        import matplotlib.pyplot as plt

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title('MAE Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

        # Save the plot
        plot_path = os.path.join(args.load_weights, "mae_distribution.png")
        plt.savefig(plot_path)
        print("Plot saved at:", plot_path)
        print(f"MAE distribution plot saved at {plot_path}")
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
    
  
