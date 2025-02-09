import train


import argparse
from argparse import RawTextHelpFormatter
import torch
import os

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
    parser.add_argument('--task', '-t', type=int, required=True, help="Task to perform:\n\t1 for pretraining RNABERTa;\n\t2 for finetuning RNABERTa;\n\t3 for finetuning with crossvalidation\n\t4 for hyperparameter optimization;\n\t5 for training the tokenizer.")
    parser.add_argument('--tok_file', '-tf', type=str, required=False, nargs='+', help="List of files for training the tokenizer (tokenizer training only)")
    parser.add_argument('--path', '-p', type=str, required=False, help="Path to the folder containing the datasets")
    parser.add_argument('--lr', '-lr', type=float, required=False, default=3e-4, help="Learning rate for the training")
    parser.add_argument('--warmup', '-wu', type=float, required=False, default=0.1, help="Warmup ratio for the training")
    parser.add_argument('--full', '-fll', action='store_true', help="Train the full encoder rather than the small version (pretraining only)")
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

        training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
            '--output_dir', 'tmp',
            '--logging_steps', '200',
            '--save_steps', '1000',
            '--save_strategy', 'steps',
            '--max_grad_norm', '10.0',
            '--per_device_eval_batch_size', '64',
            '--per_device_train_batch_size', '32',
            '--gradient_accumulation_steps', str(8//int(os.environ["WORLD_SIZE"])),
            '--do_train',
            '--do_eval',
            '--num_train_epochs', '1',
            '--dataloader_pin_memory', 'True',      
            '--weight_decay', '0.01',
            '--adam_epsilon', '1e-6',
            '--ddp_find_unused_parameters', 'False'
        ])

        train_parameters = {
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-6,
            "warmup_ratio": args.warmup,
        }

        mu_model = train.model_init(0, args.full)
        datasets = train.load_pretrain_data(train_datapath, val_datapath, tokenizer, eval_only=False)
        train.pretrain_and_evaluate(training_args, train_parameters, datasets, mu_model, tokenizer, False, None, False, True)

    elif args.task == 2:
        raise NotImplementedError("Finetune RNABERTa is not implemented yet")
    
    elif args.task == 3:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        train.pretrain_optimize(args.path)

    elif args.task == 4:
        raise NotImplementedError("Optimize finetuning hyperparameters is not implemented yet")
    
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
