import train
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
import torch
import os
import pandas as pd 
import model
from transformers import TrainingArguments, HfArgumentParser
import evaluate
from tqdm import tqdm
import analysis

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
    parser.add_argument('--task', '-t', type=int, required=True, help="Task to perform:\n\t1 for pretraining RNABERTa;\n\t2 for finetuning with crossvalidation;\n\t3 for pretraining hyperparameter optimization\n\t4 for finetuning hyperparameter optimization;\n\t5 for training the tokenizer;\n\t6 for finetuning the model;\n\t7 for prediction.\n\t8 for interpretability analysis;\n\t9 for interpretability analysis with attention modification.")
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
    parser.add_argument("--input", "-i", type=str, default=None, help="Input a .csv file to use for prediction (task 7). The file should contain target-drug pairs in .csv format with 'SMILES' and 'Target_RNA_sequence' columns, if the --cross argument is not specified. If the --cross argument is specified, the file should contain drugs in rows and targets in columns, with the first column being 'SMILES' and the rest being target sequences")
    parser.add_argument("--output", "-o", type=str, default=None, help="Outputs a .csv file containing predictions for each target-drug pair. If not specified, the predictions will be printed to the console")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use for prediction (task 7). Default is 'cuda'.")
    parser.add_argument("--cross", action='store_true', default=False, help="If specified, the model will take as input a file with drugs in rows and targets in columns and predict the interactions between each one of them. If false, the file should contain target-drug pairs in .csv format with 'SMILES' and 'Target_RNA_sequence' columns. Default is False.")
    parser.add_argument("--split", type=str, default="random", help="Specify 'random' to use random split, or 'group' to use group splitting based on RNA sequence. Default is 'random'.")
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
            "num_epochs": 100,
            "weight_decay": args.weight_decay,
            "hidden_dropout": args.hidden_do,
            "attention_dropout": args.attention_do,
            "decay_ratio": args.decay_ratio,
            "target_epochs":100,
        }
       
        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
        cols = ["SMILES", "Target_RNA_sequence"]
        if "fold" in inters.columns:
            cols.append("fold")

        X = inters[cols]
        y = inters['pKd'].values
        classes = inters["Category"].values

    
        scaler = model.StdScaler()
        pretrained_model_path = f"{args.path}/pretrained"
        has_local = "LOCAL_RANK" in os.environ 
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        if has_local:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)
            
        result = train.crossvalidate(X, y, 10, train_parameters, scaler, classes, path=pretrained_model_path, compute_weights=args.compute_weights, split_type=args.split.lower())
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
        cols = ["SMILES", "Target_RNA_sequence"]
        if "fold" in inters.columns:
            cols.append("fold")

        X = inters[cols]
        y = inters['pKd'].values
        scaler = model.StdScaler()
        classes = inters["Category"].values

        train.finetune_optimize(X, y, 10, scaler, classes, True, pretrained_model_path, compute_weights=args.compute_weights, optimize_path=args.continue_optimize, split_type=args.split.lower())

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
            "num_epochs": 100,
            "weight_decay": args.weight_decay,
            "hidden_dropout": args.hidden_do,
            "attention_dropout": args.attention_do,
            "decay_ratio": args.decay_ratio,
            "target_epochs":100
        } 
        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
        cols = ["SMILES", "Target_RNA_sequence"]
        if "fold" in inters.columns:
            cols.append("fold")

        X = inters[cols]
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
        # sanity checks
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

        # load model and tokenizers
        target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
        pretrained_model_path = f"{args.path}/pretrained"
        finetune_model, _, train_dataset, val_dataset, scaler = train.load_finetuned_model(
            pretrained_model_path, args.load_weights
        )
        finetune_model = finetune_model.to(args.device)
        finetune_model.eval()

        if args.cross:
            # -------------------------------------------------------
            # CROSS MODE: input is a matrix of drugs×targets
            # -------------------------------------------------------
            inters = pd.read_csv(args.input, index_col=0)
            drugs = inters.index
            targets = inters.columns[1:]  # first col is SMILES

            # prep output DataFrame
            predictions = pd.DataFrame(index=drugs, columns=targets)

            # tokenize once
            tokenized_drugs = {
                drug: evaluate.tokenize_inputs(
                    None, drug, target_tokenizer, drug_tokenizer, only_drug=True
                )[1]
                for drug in tqdm(drugs, desc="Tokenizing drugs")
            }
            tokenized_targets = {
                tgt: evaluate.tokenize_inputs(
                    tgt, None, target_tokenizer, drug_tokenizer, only_target=True
                )[0]
                for tgt in tqdm(targets, desc="Tokenizing targets")
            }
            all_tgt_vs = [tokenized_targets[t] for t in targets]

            if has_local:
                torch.distributed.barrier()

            # predict all pairs
            print("Predicting drug–target interactions")
            for i, drug in enumerate(drugs, start=1):
                start_time = datetime.now()
                drug_v = tokenized_drugs[drug]
                scores = evaluate.predict(
                    finetune_model,
                    all_tgt_vs,
                    [drug_v] * len(all_tgt_vs),
                    device=args.device
                )
                row = pd.Series(list(scores), index=targets)
                # only keep “active” predictions >= 6
                predictions.loc[drug, row >= 6] = row[row >= 6]
                print(f"Drug {i}/{len(drugs)} done in {(datetime.now() - start_time).total_seconds():.1f}s")

            predictions = predictions.fillna('')

        else:
            # —————————————————————————
            # STANDARD MODE
            # —————————————————————————
            inters = pd.read_csv(f"{args.input}")
            smiles_list = inters["SMILES"].tolist()
            seq_list    = inters["Target_RNA_sequence"].tolist()
            y_true = inters["pKd"].values if "pKd" in inters.columns else "NA"

            # tokenize each pair
            tokenized_targets = []
            tokenized_drugs   = []
            for seq, smi in tqdm(zip(seq_list, smiles_list), desc="Tokenizing input data..."):
                t_t, d_t = evaluate.tokenize_inputs(
                    seq, smi,
                    target_tokenizer, drug_tokenizer,
                    only_target=False, only_drug=False
                )
                tokenized_targets.append(t_t)
                tokenized_drugs.append(d_t)
                

            # predict pKd for each pair
            preds = evaluate.predict(
                finetune_model,
                tokenized_targets,
                tokenized_drugs,
                device=args.device
            )
            preds = list(preds)  # if it returns a generator

            # assemble output DataFrame
            predictions = pd.DataFrame({
                "SMILES":              smiles_list,
                "Target_RNA_sequence": seq_list,
                "true_pKd":            y_true,
                "predicted_pKd":       preds
            })

        # save or print
        if args.output:
            # if cross: index is drugs, else default RangeIndex
            predictions.to_csv(args.output, index=args.cross)
            print(f"Predictions saved to {args.output}")
        else:
            for i, row in predictions.iterrows():
                smi = row["SMILES"]
                seq = row["Target_RNA_sequence"]
                true_val = row["true_pKd"]
                pred_val = row["predicted_pKd"]

                # Truncate long SMILES and sequences
                smi_display = smi if len(smi) <= 20 else smi[:17] + "..."
                seq_display = seq if len(seq) <= 20 else seq[:17] + "..."

                print(f"[{i+1:>3}] SMILES: {smi_display:<20}\t Target: {seq_display:<20}\t True pKd: {true_val:.3f}\t Predicted pKd: {pred_val:.3f}")



    elif args.task == 8: 
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        if not args.load_weights:
            print("Please, specify the path to the weights to load using the -g flag")
            exit(1)

        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
        cols = ["SMILES", "Target_RNA_sequence"]
        if "fold" in inters.columns:
            cols.append("fold")

        X = inters[cols]
        y = inters['pKd']

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
        
        
        finetune_model = finetune_model.to(args.device)
        finetune_model.eval()
        finetune_model.INTERPR_ENABLE_MODE()

        for (target, drug, res, presum), yval in zip(evaluate.interp_predict(finetune_model, targets, drugs), y):
            train.print_if_0_rank("PREDICTED, REAL:", res[0][0], yval)
            train.print_if_0_rank("Difference:", res[0][0] - yval)

            if yval > 8 and res[0][0] > yval - 0.5 and res[0][0] < yval + 0.5:
                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, i, raw_affinities=True, path=args.load_weights)
                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, i, raw_affinities=False, path=args.load_weights)
                analysis.plot_crossattention_weights(target["attention_mask"][0], drug["attention_mask"][0], target, drug, finetune_model.model.crossattention_weights[0][0], i, path=args.load_weights)

            # Calculate the absolute difference
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


    elif args.task == 9:
        if not args.path:
            print("Please, specify the folder containing the training and validation datasets using the -p flag")
            exit(1)

        if not args.load_weights:
            print("Please, specify the path to the weights to load using the -g flag")
            exit(1)

        inters = pd.read_csv(f"{args.path}/processed/interactions/{args.finetune_data}")
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
        
        finetune_model = finetune_model.to(args.device)
        finetune_model.eval()
        finetune_model.INTERPR_ENABLE_MODE()

        for (target, drug, res, presum), yval in zip(evaluate.interp_predict(finetune_model, targets, drugs), y):
            orig_pred = res[0][0].item()
            orig_diff = abs(orig_pred - yval.item())
            train.print_if_0_rank("PREDICTED, REAL:", orig_pred, yval)
            train.print_if_0_rank("Difference:", orig_diff)
            if yval.item() > 8 and orig_pred > yval.item() - 0.5 and orig_pred < yval.item() + 0.5:

                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, f"{i}pre", raw_affinities=False, path=args.load_weights)
                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, f"{i}pre", raw_affinities=True, path=args.load_weights)
                analysis.plot_crossattention_weights(target["attention_mask"][0], drug["attention_mask"][0], target, drug, finetune_model.model.crossattention_weights[0][0], f"{i}pre", path=args.load_weights)

                scores = finetune_model.model.scores
                new_attention_scores = finetune_model.model.scores[0][0]
                weights = torch.nn.functional.softmax(scores, dim=-1)[0][0]

                max_index = torch.argmax(weights)

                new_attention_scores.view(-1)[max_index] = -10000
                finetune_model.INTERPR_OVERRIDE_ATTN(new_attention_scores.reshape(1, 1, *new_attention_scores.shape))
                # re-predict with modified attention
                gen = evaluate.interp_predict(finetune_model, [target], [drug])
                

                _, _, after_res, presum = next(gen)
                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, f"{i}post", raw_affinities=False, path=args.load_weights)
                analysis.plot_presum(target, presum, finetune_model.model.scaler, finetune_model.model.w.squeeze(1), finetune_model.model.b, f"{i}post", raw_affinities=True, path=args.load_weights)
                analysis.plot_crossattention_weights(target["attention_mask"][0], drug["attention_mask"][0], target, drug, finetune_model.model.crossattention_weights[0][0], f"{i}post", path=args.load_weights)
                new_pred = after_res[0][0].item()
                new_diff = abs(new_pred - yval.item())

                # Print after-modification results
                train.print_if_0_rank("MOD  PREDICTED, REAL:", new_pred, yval.item())
                train.print_if_0_rank("MOD    Difference:", new_diff)

                # Compute change
                diff_change = new_diff - orig_diff
                sign = "increased" if diff_change > 0 else "decreased" if diff_change < 0 else "unchanged"
                magnitude = abs(diff_change)

                train.print_if_0_rank(
                    f"ERROR {sign} by", magnitude
                )
                train.print_if_0_rank(
                    f"PRED change:", new_pred - orig_pred
                )

                    
            finetune_model.INTERPR_RESET_OVERRIDE_ATTN()
            # Calculate the absolute difference
            difference = abs(res[0][0] - yval)
            differences.append(difference)
            MAE_mean += difference
            i += 1
        
        MAE_mean /= i
        train.print_if_0_rank("MAE:", MAE_mean)
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
    
  
