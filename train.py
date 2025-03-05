
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing
from torchmetrics.regression import R2Score, MeanAbsoluteError, PearsonCorrCoef
from datetime import datetime

from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from mup import MuAdamW, set_base_shapes, make_base_shapes
from datasets import config
from model import MuTrainer
from accelerate.utils import DistributedDataParallelKwargs

import mutransformers as mu
import matplotlib.pyplot as plt
import torch.nn as nn
import model as md
import numpy as np
import pandas as pd
import torch
import copy
import optuna
import os
import pickle
import math
import gc
import psutil

creation_time = datetime.now().strftime('%Y%m%d_%H%M%S')
PLOT_DIR = f"plots/{creation_time}"

def train_tokenizer(train_files):
    """
        Trains a RoBERTa tokenizer from the files ``train_files``.
        Uses a character tokenizer for interpretability of tokens.
        
        Parameters:
            train_files (list str): a list of paths from which to train the tokenizer.
    """

    roberta_base_tokenizer = CharBPETokenizer()

    # Customize training (change vocab size)
    roberta_base_tokenizer.train(train_files, vocab_size=9700, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    roberta_base_tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", roberta_base_tokenizer.token_to_id("</s>")),
        ("<s>", roberta_base_tokenizer.token_to_id("<s>")),
    )

    roberta_base_tokenizer.enable_truncation(max_length=512)
    roberta_base_tokenizer.save_model("./tokenizer")

def __print_if_debug__(to_print, train_parameters):
    if "debug" in train_parameters:
        if train_parameters["debug"]:
            print(to_print)


def __in_train_evaluate_model__(model, accelerator, val_dataset, state, scaler, thread_id):
    """
        Evaluates the model on the validation data. 

        Parameters:
            model: the model to be evaluated
            accelerator: transformers accelerator used in training the model
            val_dataset (torch Dataset): the validation dataset
            state (dict str -> float): state of the training process, used to memorize metrics and other important state variables
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation
            thread_id (str): name of the thread
    """
    has_local = "LOCAL_RANK" in os.environ
    model.eval() 
    val_loader = DataLoader(val_dataset, shuffle=True)
    val_loader = accelerator.prepare(val_loader)

    r2_metric = R2Score().to(model.device)
    mae_metric = MeanAbsoluteError().to(model.device)
    pearson_r_metric = PearsonCorrCoef().to(model.device)
    r2_metric, mae_metric, pearson_r_metric = accelerator.prepare(r2_metric, mae_metric, pearson_r_metric)

    eval_n = 0

    with torch.no_grad():
        for source, targets in val_loader:
            targets = targets.reshape(-1, 1).to(model.device)
            output = model(source[0], source[1])
            
            targets = __unscale__(targets, scaler)
            output = __unscale__(output, scaler)

            r2_metric.update(output, targets)
            mae_metric.update(output, targets)
            pearson_r_metric.update(output, targets)

            eval_n += 1


    state["last_valid_mae_score"] = state["valid_mae_score"]
    if has_local:
        torch.distributed.barrier()
    state["valid_r2_score"] = r2_metric.compute()
    state["valid_mae_score"] = mae_metric.compute()
    state["valid_pearson_r_score"] = pearson_r_metric.compute()
    
    if state["valid_r2_score"] > state["best_r2"]: 
        state["best_r2"] = state["valid_r2_score"]
        state["best_mae_score"] = state["valid_mae_score"]
        state["best_model"] = copy.deepcopy(model)

    state["val_maes"].append(state["valid_mae_score"].cpu())
    state["val_r2s"].append(state["valid_r2_score"].cpu())
    state["val_pears"].append(state["valid_pearson_r_score"].cpu())
    
    accelerator.free_memory(r2_metric, mae_metric, pearson_r_metric)
    del r2_metric
    del mae_metric
    del pearson_r_metric
    print_if_0_rank(f"Step {thread_id} - R2 val: {state['valid_r2_score']:.4f}, MAE val: {state['valid_mae_score']:.4f}, Pearson R val: {state['valid_pearson_r_score']:.4f}")
      

def __check_mae_score_exit__(train_parameters, counter, train=True):
    """
        Checks whether the mae score has converged. If the  ``counter`` is greater then 5, then the loss
        is stable and training is considered complete.

        Parameters:
            train_parameters (dict str -> obj): training parameters
            counter (int): number of consecutive logging steps in which loss hasn't decreased
            train (bool): whether we are checking the train mae. If ``False``, validation mae is checked instead.
                Default: ``True``

        Return:
            ``True`` if the mae score converged, ``False`` otherwise.
    """
    if train:
        check = "exit_if_train_mae_converges"
    else:
        check = "exit_if_valid_mae_converges"

    if check in train_parameters:
        if counter >= 5:
            return True
    return False 
    
# https://huggingface.co/docs/peft/main/en/task_guides/semantic_segmentation_lora
def print_trainable_parameters(name, model):
    """
        Prints the number of trainable parameters in the model.
        
        Parameters:
            name (str): name of the model
            model: pytorch model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"{name} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def __update_exit_conditions__(mean_loss, train_parameters, state):
    """
        Checks the additional exit conditions. Returns ``True`` if the training process is finished.

        Parameters:
            mean_loss (float): mean training loss.
            train_parameters (dict str -> obj): training parameters
            state (dict str -> float): state of the training process, used to memorize metrics and other important state variables

        Returns:
            ``True`` if the training process is completed, ``False`` otherwise.
    """
    to_exit = False
    if "exit_if_train_loss_less_than_or_converges" in train_parameters:
        if mean_loss < train_parameters["exit_if_train_loss_less_than_or_converges"]:
            to_exit = True

        if (state["train_mae_score"] <= (state["last_train_mae_score"]+0.005)):
            state["counter_train"] += 1
        else:
            state["counter_train"] = 0

    if ("exit_if_valid_mae_converges" in train_parameters) and ("validate_while_training" in train_parameters) and train_parameters["validate_while_training"]:
        if (state["valid_mae_score"] <= (state["last_valid_mae_score"]+0.005)):
            state["counter_valid"] += 1
        else:
            state["counter_valid"] = 0
    
    to_exit = to_exit or __check_mae_score_exit__(train_parameters, state["counter_valid"], True) or __check_mae_score_exit__(train_parameters, state["counter_valid"], False)

    return to_exit

def __unscale__(target_value, scaler=None):
    """
        Unscales the labels using a scaler. If the scaler is not specified, don't do anything.

        Parameters:
            target_value: the target values to be unscaled
            scaler: the scaler used to scale the target values
    """
    if scaler is None:
        return target_value
    return scaler.inverse_transform(target_value)

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

def load_pretrain_data(path, train_datapath, val_datapath, tokenizer, eval_only=False):
    """
        Loads the pretraining data from the specified paths to RAM. 
        If the data is not tokenized, it will be tokenized and saved to disk.

        Parameters:
            path (str): path to the directory where the pretraining datasets will be saved
            train_datapath (str): path to the training data
            val_datapath (str): path to the validation data
            tokenizer: tokenizer used to tokenize the RNA sequences
            eval_only (bool): if ``True``, only the validation data will be loaded
                Default: ``False``
            
            Returns:
                The datasets loaded into memory."""
    
    data_files = {"val": val_datapath}
    
    if eval_only:
        data_files["train"] = data_files["val"]
    else:
        print_if_0_rank(f'Loading and tokenizing training data is usually slow: {train_datapath}')
        data_files["train"] = train_datapath

    if not os.path.exists(f"{path}/data/pretraining"):
        datasets = load_dataset("text", data_files=data_files)

        cache_files = {
             "train": f"{path}/train.parquet",
             "val": f"{path}/val.parquet"
        }
        
        datasets = datasets.map(lambda x: tokenize_function(tokenizer, x), batched=True, num_proc=10, cache_file_names=cache_files, load_from_cache_file=True)
    
        datasets.save_to_disk(f"{path}/data/pretraining")
    else:
        config.IN_MEMORY_MAX_SIZE = 45 * 1024**3        
        datasets = load_from_disk(f"{path}/data/pretraining", keep_in_memory=True)
    
    return datasets

def print_if_0_rank(*args):    
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(*args)


def pretrain_and_evaluate(args, training_parameters, datasets, model, tokenizer, eval_only, checkpoint_path=None, eval_first=False, save_result=True):
    """
        Pretrains the target encoder model and evaluates it on the validation data.
        This method assumes the model uses muParametrization.

        Parameters:
            args: training arguments
            training_parameters (dict str -> obj): training parameters
            datasets: dataset containing training and validation data for pretraining
            model: the model to be trained (target encoder)
            tokenizer: tokenizer used to tokenize the RNA sequences
            eval_only (bool): if ``True``, only validation will be performed
            checkpoint_path (str): path to the checkpoint from which to resume training
                Default: ``None``
            eval_first (bool): if ``True``, the model will be evaluated before training
                Default: ``False``
            save_result (bool): if ``True``, the pretrained model will be saved after training
                Default: ``True``
    """
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    optimizer = MuAdamW(model.parameters(), lr=training_parameters["learning_rate"], weight_decay=training_parameters["weight_decay"], eps=training_parameters["adam_epsilon"])    

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        print_if_0_rank("MAX STEPS MODE")
    else:
        num_training_steps = len(datasets["train"]) * args.num_train_epochs // (args.per_device_train_batch_size*args.gradient_accumulation_steps*int(os.environ["WORLD_SIZE"]))
        print_if_0_rank("NUM_EPOCHS MODE")

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_parameters["warmup_ratio"]*5000, num_training_steps=num_training_steps, num_cycles=0.41)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=datasets["train"], eval_dataset=datasets["val"], optimizers=(optimizer, scheduler))
    print(training_parameters["warmup_ratio"]*5000, num_training_steps)
    print(f"Trial started at {datetime.now().strftime('%H:%M:%S')}")
    
    if eval_first:
        print_if_0_rank("I'm evaluating...")
        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']/math.log(2)
        print_if_0_rank(f"Initial eval bpc: {eval_loss}")
    
    if not eval_only:
        print_if_0_rank("I'm training...")
        trainer.train(resume_from_checkpoint=checkpoint_path)
        if save_result:
            trainer.save_model(output_dir=args.output_dir)
        train_loss = trainer.state.log_history[-1]["train_loss"]
        print_if_0_rank(f"Train loss: {train_loss}")
        if "train_flos" in trainer.state.log_history[-1]:
            print_if_0_rank("Total FLOS:", trainer.state.log_history[-1]["train_flos"])

        # Plot training and validation loss
        training_loss = [log["loss"]/math.log(2) for log in trainer.state.log_history if "loss" in log]
        validation_loss = [log["eval_loss"]/math.log(2) for log in trainer.state.log_history if "eval_loss" in log]
        if int(os.environ["RANK"]) == 0:
            plt.figure(figsize=(10, 5))
            x = range(int(args.logging_steps), (len(training_loss)+1)*int(args.logging_steps), int(args.logging_steps))
            plt.plot(x, training_loss, label="Training Loss")
            plt.plot(x, validation_loss, label="Validation Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{args.output_dir}/training_validation_loss.png")


        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']/math.log(2)

        print_if_0_rank(f"Eval bpc after pretraining: {eval_loss}")

    return eval_loss, train_loss

def split(inters, X, y, train_size, random_state):
    """
    Splits the interaction datasets into training and validation sets while ensuring that 
    each class has at least two samples. Removes classes with only one sample and adds 
    them back to the training set after the split.
    Parameters:
        inters (pd.DataFrame): DataFrame containing interaction data with a "Category" column 
                            and a "pKd" column.
        X (pd.DataFrame): Dataframe containing features.
        y (np.ndarray): Array containing targets.
        train_size (float): Proportion of the dataset to include in the training set.
        random_state (int): Random seed for reproducibility.
    Returns:
        A tuple containing the training and validation feature matrices (train_X, val_X) 
        and the training and validation target vectors (train_y, val_y).
    """
    #classes = list(zip(inters["Category"].values, (inters["pKd"]+0.5).astype(int)))    
    classes = [f"{cat}_{int(pkd+0.5)}" for cat, pkd in zip(inters["Category"].values, inters["pKd"].values)]

    vals = dict()
    for i in range(len(classes)):
        c = classes[i]
        if not c in vals:
            vals[c] = []
        vals[c].append(i)

    to_remove = []
    for k,v in vals.items():
        if len(v) == 1:
            to_remove.append(v[0])

    X_res = X.loc[to_remove]
    y_res = y[to_remove]

    X = X.drop(to_remove)
    y = np.delete(y, to_remove)

    for index in to_remove[::-1]:
        classes.pop(index)

    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=train_size, stratify=classes, random_state=random_state)
    train_X = pd.concat((train_X, X_res))
    train_y = np.concatenate((train_y, y_res))
    return train_X, val_X, train_y, val_y

    
def finetune_and_evaluate(model, accelerator, train_parameters, train_dataset, val_dataset, scaler=None):
    """
        Trains (and evaluates) the model.
        Metrics are ``R2Score`` and ``MeanAbsoluteError`` from torchvision.

        Parameters:
            model: the model to be trained.
            accelerator: transformers library accelerator
            train_parameters (dict str -> obj): training parameters
            train_dataset (torch Dataset): Dataset containing training examples
            val_dataset (torch Dataset): Dataset containing validation examples
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation
        Returns:
            A list containing the values of the ``R2Score`` and ``MeanAbsoluteError`` metrics

    """
    has_local = "LOCAL_RANK" in os.environ
    world_size = int(os.getenv("WORLD_SIZE", 1))
    r2_train = R2Score()
    mae_train = MeanAbsoluteError()
    pearson_r_train = PearsonCorrCoef()
    state = {"valid_mae_score":0, "train_mae_score":0, "train_r2_score":0, "train_pearson_r_score":0, "valid_r2_score":0, "valid_pearson_r_score": 0, "counter_train":0, "counter_valid":0, "last_train_mae_score":0, "last_valid_mae_score":0, "best_r2":-100, "val_maes":[], "val_r2s":[], "val_pears":[]}

    optimizer = AdamW(model.parameters(), lr=train_parameters["learning_rate"], weight_decay=train_parameters["weight_decay"])
    train_loader = DataLoader(train_dataset, batch_size=train_parameters["train_batch_size"], shuffle=True)

    if "num_training_steps" in train_parameters:
        max_steps = train_parameters["num_training_steps"]//world_size
    else:
        max_steps = train_parameters["num_epochs"]*len(train_loader)//world_size

    warmup_steps = 0

    scheduler_training_steps = max_steps*world_size if "override_scheduler_steps" not in train_parameters else train_parameters["override_scheduler_steps"]
    # Create the cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps*world_size,
        num_training_steps=scheduler_training_steps,
        num_cycles=train_parameters["num_cycles"]
    )

    # Prepare objects for distributed training with Accelerator
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    r2_train, mae_train, pearson_r_train = accelerator.prepare(r2_train, mae_train, pearson_r_train)
    scheduler = accelerator.prepare(scheduler)

    print_if_0_rank(f"ALLOCATED MEMORY ON CUDA: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    print_if_0_rank(accelerator.state)
    if accelerator.state.deepspeed_plugin is not None:
        print_if_0_rank("DeepSpeed Plugin is enabled and working.")
    else:
        print_if_0_rank("DeepSpeed Plugin is not enabled.")
    model.train()

    step = 0  # Initialize the step variable
    epoch = 1


    print_if_0_rank(f"Training for {max_steps} steps")
    criterion = torch.nn.MSELoss()
    mean_loss = 0
    to_exit = False
    grad_mean = []

    start_time = datetime.now()
    while step < max_steps and not to_exit:
            print_if_0_rank(f"EPOCH {epoch}:")
            for train_source, train_targets in train_loader:
                train_targets = train_targets.reshape(-1, 1)
                
                with accelerator.accumulate(model):
                    optimizer.zero_grad()  # Reset gradients

                    output = model(train_source[0], train_source[1])

                    loss = criterion(output, train_targets)
                    mean_loss+=loss.item()
                    accelerator.backward(loss)

                    output = __unscale__(output.detach(), scaler)
                    train_targets = __unscale__(train_targets.detach(), scaler)

                    # Clip gradients and calculate the grad mean, if "plot_grads" is enabled
                    if accelerator.sync_gradients:
                        __print_if_debug__(f"{step}) CLIPPED THIS ITERATION", train_parameters)
                        
                        if "max_norm" in train_parameters:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=train_parameters["max_norm"])
                        
                        if "plot_grads" in train_parameters and train_parameters["plot_grads"]:
                            grad_mean = []
                            for name, module in model.named_modules():  # Iterate through all modules
                                if len(list(module.parameters())) > 0:  # Skip modules without parameters
                                    for param_name, param in module.named_parameters(recurse=False):
                                        if not param.grad is None:
                                            __print_if_debug__(f"Layer: {name}, Type: {type(module).__name__}, Parameter: {param_name}, Shape: {param.shape}, Max Grad: {param.grad.max()}, L2 Norm: {param.grad.data.norm(2)}", train_parameters)
                                            grad_mean.append(param.grad.detach().cpu().mean())
                            
                    r2_train.update(output, train_targets)
                    mae_train.update(output, train_targets)
                    pearson_r_train.update(output, train_targets)
                    del output
                    del train_targets
                    optimizer.step()
                    scheduler.step()

                    # To do every logging step.
                    # Note that the ``step`` variable indicates each iteration in the training loop, while for logging steps
                    # only steps which update the gradients are considered
                    if (step + 1) % (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"]) == 0:
                        # Compute metrics for validation set if "validate_while_training" setting is enabled
                        if ("validate_while_training" in train_parameters and train_parameters["validate_while_training"]):
                            __in_train_evaluate_model__(model, accelerator, val_dataset, state, scaler, step+1)
 
                        state["last_train_mae_score"] = state["train_mae_score"]
                        state["train_r2_score"] = r2_train.compute()
                        state["train_mae_score"] = mae_train.compute()
                        state["train_pearson_r_score"] = pearson_r_train.compute()
                        mae_train.reset()
                        r2_train.reset()
                        pearson_r_train.reset()

                        # check whether division by gradient_steps is necessary or done automatically
                        mean_loss /= (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"])
                        current_lr = optimizer.param_groups[0]["lr"]
                        print_if_0_rank(f"Step {step + 1} - LR: {current_lr:.6f}, Loss: {mean_loss:.4f}, R2: {state['train_r2_score']:.4f}, MAE: {state['train_mae_score']:.4f}, Pearson R: {state['train_pearson_r_score']:.4f}")
                        
                        if "plot_grads" in train_parameters and train_parameters["plot_grads"]:
                            plt.yscale("log")
                            plt.plot(grad_mean)
                            plt.show()

                        __print_if_debug__(f"########## STATE ###########\n{state}", train_parameters)    
                        to_exit = __update_exit_conditions__(mean_loss, train_parameters, state)

                        mean_loss = 0

                    step += 1

                    if step >= max_steps:
                        break
            epoch+=1

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print_if_0_rank(f"Training took {elapsed_time:.2f} seconds")
    if has_local:
        torch.distributed.barrier()
    if accelerator.is_main_process:
        __plot_metrics__(state)

    print_if_0_rank("\nTRAIN EVALUATION:")
    train_scores = evaluate(model, accelerator, train_dataset, scaler)
    print_if_0_rank(train_scores)
    print_if_0_rank("\nVALIDATION EVALUATION:")
    scores = evaluate(model, accelerator, val_dataset, scaler)

    # Evaluate best model
    print_if_0_rank("BEST MODEL EVALUATION:")
    print_if_0_rank("Best R^2 recoded:", state["best_r2"])
    best_scores = evaluate(state["best_model"], accelerator, val_dataset, scaler)
    #save_finetuned_model(state["best_model"], train_parameters, train_dataset, val_dataset, scaler, suffix="_best")
    
    accelerator.free_memory(optimizer, train_loader)
    accelerator.free_memory(scheduler)
    best_model = copy.deepcopy(state["best_model"])
    difference_train_validation = train_scores[0][1] - scores[0][1]

    del state
    del train_dataset
    del val_dataset
    del optimizer
    del train_loader
    del scheduler

    torch.cuda.empty_cache()

    return scores, model, best_scores, best_model, difference_train_validation

def __plot_metrics__(state):
    """
    Plots the validation metrics and saves them into a folder "plots" in current directory.

    Parameters:
        state (dict str -> float): state of the training process, used to memorize metrics and other important state variables
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Plot validation MAE
    axes[0].plot(state["val_maes"], label="Validation MAE")
    axes[0].set_title("Validation Mean Absolute Error (MAE)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].legend()

    # Plot validation R^2 scores
    axes[1].plot(state["val_r2s"], label="Validation R² Score", color="orange")
    axes[1].set_title("Validation R² Scores")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("R² Score")
    axes[1].legend()

    # Plot validation Pearson Coefficient
    axes[2].plot(state["val_pears"], label="Validation Pearson R Score", color="darkgreen")
    axes[2].set_title("Validation Pearson R Scores")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Pearson R Score")
    axes[2].legend()
    # Adjust layout
    plt.tight_layout()

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_name = f"{PLOT_DIR}/metricsplots_{current_time}.png"
    print(f"Saving plots to {fig_name}")
    plt.savefig(fig_name)
    plt.close(fig)
    plt.close('all')

def save_finetuned_model(finetune_model, train_parameters, train_dataset, val_dataset, scaler, suffix=""):
    """
        Saves the finetuned model, datasets and scaler to drive so that they can be easily loaded in any time.

        Parameters:
            finetune_model: huggingface finetuned model
            train_parameters (dict str -> obj): training parameters
            train_dataset (md.InterDataset): the train dataset used for finetuning.
            val_dataset (md.InterDataset): the validation dataset used for finetuning.
            scaler: scaler used for finetuning.
            suffix (str): suffix to add to the folder name the model will be saved in.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"./saves/{current_time}{suffix}"


    finetune_model.save_pretrained(save_folder, save_adapter=True, save_config=True)    
    with open(save_folder+'/train_parameters.pkl', 'wb') as f:
        pickle.dump(train_parameters, f)
    train_dataset.save(save_folder+"/train")
    val_dataset.save(save_folder+"/val")
    scaler.save(save_folder)

def load_finetuned_model(directory):
    """
        Loads a finetuned model, datasets and scaler from drive.

        Parameters:
            directory (str): directory containing all saved files.
        
        Returns:
            the finetuned model, an accelerator, the train_dataset, the val_dataset and the scaler used for the training of the model.
    """
    
    train_dataset = md.InterDataset.load(directory+"/train")
    val_dataset = md.InterDataset.load(directory+"/val")
    with open(directory+'/train_parameters.pkl', 'rb') as f:
        train_parameters = pickle.load(f)
    scaler = md.StdScaler()
    scaler.load(directory)
    accelerator, model = create_finetune_model(train_parameters, directory)
    return model, accelerator, train_dataset, val_dataset, scaler

def __prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot=False):
    """
        Preprocesses the training and validation datasets and returns them as instances of model.InterDataset.
        Tokenization of sequences and drugs and scaling of regression label is executed.

        Parameters:
            drug_tokenizer: tokenizer of drug structure model
            target_tokenizer: tokenizer of target sequence model
            train_X (pandas.DataFrame): dataframe containing training data used for prediction
            val_X (pandas.DataFrame): dataframe containing validation data used for performance evaluation
            train_y (List): list containing training labels to predict (dissociation constant)
            val_y (List): list containing validation labels to evaluate model (dissociation constant)
            plot (bool): if ``True``, plots an histogram of the training and validation data distribution, binned by label value
                Default: ``False``

        Returns:
            A pair containing the processed training dataset and the validation dataset
        
    """
    train_y = np.array(train_y)
    val_y = np.array(val_y)

    smiles = drug_tokenizer(train_X["SMILES"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
    targets = target_tokenizer(train_X["Target_RNA_sequence"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
        
    train_pkd = torch.Tensor(train_y.reshape(-1,1))
    if scaler:
        train_pkd = scaler.fit_transform(train_pkd).type(torch.float32)

    train_dataset = md.InterDataset(targets, smiles, train_pkd)

    smiles = drug_tokenizer(val_X["SMILES"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
    
    targets = target_tokenizer(val_X["Target_RNA_sequence"].tolist(),
                                padding="max_length", 
                                truncation=True, 
                                max_length=512,
                                return_tensors="pt")
    
    val_pkd =  torch.Tensor(val_y.reshape(-1,1))
    if scaler:
        val_pkd = scaler.transform(val_pkd).type(torch.float32)
    val_dataset = md.InterDataset(targets, smiles, val_pkd)
    
    if plot:
        # Define a common x-axis range based on your data
        common_range = (-4, 4)

        # Plotting overlapping histograms with proportions
        plt.figure(figsize=(8, 6))
        plt.hist(train_pkd.reshape(-1), bins=30, range=common_range, color='blue', alpha=0.5, edgecolor='black', label='train_pkd', density=True)
        plt.hist(val_pkd.reshape(-1), bins=30, range=common_range, color='red', alpha=0.5, edgecolor='black', label='val_pkd', density=True)

        # Adding titles and labels
        plt.title('Overlapping Distributions of train_pkd and val_pkd')
        plt.xlabel('Scaled Values')
        plt.ylabel('Proportion')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()
    
    return train_dataset, val_dataset


def crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode=False, path=None, use_best_scores=False):
    """
        Executes finetuning with crossvalidation.

        Parameters:
            X (pandas.DataFrame): DataFrame containing data used for prediction.
            y (List): List containing labels to predict for each interaction (dissociation constant).
            n_split (int): Number of splits for cross-validation.
            train_parameters (dict): Dictionary containing training parameters.
            scaler (object): Scaler object used for data normalization.
            classes (List): List of class labels.
            short_mode (bool): If True, crossvalidation is not complete, meaning that only half of crossvalidation rounds are applied. Default: False.
            path (str, optional): Path to save the model. 
                Default: None.
            use_best_scores (bool): If True, use the best scores from the finetuning process. 
                Default: False.

        Returns:
            The mean, the minimum and maximum validation scores of the runs, for each metric, and the mean difference between training and validation R2 scores.
    """
    has_local = "LOCAL_RANK" in os.environ
    all_scores = dict()
    i = 0

    differences = []
    for train_dataset, val_dataset in get_crossvalidate_datasets(X, y, n_split, scaler, classes):
        if (not short_mode) or (short_mode and i < n_split//2):    
            process = psutil.Process(os.getpid())  # Get current process
            mem_info = process.memory_info()  # Get memory usage details

            print_if_0_rank(f"RAM Usage: {mem_info.rss / 1024 ** 2:.2f} MB")  # Convert bytes to MB
            
                
            print_if_0_rank(f"Fold {i+1}: Creating finetuning_model...")
            accelerator, finetune_model = create_finetune_model(train_parameters, path)
            scores, finetune_model, best_scores, best_finetune_model, difference_train_validation = finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)
            differences.append(difference_train_validation)
            print_if_0_rank(f"Fold {i+1}: Finished finetuning and evaluation.")
            mem_info = process.memory_info()  # Get memory usage details
            print_if_0_rank(f"RAM Before cleaning: {mem_info.rss / 1024 ** 2:.2f} MB")  # Convert bytes to MB
            free_model(finetune_model, accelerator)
            free_model(best_finetune_model, accelerator)
            del accelerator

            mem_info = process.memory_info()  # Get memory usage details
            print_if_0_rank(f"RAM After cleaning: {mem_info.rss / 1024 ** 2:.2f} MB")  # Convert bytes to MB
            # This doesn't clean anything
            # There is a memory leak somewhere
            if use_best_scores and best_scores[0][1] > scores[0][1]:
                scores = best_scores

            for score in scores:
                if not score[0] in all_scores:
                    all_scores[score[0]] = []

                all_scores[score[0]].append(score[1])

            if has_local:
                torch.distributed.barrier()
            gc.collect()
            
        i+=1

    result = dict()
    for (metric, scores) in all_scores.items():
        result[metric] = ((sum(scores)/len(scores)), min(scores), max(scores))
    
    if has_local:
        torch.distributed.barrier()

    return result, np.mean(differences)
    
    

def get_crossvalidate_datasets(X, y, n_split, scaler, classes, plot=False):
    """
        Generate ``n_split`` crossvalidation datasets. If ``classes`` is not None, generated ``n_split`` folds stratified by classes.

        Parameters:
            X (pandas.DataFrame): dataframe containing data used for prediction
            y (List): list containing labels to predict for each interaction (dissociation constant)
            n_split (int): number of cross validation folds to split the data into
            scaler: scaler to apply on labels
            classes (List): classes by which to stratify the folds.
            plot (bool): if ``True``, plots an histogram of the training and validation data distribution, binned by label value
                Default: ``False``

        Yields:
            A pair of datasets obtained from the next crossvalidation split

    """
    skf = StratifiedKFold(n_splits=n_split, shuffle=True)
    target_tokenizer, drug_tokenizer = __get_tokenizers__()
    for i, (train_index, val_index) in enumerate(skf.split(X, classes)):
        train_X = X.iloc[train_index]
        val_X = X.iloc[val_index]
        train_y = y[train_index]
        val_y = y[val_index]


        train_dataset, val_dataset = __prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot)

        yield train_dataset, val_dataset

    
def evaluate(model, accelerator, dataset, scaler, metrics = [R2Score(), MeanAbsoluteError(), PearsonCorrCoef()]):
    """
        Evaluates the model on the specified dataset against a list of metric ``metrics``.

        Parameters:
            model: the model to evaluate
            accelerator: transformers library accelerator
            dataset (torch Dataset): Dataset containing the data to evaluate the model on
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation
            metrics: a list of metrics the model will be evaluated against
                Default: ``[R2Score(), MeanAbsoluteError(), PearsonCorrCoef()]``

            Returns:
                A list of pairs with metrics and their corresponding scores
    """

    for i in range(len(metrics)):
        metrics[i].reset()
        metrics[i] = accelerator.prepare(metrics[i])
    
    model.eval()

    val_loader = DataLoader(dataset, shuffle=True)
    print_if_0_rank("Number of validation data points:", len(val_loader))

    val_loader = accelerator.prepare(val_loader)    

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for source, targets in val_loader:
            targets = targets.to(model.device)
            output = model(source[0], source[1])
            output = __unscale__(output, scaler)
            targets = __unscale__(targets, scaler)
            for metric in metrics:
                metric.update(output, targets.reshape(-1,1))
            all_outputs.extend(output.cpu().numpy().ravel())
            all_targets.extend(targets.cpu().numpy().ravel())
    
            del output
            del targets

    all_local_pairs = torch.tensor(list(zip(all_targets, all_outputs)), device=model.device)
    all_pairs = accelerator.gather(all_local_pairs)
    all_pairs = all_pairs.cpu().tolist()
    
    all_pairs = sorted(all_pairs, key=lambda x: x[0])
    print_if_0_rank("Len pair:", len(all_pairs))
    sorted_targets, sorted_outputs = zip(*all_pairs)
    sorted_targets = list(sorted_targets)
    sorted_outputs = list(sorted_outputs)
    print_if_0_rank(len(sorted_targets), len(sorted_outputs))
    print_if_0_rank("####### EVALUATION METRICS ########")

    scores = []
    for metric in metrics:
        metric_score = metric.compute()
        ig_, ax_ = metric.plot()
        metric_name = str(metric)[:-2]
        scores.append((metric_name, metric_score.cpu()))

        print_if_0_rank(f"{metric_name}: {metric_score}")
        plt.close(ig_)

    plt.clf()  # Clears the current figure
    plt.plot(sorted_outputs, sorted_targets, 'o', label="Model Predictions")
    # Plot the red semi-transparent line
    plt.plot(sorted_targets, sorted_targets, color='red', linestyle='--', alpha=0.5, label="Perfect Calibration")

    # Add labels, legend, and title
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Calibration Plot")
    plt.legend()

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    fig = plt.gcf()  # Get the current figure
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"{PLOT_DIR}/calibration_{current_time}.png")
    plt.close(fig)
    return scores

def __get_tokenizers__():
    """
        Returns the tokenizers of the target and drug encoders.

        Returns:
            the tokenizers of the target and drug encoders.
    """
    target_tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer')
    drug_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    return target_tokenizer, drug_tokenizer

def load_RNABERTa(hidden_size=512, num_hidden_layers=12, num_attention_heads=16):
    """
        Initializes a RoBERTa target encoder given the hidden size, number of hidden layers and number of attention heads.
        This method should be employed only for finetuning, since it does not apply muParametrization.

        Parameters:
            hidden_size (int): hidden size of the model
                Default: 512
            num_hidden_layers (int): number of hidden layers in the model
                Default: 12
            num_attention_heads (int): number of attention heads in the model
                Default: 16
        Returns:
            the target encoder.
    """
    target_tokenizer, _ = __get_tokenizers__()
    configuration = RobertaConfig(vocab_size=len(target_tokenizer),
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,  
                                num_attention_heads=num_attention_heads,  
                                intermediate_size=3072,
                                max_position_embeddings=514,
                                attn_mult=(32**0.5),
                                output_hidden_states=True)

    target_encoder = RobertaForMaskedLM(configuration)
    return target_encoder    

def create_finetune_model(train_parameters, path, from_pretrained=None):
    """
            Load the pretrained models and prepare them for finetuning.
            LoRA is prepared for finetuning and LoftQ is applied for the target encoder.

            Parameters:
                train_parameters (dict str -> obj): training parameters
                path (str): path to the pretrained model
                from_pretrained (str): path of the pretrained model to load, if available. If not, create a new model. 
                    Default: None
            Returns:
                a transformers accelerator and the model to be trained.

    """

    torch.cuda.empty_cache()

    drug_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    target_encoder = RobertaForMaskedLM.from_pretrained(path).roberta

    print("Models loaded!")
    torch.cuda.empty_cache()


    #print_trainable_parameters("Target encoder:", target_encoder)
    #print_trainable_parameters("Drug encoder:", drug_encoder)

    if not from_pretrained:
        config = md.InteractionModelATTNConfig(train_parameters["model_dropout"])
        model = md.InteractionModelATTNForRegression(config, target_encoder, drug_encoder)

    else:
        model = md.InteractionModelATTNForRegression.from_pretrained(from_pretrained, target_encoder, drug_encoder)
    deepspeed_plugin = None

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=train_parameters["gradient_accumulation_steps"], 
                              kwargs_handlers=[kwargs])

    return accelerator, model

def free_model(model, accelerator=None):
    """
        Frees the model from GPU and RAM memory.
    """
    
    for parameters in model.parameters():
        parameters.grad = None
        del parameters

    if hasattr(model, "module"):
        model = model.module 
        
    if accelerator:
        accelerator.free_memory(model)

    del model.model.target_encoder
    del model.model.drug_encoder
    del model.model
    del model

    torch.cuda.empty_cache()
    gc.collect()
    
def make_bsh(filename=None):
    """
        Creates the base shapes for the model to allow MuParametrization.
    """
    tokenizer, _ = __get_tokenizers__()
    base_config = RobertaConfig(vocab_size=len(tokenizer),
                                hidden_size=512,
                                num_hidden_layers=12,  
                                num_attention_heads=16,  
                                intermediate_size=3072,
                                max_position_embeddings=514,
                                attn_mult=(32**0.5),
                                output_hidden_states=True)
    base_model = RobertaForMaskedLM(config=base_config)
    # define a delta models where we vary all "widths" we want to vary

    delta_config = RobertaConfig(vocab_size=len(tokenizer),
                                    hidden_size=128,
                                    num_hidden_layers=12,  
                                    num_attention_heads=16,  
                                    intermediate_size=512,
                                    max_position_embeddings=514,
                                    attn_mult=(32**0.5),
                                    output_hidden_states=True)
    delta_model = RobertaForMaskedLM(config=delta_config)
    base_shapes = make_base_shapes(base_model, delta_model, savefile=filename)
    return base_shapes

def finetune_objective(trial, X, y, n_split, scaler, classes, short_mode, path):
    
    has_local = "LOCAL_RANK" in os.environ
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))  # Assuming world size defaults to 1

    train_parameters = {
        "train_batch_size": 32//world_size,
        "device": "cuda",
        "validate_while_training": True,
        "gradient_accumulation_steps": 1,
    }

    gas = train_parameters["gradient_accumulation_steps"]
    # Let rank 0 suggest the hyperparameters and then broadcast them.
    if rank == 0:
        suggested_learning_rate = trial.suggest_float("learning_rate", 4e-6, 5e-4, log=True) * np.sqrt(gas * train_parameters["train_batch_size"]*world_size)
        suggested_weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
        suggested_model_dropout = trial.suggest_categorical("model_dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
        suggested_num_cycles = trial.suggest_categorical("num_cycles", [0, 1/6, 0.25])

        print_if_0_rank(f"Suggested learning rate: {suggested_learning_rate}\nSuggested weight decay: {suggested_weight_decay}\nSuggested model dropout: {suggested_model_dropout}\nSuggested num cycles: {suggested_num_cycles}")
        # Pack all suggestions into a dictionary
        hyperparams = {
            "learning_rate": suggested_learning_rate,
            "weight_decay": suggested_weight_decay,
            "model_dropout": suggested_model_dropout,
            "num_cycles": suggested_num_cycles
        }
    else:
        hyperparams = {}
    
    # Synchronize all processes here so that rank 0 has finished suggestion.
    if has_local:
        torch.distributed.barrier()
    # Broadcast the hyperparameters dictionary from rank 0 to all other processes.
    # Note: Broadcasting a Python object can be done with torch.distributed.broadcast_object_list.
    if has_local:
        hyperparams_list = [hyperparams]
        torch.distributed.broadcast_object_list(hyperparams_list, src=0)
        print("WE ARE ALL HERE")
        hyperparams = hyperparams_list[0]
    
    print("Preparing parameters...")
    # Now update your train_parameters using the received hyperparams.
    train_parameters = {
        "train_batch_size": 32//world_size,
        "device": "cuda",
        "validate_while_training": True,
        "gradient_accumulation_steps": 1,
    }
    train_parameters.update({
        "learning_rate": hyperparams["learning_rate"],
        "adam_epsilon": 1e-6,
        "num_epochs": 50,
        "log_performance_every": 40,
        "weight_decay": hyperparams["weight_decay"],
        "model_dropout": hyperparams["model_dropout"],
        "max_norm": 1,
        "num_cycles": hyperparams["num_cycles"]
    })

    results, difference = crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode, path=path)


    plt.close('all')  # Close all open figures
    gc.collect()  # Force garbage collection
    return np.mean(results["R2Score"][:2]), np.mean([results["MeanAbsoluteError"][0], results["MeanAbsoluteError"][2]]), difference


def finetune_optimize(X, y, n_split, scaler, classes, short_mode, path=None):
    """
        Use Optuna TPESampler to optimize finetuning parameters.
        
        Parameters:
            X (pandas.DataFrame): dataframe containing data used for prediction
            y (List): list containing labels to predict for each interaction (dissociation constant)
            n_split (int): number of cross validation folds to split the data into
            scaler: scaler to apply on labels
            classes (List): classes by which to stratify the folds.
            short_mode (bool): if True, crossvalidation is not complete, meaning that only half of crossvalidation rounds are applied.
                Default: False
            path (str): path to the pretrained model
                Default: None
        Returns:
            The best hyperparameters found by the optimization.
    """
    n_trials = 50
    has_local = "LOCAL_RANK" in os.environ 
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))  # Assuming world size defaults to 1
    if has_local:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    print_if_0_rank("WORLD SIZE:", world_size)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = f"study_results/{now}"
    
    storage = f"sqlite:///finetuning_{now}.db"

    if rank == 0:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=(world_size+1)*14),
            study_name="finetuning_parameter_selection",
            load_if_exists=True,
            storage=storage)
        print("Starting optimization...")

    if has_local:
        torch.distributed.barrier()  # Ensure rank 0 finishes creating the study

    #study = optuna.load_study(
    #    study_name="finetuning_parameter_selection",
    #    storage=storage
    #)

    if has_local:
        torch.distributed.barrier()  # Optional: synchronize after loading the study

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print("Starting optimization...")
    if rank == 0:
        study.optimize(lambda trial: finetune_objective(trial, X, y, n_split, scaler, classes, short_mode, path), n_trials=n_trials, n_jobs=1)
    else:
        for i in range(n_trials):
            finetune_objective(0, X, y, n_split, scaler, classes, short_mode, path)
    if has_local:
        torch.distributed.barrier()

    if rank == 0:
        save_study_plot(study, result_path, True)
        
        with open(f"{result_path}/finetune_best_params.txt", "w") as f:
            for trial in study.best_trials:
                f.write(f"Trial number: {trial.number}\n")
                for key, value in trial.params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

    if has_local:
        torch.distributed.destroy_process_group()


def optuna_hp_space(trial):
    """
        Defines the hyperparameter space for the Optuna optimization.
        
        Parameters:
            trial: Optuna trial object
        Returns:
            A dictionary containing the hyperparameters to optimize.
    """
    train_parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.2),
    }
    return train_parameters

def pretraining_model_init(trial, full=False):
    """
        Helper function passed to the MuTrainer to initialize a new model.
        
        Parameters:
            trial: Optuna trial object
        Returns:
            A new model with the specified configuration.
    """
    target_tokenizer, _ = __get_tokenizers__()
    if not full:
        target_config = mu.RobertaConfig(vocab_size=len(target_tokenizer),
                                    hidden_size=128,
                                    num_hidden_layers=12,
                                    num_attention_heads=16,  
                                    intermediate_size=1024,
                                    max_position_embeddings=514,
                                    attn_mult=(32**0.5),
                                    output_hidden_states=True)
    else:
        target_config = RobertaConfig(vocab_size=len(target_tokenizer),
                                    hidden_size=512,
                                    num_hidden_layers=12,  
                                    num_attention_heads=16,  
                                    intermediate_size=3072,
                                    max_position_embeddings=514,
                                    attn_mult=(32**0.5),
                                    output_hidden_states=True)   
        
    target_model = mu.RobertaForMaskedLM(config=target_config)
    print(f"Number of parameters in target_model: {sum(p.numel() for p in target_model.parameters())}")
    set_base_shapes(target_model, "roberta512.bsh")

    target_model.apply(target_model._init_weights)
    return target_model

def compute_objective(metrics):
    """
        Given a metric dictionary, computes the objective value for the optimization.

        Parameters:
            metrics: dictionary containing the evaluation metrics of the model
    """
    return metrics["eval_loss"]/math.log(2)

def pretrain_optimize(path):
    """
        Use Optuna TPESampler to optimize pretraining parameters.
        The "RANK" environment variable is used to allow for multi-GPU training.
        MuParametrization is used.
        Saves a series of plots from the Optuna study and the best parameters in a text file.
        
        Parameters:
            path: path where to save or load the tokenized pretraining datasets.
    """
    
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.getenv("RANK", 0))

    target_tokenizer, _ = __get_tokenizers__()

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = f"study_results/{now}"

    if rank == 0:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    
    train_datapath = f'{path}/processed/dataset/train/train.txt'
    val_datapath = f'{path}/processed/dataset/test/test.txt'
    datasets = load_pretrain_data(path, train_datapath, val_datapath, target_tokenizer) 
    data_collator = DataCollatorForLanguageModeling(tokenizer=target_tokenizer, mlm=True, mlm_probability=0.15)

    parser = HfArgumentParser((TrainingArguments, md.ModelArgs,))
    print_if_0_rank("WORLD SIZE:", os.environ["WORLD_SIZE"])

    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--logging_steps', '200',
        '--save_strategy', 'no',
        '--per_device_eval_batch_size', '64',
        '--per_device_train_batch_size', '32',
        '--gradient_accumulation_steps', str(8//int(os.environ["WORLD_SIZE"])),
        '--do_train',
        '--do_eval',
        '--max_steps', '5000',
        '--dataloader_pin_memory', 'True',      
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--ddp_find_unused_parameters', 'False'
    ])

    init = lambda x: pretraining_model_init(x, False)

    trainer = MuTrainer(
        model=None,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        model_init=init,
        data_collator=data_collator
    )

    storage = "sqlite:///pretraining.db"
    est_run = trainer.hyperparameter_search(hp_space=optuna_hp_space, direction="minimize", backend="optuna", compute_objective=compute_objective, n_trials=32, study_name="pretraining_parameter_selection", storage=storage)
    
    if rank == 0:
        study = optuna.load_study(study_name="pretraining_parameter_selection", storage=storage)
        save_study_plot(study, result_path)

        with open(f"{result_path}/pretrain_best_params.txt", "w") as f:
            for key, value in study.best_params.items():
               f.write(f"{key}: {value}\n")

        optuna.delete_study(study_name="pretraining_parameter_selection", storage=storage)

def save_study_plot(study, path, is_multiobjective=False):
    """
    Saves a series of plots from the Optuna study.

    For single-objective studies, the following plots are saved:
        - Parameter Importances
        - Parallel Coordinate
        - Contour
        - Slice

    For multiobjective studies, a Pareto front plot is saved and parameter
    importances and contour plots are computed separately for each objective.

    Parameters:
        study: Optuna study object
        path (str): path where to save the plots
        is_multiobjective (bool): Flag indicating if the study is multi-objective
            Default: False

    Assumes that in a multiobjective study the first metric is "R2score"
    and the second metric is "MAE".
    """
    if is_multiobjective:
        # Define metric names based on the order of objectives.
        metric_names = ["R2score", "MAE", "R2Difference"]

        # Plot and save the Pareto front.
        fig = optuna.visualization.plot_pareto_front(study, target_names=metric_names)
        fig.update_layout(width=1400, height=1300)
        fig.write_image(f"{path}/pareto_front.png")

        # Plot parameter importances and contour plots for each metric.
        for i, metric_name in enumerate(metric_names):
            # Parameter importances plot for the metric.
            fig = optuna.visualization.plot_param_importances(
                study, target=lambda t, i=i: t.values[i], target_name=metric_name
            )
            fig.update_layout(width=800, height=800)
            fig.write_image(f"{path}/param_importances_{metric_name}.png")

            # Contour plot for the metric.
            fig = optuna.visualization.plot_contour(
                study, target=lambda t, i=i: t.values[i], target_name=metric_name
            )
            fig.update_layout(width=1600, height=1600)
            fig.write_image(f"{path}/contour_{metric_name}.png")
    else:
        # Single-objective plots
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(width=800, height=800)
        fig.write_image(f"{path}/param_importances.png")

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.update_layout(width=800, height=800)
        fig.write_image(f"{path}/parallel_coordinate.png")

        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(width=800, height=800)
        fig.write_image(f"{path}/contour.png")

        fig = optuna.visualization.plot_slice(study)
        fig.update_layout(width=800, height=800)
        fig.write_image(f"{path}/slice.png")

    print(f"Study plots saved in {path}!")