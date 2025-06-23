
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing
from torchmetrics.regression import R2Score, MeanAbsoluteError, PearsonCorrCoef
from datetime import datetime

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer, RobertaModel, RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, AutoConfig, AutoModel
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
from typing import Optional
from torch.overrides import has_torch_function_variadic, handle_torch_function
from sklearn.utils.class_weight import compute_class_weight
from model import ChembertaTokenizer

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
import warnings
import torch.nn._reduction as _Reduction

creation_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')
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


def __in_train_evaluate_model__(model, accelerator, val_dataset, state, thread_id):
    """
        Evaluates the model on the validation data. 

        Parameters:
            model: the model to be evaluated
            accelerator: transformers accelerator used in training the model
            val_dataset (torch Dataset): the validation dataset
            state (dict str -> float): state of the training process, used to memorize metrics and other important state variables
            thread_id (str): name of the thread
    """
    with torch.no_grad():
        has_local = "LOCAL_RANK" in os.environ
        model.eval() 
        val_loader = DataLoader(val_dataset, shuffle=True)
        val_loader = accelerator.prepare(val_loader)

        r2_metric = R2Score().to(model.device)
        mae_metric = MeanAbsoluteError().to(model.device)
        pearson_r_metric = PearsonCorrCoef().to(model.device)
        r2_metric, mae_metric, pearson_r_metric = accelerator.prepare(r2_metric, mae_metric, pearson_r_metric)

        eval_n = 0

        for source, targets in val_loader:
            targets = targets.reshape(-1, 1).to(model.device)
            output = model(source[0], source[1])
                
            model_to_call = model.module if hasattr(model, "module") else model
            targets = model_to_call.unscale(targets)
            output = model_to_call.unscale(output)

            r2_metric.update(output, targets)
            mae_metric.update(output, targets)
            pearson_r_metric.update(output, targets)

            eval_n += 1

        model.train()

        state["last_valid_mae_score"] = state["valid_mae_score"]
        if has_local:
            torch.distributed.barrier()

        state["valid_r2_score"] = r2_metric.compute()
        state["valid_mae_score"] = mae_metric.compute()
        state["valid_pearson_r_score"] = pearson_r_metric.compute()

        state["val_maes"].append(state["valid_mae_score"].cpu())
        state["val_r2s"].append(state["valid_r2_score"].cpu())
        state["val_pears"].append(state["valid_pearson_r_score"].cpu())
        
        accelerator.free_memory(r2_metric, mae_metric, pearson_r_metric)
        del r2_metric
        del mae_metric
        del pearson_r_metric
        print_if_0_rank(f"Epoch {thread_id} - R2 val: {state['valid_r2_score']:.4f}, MAE val: {state['valid_mae_score']:.4f}, Pearson R val: {state['valid_pearson_r_score']:.4f}")
        
    
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

    classes = inters["Category"].values

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


def finetune_and_evaluate(model, accelerator, train_parameters, train_dataset, val_dataset, save_result=False):
    """
    Fine-tunes the interaction model and evaluates performance on training and validation datasets.

    Trains the model for a specified number of epochs, logging training metrics (R², MAE, Pearson R),
    optionally validating during training, plotting gradient norms in debug mode, and saving the final model.

    Parameters:
        model (nn.Module): The interaction model to fine-tune.
        accelerator (Accelerator): HuggingFace Accelerator for distributed or mixed‐precision training.
        train_parameters (dict): Training configuration, must include:
            - "num_epochs" (int): Number of training epochs.
            - "learning_rate" (float): Optimizer learning rate.
            - "weight_decay" (float): Optimizer weight decay.
            - "train_batch_size" (int): Batch size for training.
            - "gradient_accumulation_steps" (int): Steps to accumulate gradients before optimizer step.
            - Optional "decay_ratio" (float): If <0.99, used to schedule learning‐rate decay.
            - Optional "target_epochs" (int): Epoch count at which decay calculation is anchored.
            - Optional "validate_while_training" (bool): If True, runs validation at end of each epoch.
            - Optional "debug" (bool): If True, plots per‐parameter gradient norms each epoch.
        train_dataset (InterDataset): Dataset for training.
        val_dataset (InterDataset): Dataset for validation.
        save_result (bool): If True and on main process, saves the fine‐tuned model after training.
            Default: False

    Returns:
        tuple:
            - scores (tuple): Evaluation metrics (R², MAE, Pearson R) on the validation or training set.
            - model (nn.Module): The fine‐tuned model instance.
            - None: Placeholder for backward‐compatibility.
            - None: Placeholder for backward‐compatibility.
            - float: Difference between validation MAE and training MAE (validation minus training).
    """
        
    has_local = "LOCAL_RANK" in os.environ
    r2_train = R2Score()
    mae_train = MeanAbsoluteError()
    pearson_r_train = PearsonCorrCoef()

    n_epochs = train_parameters["num_epochs"]

    optimizer = AdamW(model.parameters(), lr=train_parameters["learning_rate"], weight_decay=train_parameters["weight_decay"])
    train_loader = DataLoader(train_dataset, batch_size=train_parameters["train_batch_size"], shuffle=True)
    scheduler = None

    decay_ratio = 1 if "decay_ratio" not in train_parameters else 1/train_parameters["decay_ratio"]

    if decay_ratio < 0.99:
        steps_per_epoch = (len(train_loader) * n_epochs ) * train_parameters["gradient_accumulation_steps"] / n_epochs
        sched_max_step = int(train_parameters["target_epochs"] * steps_per_epoch / (1 - decay_ratio) + 0.5)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=sched_max_step
            )
        
        scheduler = accelerator.prepare(scheduler)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    r2_train, mae_train, pearson_r_train = accelerator.prepare(r2_train, mae_train, pearson_r_train)

    model.train()
    criterion = mse_loss
    grad_norm_history = []
    start_time = datetime.now()
    state = {"valid_mae_score":0, "valid_r2_score":0, "valid_pearson_r_score": 0, "val_maes":[], "val_r2s":[], "val_pears":[]}

    for i in range(n_epochs):
        mean_loss = 0
        steps = 0

        for train_source, train_targets in train_loader:
            train_targets = train_targets.reshape(-1, 1)
                
            with accelerator.accumulate(model):
                output = model(train_source[0], train_source[1])
         
                loss = criterion(output, train_targets, weight=train_source[2].reshape(-1, 1))

                mean_loss+=loss.item()
                accelerator.backward(loss)

                model_to_call = model.module if hasattr(model, "module") else model
                output = model_to_call.unscale(output.detach())
                train_targets = model_to_call.unscale(train_targets.detach())
                
                r2_train.update(output, train_targets)
                mae_train.update(output, train_targets)
                pearson_r_train.update(output, train_targets)

                del output
                del train_targets

                optimizer.step()

                if has_local:
                    mean_loss_tensor = torch.tensor(mean_loss, device=model.device)
                    mean_loss = accelerator.gather(mean_loss_tensor).mean().item()
                    grad_norm_history = update_grad_norm_history(model, grad_norm_history, i)

                if ("target_epochs" in train_parameters and i <= train_parameters["target_epochs"]) or (not "target_epochs" in train_parameters):
                    if scheduler:
                        scheduler.step()
       
                optimizer.zero_grad()

                steps += 1

  
        mean_loss = mean_loss/steps 
        current_lr = optimizer.param_groups[0]["lr"]

        r2 = r2_train.compute()
        mae = mae_train.compute()
        rscore = pearson_r_train.compute()
        total_norm = torch.norm(torch.tensor(list(grad_norm_history[-1].values())), 2)
        print_if_0_rank(f"Epoch {i + 1} - LR: {current_lr:.6f}, Loss: {mean_loss:.4f}, Grad L2 norm: {total_norm}, R2: {r2:.4f}, MAE: {mae:.4f}, Pearson R: {rscore:.4f}")
        if "validate_while_training" in train_parameters and train_parameters["validate_while_training"]:
            __in_train_evaluate_model__(model, accelerator, val_dataset, state, i+1)

        if has_local:
            torch.distributed.barrier()

        mae_train.reset()
        r2_train.reset()
        pearson_r_train.reset()

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print_if_0_rank(f"Training took {elapsed_time:.2f} seconds")
    if has_local:
        torch.distributed.barrier()

    print_if_0_rank("\nTRAIN EVALUATION:")
    train_scores = evaluate(model, accelerator, train_dataset, train=True)

    train_scores = evaluate(model, accelerator, train_dataset)
   
    difference_validation_train = 0
    scores = train_scores

    if "validate_while_training" in train_parameters and train_parameters["validate_while_training"]:
        print_if_0_rank("\nVALIDATION EVALUATION:")
        scores = evaluate(model, accelerator, val_dataset, train=True)

        scores = evaluate(model, accelerator, val_dataset)
        difference_validation_train = scores[1][1] - train_scores[1][1]

          
    if "debug" in train_parameters and os.environ.get("RANK", "0") == "0":
        param_names = list(next(iter(grad_norm_history.values())).keys())

        # Create a plot for each parameter
        for name in param_names:
            epochs = list(grad_norm_history.keys())
            norms = [grad_norm_history[epoch][name] for epoch in epochs]
            
            plt.figure()
            plt.plot(epochs, norms, marker='o', linestyle='-')
            plt.title(f"Gradient Norm for {name}")
            plt.xlabel("Epoch")
            plt.ylabel("L2 Gradient Norm")
            plt.grid(True)
            plot_filename = os.path.join(PLOT_DIR, f"{name.replace('.', '_')}_grad_norm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved plot for {name} to {plot_filename}")
    
    if save_result and accelerator.is_main_process:
        save_finetuned_model(model.cpu(), train_parameters, train_dataset, val_dataset, suffix="")
        
    if accelerator.is_main_process and "validate_while_training" in train_parameters and train_parameters["validate_while_training"]:
        __plot_metrics__(state)

    accelerator.free_memory(optimizer, train_loader)
    if scheduler:
        accelerator.free_memory(scheduler)
        del scheduler
  
    torch.cuda.empty_cache()
    return scores, model, None, None, difference_validation_train

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

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    fig_name = f"{PLOT_DIR}/metricsplots_{current_time}.png"
    print(f"Saving plots to {fig_name}")
    plt.savefig(fig_name)
    plt.close(fig)
    plt.close('all')

def save_finetuned_model(finetune_model, train_parameters, train_dataset, val_dataset, suffix=""):
    """
        Saves the finetuned model, datasets and scaler to drive so that they can be easily loaded in any time.

        Parameters:
            finetune_model: huggingface finetuned model
            train_parameters (dict str -> obj): training parameters
            train_dataset (md.InterDataset): the train dataset used for finetuning.
            val_dataset (md.InterDataset): the validation dataset used for finetuning.
            suffix (str): suffix to add to the folder name the model will be saved in.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"./saves/{current_time}{suffix}"

    model_to_call = finetune_model.module if hasattr(finetune_model, "module") else finetune_model

    model_to_call.save_pretrained(save_folder, save_config=True)    
    scaler = model_to_call.model.scaler
    with open(save_folder+'/train_parameters.pkl', 'wb') as f:
        pickle.dump(train_parameters, f)
    train_dataset.save(save_folder+"/train")
    val_dataset.save(save_folder+"/val")
    scaler.save(save_folder)

def update_grad_norm_history(model, grad_norm_history, epoch):
    """
    Updates the gradient norm history for each parameter in the model.

    Parameters:
        model: The model whose gradients are being tracked.
        grad_norm_history (list): A list of dictionaries containing gradient norms for each epoch.
        epoch (int): The current epoch number.

    Returns:
        Updated grad_norm_history.
    """
    with torch.no_grad():
        epoch_grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm(2).item()
                epoch_grad_norms[name] = norm
        if epoch >= len(grad_norm_history):
            grad_norm_history.append(epoch_grad_norms)
        else:
            grad_norm_history[epoch] = epoch_grad_norms
    return grad_norm_history

# Pytorch's mse_loss function
def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

    Measures the element-wise mean squared error, with optional weighting.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (use reduction).
        reduce (bool, optional): Deprecated (use reduction).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Mean Squared Error loss (optionally weighted).
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            mse_loss,
            (input, target, weight),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            weight=weight,
        )

    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)

    if weight is not None:
        if weight.size() != input.size():
            raise ValueError("Weights and input must have the same size.")

        # Perform weighted MSE loss manually
        squared_errors = torch.pow(expanded_input - expanded_target, 2)
        weighted_squared_errors = squared_errors * weight

        if reduction == "none":
            return weighted_squared_errors
        elif reduction == "sum":
            return torch.sum(weighted_squared_errors)
        elif reduction == "mean":
            return torch.sum(weighted_squared_errors) / torch.sum(weight)
        else:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
            )
    else:
        return torch._C._nn.mse_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction)
        )
    

def load_finetuned_model(path, directory, train=False):
    
    train_dataset = md.InterDataset.load(directory+"/train")
    val_dataset = md.InterDataset.load(directory+"/val")
    with open(directory+'/train_parameters.pkl', 'rb') as f:
        train_parameters = pickle.load(f)
    scaler = md.StdScaler()
    scaler.load(directory)
    accelerator, model = create_finetune_model(train_parameters, path, scaler, load_weights=None, pretrained=directory)
    if not train:
        del accelerator
        accelerator = None 
        model = model.eval()
    return model, accelerator, train_dataset, val_dataset, scaler

def __prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot=False, weights=None):
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
            scaler (object): Scaler object used for data normalization.
            plot (bool): if ``True``, plots an histogram of the training and validation data distribution, binned by label value
                Default: ``False``
            weights (List[int]): Class labels for each training example, used to compute balanced sample weights.
                Default: ``False``

        Returns:
            A pair containing the processed training dataset and the validation dataset
        
    """
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    if weights is not None:
        uniq_classes = np.unique(weights)
        class_weights = compute_class_weight(class_weight="balanced",
                                            classes=uniq_classes,
                                            y=weights)
        label_to_id = {label: i for i, label in enumerate(uniq_classes)}
        train_classes = [class_weights[label_to_id[cl]] for cl in weights]
    else:
        train_classes = None
    
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

    train_dataset = md.InterDataset(targets, smiles, train_pkd, train_classes)
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


def crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode=False, path=None, use_best_scores=False, compute_weights=False):
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
                Default: ``None``.
            use_best_scores (bool): If True, use the best scores from the finetuning process. 
                Default: ``False``.
            compute_weights (bool): If True, compute class weights for the training dataset.
                Default: ``False``

        Returns:
            The mean, the minimum and maximum validation scores of the runs, for each metric, and the mean difference between validation and training MAE scores.
    """
    has_local = "LOCAL_RANK" in os.environ
    all_scores = dict()
    i = 0

    differences = []
    for train_dataset, val_dataset in get_crossvalidate_datasets(X, y, n_split, scaler, classes, compute_weights=compute_weights):
        if (not short_mode) or (short_mode and i < n_split//2):    
            process = psutil.Process(os.getpid())  # Get current process
            mem_info = process.memory_info()  # Get memory usage details

            print_if_0_rank(f"RAM Usage: {mem_info.rss / 1024 ** 2:.2f} MB")  # Convert bytes to MB
            
                
            print_if_0_rank(f"Fold {i+1}: Creating finetuning_model...")
            accelerator, finetune_model = create_finetune_model(train_parameters, path, scaler)
            scores, finetune_model, best_scores, best_finetune_model, difference_validation_train = finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, False)
            differences.append(difference_validation_train)
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
    
    

def get_crossvalidate_datasets(X, y, n_split, scaler, classes, plot=False, compute_weights=False):
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
        if compute_weights == False:
            w_classes = None
        else:   
            w_classes = classes[train_index]

        train_dataset, val_dataset = __prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot, w_classes)

        yield train_dataset, val_dataset

    
def evaluate(model, accelerator, dataset, metrics = [R2Score(), MeanAbsoluteError(), PearsonCorrCoef()], train=False):
    """
        Evaluates the model on the specified dataset against a list of metric ``metrics``.

        Parameters:
            model: the model to evaluate
            accelerator: transformers library accelerator
            dataset (torch Dataset): Dataset containing the data to evaluate the model on
            metrics: a list of metrics the model will be evaluated against
                Default: ``[R2Score(), MeanAbsoluteError(), PearsonCorrCoef()]``
            train (bool): if ``True``, the model will be set to training mode, otherwise it will be set to evaluation mode
                Default: ``False``

            Returns:
                A list of pairs with metrics and their corresponding scores
    """

    for i in range(len(metrics)):
        metrics[i].reset()
        metrics[i] = accelerator.prepare(metrics[i])
    
    if train:  
        model.train()
    else:
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

            model_to_call = model.module if hasattr(model, "module") else model
            output = model_to_call.unscale(output)
            targets = model_to_call.unscale(targets)

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
    plt.plot(sorted_targets, sorted_targets, color='red', linestyle='--', alpha=0.5, label="Perfect Prediction")

    # Add labels, legend, and title
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Predicted vs. Actual Plot")
    plt.legend()

    fig = plt.gcf()  # Get the current figure
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    fig.savefig(f"{PLOT_DIR}/prediction_{current_time}.png")
    plt.close(fig)

    model.train()

    return scores

def __get_tokenizers__():
    """
        Returns the tokenizers of the target and drug encoders.

        Returns:
            the tokenizers of the target and drug encoders.
    """
    target_tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer')
    drug_tokenizer = ChembertaTokenizer("./chemberta/vocab.json")

    return target_tokenizer, drug_tokenizer

def create_finetune_model(train_parameters, path, scaler=None, load_weights=None, pretrained=None):
    """
    Loads pretrained encoders and prepares the full interaction model for finetuning.

    The model combines a target encoder and a drug encoder into a joint architecture,
    and optionally loads previously saved weights or pretrained model parameters.

    Parameters:
        train_parameters (dict str -> obj): Dictionary containing training configuration parameters such as dropout rates and gradient accumulation steps.
        path (str): Path to the pretrained target encoder (e.g., a HuggingFace model directory).
        scaler: Optional scaler used for label normalization during training.
            Default: ``None``
        load_weights (str): Path to saved model weights to load. If ``None``, a new model instance is initialized.
            Default: ``None``
        pretrained (str): Path to a previously saved full model (including drug and target encoders).
            Default: ``None``

    Returns:
        Tuple[Accelerator, nn.Module]: A HuggingFace ``Accelerator`` for distributed training and the initialized model ready for finetuning.
    """

    torch.cuda.empty_cache()

    drug_encoder_config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    drug_encoder_config.attention_probs_dropout_prob = train_parameters["attention_dropout"]
    drug_encoder_config.hidden_dropout_prob = train_parameters["hidden_dropout"]
    drug_encoder_config.pooler = None
    drug_encoder = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", config=drug_encoder_config, add_pooling_layer=False)
    drug_encoder = RobertaModel(config=drug_encoder_config, add_pooling_layer=False)

    target_encoder_config = AutoConfig.from_pretrained(path)
    target_encoder_config.attention_probs_dropout_prob = train_parameters["attention_dropout"]
    target_encoder_config.hidden_dropout_prob = train_parameters["hidden_dropout"]
    target_encoder = RobertaModel.from_pretrained(path, config=target_encoder_config)

    print("Models loaded!")
    torch.cuda.empty_cache()

    if not pretrained:
        config = md.InteractionModelATTNConfig(train_parameters["attention_dropout"], train_parameters["hidden_dropout"])
        model = md.InteractionModelATTNForRegression(config, target_encoder, drug_encoder, scaler)
    else: 
        config = md.InteractionModelATTNConfig.from_pretrained(pretrained)
        model = md.InteractionModelATTNForRegression.from_pretrained(pretrained, target_encoder=target_encoder, drug_encoder=drug_encoder, scaler=scaler)

    if load_weights:
        model.load_state_dict(torch.load(load_weights))
        
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=train_parameters["gradient_accumulation_steps"], kwargs_handlers=[kwargs])

    return accelerator, model

def free_model(model, accelerator=None):
    """
        Frees the model from GPU and RAM memory.
    """
    if model:
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
    base_config = mu.RobertaConfig(vocab_size=len(tokenizer),
                                hidden_size=512,
                                num_hidden_layers=12,  
                                num_attention_heads=16,  
                                intermediate_size=3072,
                                max_position_embeddings=514,
                                attn_mult=(32**0.5),
                                output_hidden_states=True)
    base_model = mu.RobertaForMaskedLM(config=base_config)
    # define a delta models where we vary all "widths" we want to vary

    delta_config = mu.RobertaConfig(vocab_size=len(tokenizer),
                                    hidden_size=128,
                                    num_hidden_layers=12,  
                                    num_attention_heads=16,  
                                    intermediate_size=512,
                                    max_position_embeddings=514,
                                    attn_mult=(32**0.5),
                                    output_hidden_states=True)
    delta_model = mu.RobertaForMaskedLM(config=delta_config)
    base_shapes = make_base_shapes(base_model, delta_model, savefile=filename)
    return base_shapes

def finetune_objective(trial, X, y, n_split, scaler, classes, short_mode, path, compute_weights=False):
    
    has_local = "LOCAL_RANK" in os.environ
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))  # Assuming world size defaults to 1

    train_parameters = {
        "train_batch_size": 32//world_size,
        "device": "cuda",
        "validate_while_training": True,
        "gradient_accumulation_steps": 1,
    }

    # Let rank 0 suggest the hyperparameters and then broadcast them.
    if rank == 0:
        suggested_learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4)
        suggested_weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01, log=True)
        suggested_hidden_dropout = trial.suggest_float("hidden_dropout", 0, 0.4)
        suggested_attention_dropout = trial.suggest_float("attention_dropout", 0, 0.5)
        suggested_decay_ratio = trial.suggest_int("decay_ratio", 1, 4)
        print_if_0_rank(f"Suggested learning rate: {suggested_learning_rate}\nSuggested weight decay: {suggested_weight_decay}\nSuggested decay ratio: {suggested_decay_ratio}")
        print_if_0_rank(f"Suggested hidden dropout: {suggested_hidden_dropout}\nSuggested attention dropout: {suggested_attention_dropout}")

        # Pack all suggestions into a dictionary
        hyperparams = {
            "learning_rate": suggested_learning_rate,
            "weight_decay": suggested_weight_decay,
            "hidden_dropout": suggested_hidden_dropout,
            "attention_dropout": suggested_attention_dropout,
            "decay_ratio": suggested_decay_ratio
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
        "num_epochs": 100,
        "weight_decay": hyperparams["weight_decay"],
        "hidden_dropout": hyperparams["hidden_dropout"],
        "attention_dropout": hyperparams["attention_dropout"],
        "decay_ratio": hyperparams["decay_ratio"],
        "target_epochs":100,
    })

    results, difference = crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode, path=path, compute_weights=compute_weights)
    multipl = (2 ** (results["MeanAbsoluteError"][0] - 0.5)) / 2 + 0.2 * (results["MeanAbsoluteError"][0] - 0.5) ** 2 + 0.5
    difference = multipl * difference
    plt.close('all')  # Close all open figures
    gc.collect()  # Force garbage collection
    return np.mean(results["R2Score"][:2]), np.mean([results["MeanAbsoluteError"][0], results["MeanAbsoluteError"][2]]), difference


def finetune_optimize(X, y, n_split, scaler, classes, short_mode, path=None, compute_weights=False, optimize_path=None):
    """
    Uses Optuna's TPESampler to optimize finetuning hyperparameters through cross-validation.

    Runs either a fresh or resumed Optuna study depending on the presence of ``optimize_path``.
    In distributed settings, it synchronizes the study creation and optimization using ``torch.distributed``.

    Parameters:
        X (pandas.DataFrame): Dataframe containing input features for prediction.
        y (List): List of labels (e.g., dissociation constants) for each interaction.
        n_split (int): Number of cross-validation folds.
        scaler: A scaler object used to normalize or transform the target labels.
        classes (List): Class labels used to stratify the cross-validation splits.
        short_mode (bool): If ``True``, only half of the cross-validation rounds are run (for speed).
            Default: ``False``
        path (str): Path to the pretrained model to be fine-tuned.
            Default: ``None``
        compute_weights (bool): Whether to compute sample weights during training.
            Default: ``False``
        optimize_path (str): Path to a previously saved Optuna study to resume optimization.
            Default: ``None``

    Returns:
        dict: The best set of hyperparameters found during the optimization process.
    """
    n_trials = 25
    has_local = "LOCAL_RANK" in os.environ 
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))  # Assuming world size defaults to 1
    if has_local:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    print_if_0_rank("WORLD SIZE:", world_size)

    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    result_path = f"study_results/{now}"
    
    storage = f"sqlite:///finetuning_{now}.db"

    if rank == 0:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if optimize_path is not None:
            study = optuna.load_study(
                study_name="finetuning_parameter_selection",
                storage=f"sqlite:///{optimize_path}"
            )
        else:
            study = optuna.create_study(
                directions=["maximize", "minimize", "minimize"],
                sampler=optuna.samplers.TPESampler(seed=(world_size+1)*14),
                study_name="finetuning_parameter_selection",
                load_if_exists=True,
                storage=storage)
        print("Starting optimization...")

    if has_local:
        torch.distributed.barrier()  # Ensure rank 0 finishes creating the study


    if has_local:
        torch.distributed.barrier()  # Optional: synchronize after loading the study

    print("Starting optimization...")
    if rank == 0:
        study.optimize(lambda trial: finetune_objective(trial, X, y, n_split, scaler, classes, short_mode, path, compute_weights=compute_weights), n_trials=n_trials, n_jobs=1)
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

    now = datetime.now().strftime('%Y%m%d_%H%M%S%f')
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
    best_run = trainer.hyperparameter_search(hp_space=optuna_hp_space, direction="minimize", backend="optuna", compute_objective=compute_objective, n_trials=32, study_name="pretraining_parameter_selection", storage=storage)
    
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
    the second metric is "MAE" and the third the MAE difference between validation and training runs.
    """
    if is_multiobjective:
        # Define metric names based on the order of objectives.
        metric_names = ["R2score", "MAE", "MAEDifference"]

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
