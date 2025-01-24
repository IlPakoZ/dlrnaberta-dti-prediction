from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing
from torchmetrics.regression import R2Score, MeanAbsoluteError, PearsonCorrCoef
from datetime import datetime
from transformers import AutoModelForMaskedLM, RobertaForMaskedLM, AutoTokenizer, RobertaTokenizerFast, DataCollatorForLanguageModeling, AutoModel, RobertaConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torch.nn as nn
import model as md
import numpy as np
import pandas as pd
import torch
import threading
import copy
import optuna
import os
import pickle
import math

PLOT_DIR = "plots"

def train_tokenizer(train_files):
    """
        Trains a RoBERTa tokenizer from the files ``train_files``.
        Uses a character tokenizer for interpretability of tokens.
        
        Parameters:
            train_files (list str): a list of paths from which to train the tokenizer.
    """

    roberta_base_tokenizer = CharBPETokenizer()

    # Customize training (change vocab size)
    roberta_base_tokenizer.train(train_files, vocab_size=2048, min_frequency=2, special_tokens=[
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



def __in_train_evaluate_model__(cpu_model, data_loader, train_parameters, state, scaler, thread_id):
    """
        Evaluates the model on the validation data. 
        Evaluation is done on CPU, so that the training can keep going in the meanwhile.    

        Parameters:
            cpu_model: the model to be evaluated
            data_loader (torch DataLoader): data loader containing the validation data
            train_parameters (dict str -> obj): training parameters
            state (dict str -> float): state of the training process, used to memorize metrics and other important state variables
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation
            thread_id (str): name of the thread
    """
    cpu_model.eval() 

    r2_metric = R2Score()
    mae_metric = MeanAbsoluteError()
    pearson_r_metric = PearsonCorrCoef()
    eval_n = 0

    with torch.no_grad():
        for source, targets in data_loader:
            if not "val_all" in train_parameters:
                if eval_n > (train_parameters["gradient_accumulation_steps"]*train_parameters["log_performance_every"])//3:
                    break
            targets = targets.reshape(-1, 1)
            output = cpu_model(source[0], source[1])
            
            targets = __unscale__(targets, scaler)
            output = __unscale__(output, scaler)

            r2_metric.update(output, targets)
            mae_metric.update(output, targets)
            pearson_r_metric.update(output, targets)

            eval_n += 1


    state["last_valid_mae_score"] = state["valid_mae_score"]
    state["valid_r2_score"] = r2_metric.compute()
    state["valid_mae_score"] = mae_metric.compute()
    state["valid_pearson_r_score"] = pearson_r_metric.compute()
    if state["valid_r2_score"] > state["best_r2"]: 
        state["best_r2"] = state["valid_r2_score"]
        state["best_mae_score"] = state["valid_mae_score"]
        state["best_model"] = cpu_model

    state["val_maes"].append(state["valid_mae_score"])
    state["val_r2s"].append(state["valid_r2_score"])
    state["val_pears"].append(state["valid_pearson_r_score"])

    print(f"Step {thread_id} - R2 val: {state['valid_r2_score']:.4f}, MAE val: {state['valid_mae_score']:.4f}, Pearson R val: {state['valid_pearson_r_score']:.4f}")
      

def __check_mae_score_exit__(train_parameters, counter, train=True):
    """
        Checks whether the mae score has converged. If the  ``counter`` is greater then 5, then the loss
        is stable and training is considered complete.

        Parameters:
            train_parameters (dict str -> obj): training parameters
            counter (int): number of consecutive logging steps in which loss hasn't decreased
            train (bool): whether we are checking the train mae. If ``False``, validation mae is checked instead.
                Default: ``True``
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
    """
    if scaler is None:
        return target_value
    return scaler.inverse_transform(target_value)

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

def pretrain_and_evaluate(args, model, tokenizer, eval_only, checkpoint_path):
    """
        Pretrains and evaluates the RNA model. Pretraining consists of a Masked Language Modeling task.

        Parameters:
            args (DataClass): contains training parameters for the huggingface Trainer.
            model: model to be pretrained
            tokenizer: tokenizer of the model
            eval_only: whether to only evaluate the performance of the model and not train it.
            checkpoint_path: resume pretraining, starting from checkpoint in checkpoint_path.
    """
    data_files = {"val": args.test_datapath}
    
    if eval_only:
        data_files["train"] = data_files["val"]
    else:
        print(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        data_files["train"] = args.train_datapath

    datasets = load_dataset("text", data_files=data_files)
    datasets = datasets.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=datasets["train"], eval_dataset=datasets["val"],)
        
    print("I'm evaluating...")
    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    print(f"Initial eval bpc: {eval_loss/math.log(2)}")

    output_dir = "./pretrained"
    if not eval_only:
        print("I'm training...")
        trainer.train(resume_from_checkpoint=checkpoint_path)
        trainer.save_model(output_dir=output_dir)

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        print(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")
    
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
    tuple: A tuple containing the training and validation feature matrices (train_X, val_X) 
           and the training and validation target vectors (train_y, val_y).
    """
    classes = list(zip(inters["Category"].values, (inters["pKd"]+0.5).astype(int)))    
    
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
    device = accelerator.device
    
    r2_train = R2Score().to(device)
    mae_train = MeanAbsoluteError().to(device)
    pearson_r_train = PearsonCorrCoef().to(device)
    
    state = {"valid_mae_score":0, "train_mae_score":0, "train_r2_score":0, "train_pearson_r_score":0, "valid_r2_score":0, "valid_pearson_r_score": 0, "counter_train":0, "counter_valid":0, "last_train_mae_score":0, "last_valid_mae_score":0, "best_r2":-100, "val_maes":[], "val_r2s":[], "val_pears":[]}

    optimizer = AdamW(model.parameters(), lr=train_parameters["learning_rate"], weight_decay=train_parameters["weight_decay"])
    train_loader = DataLoader(train_dataset, batch_size=train_parameters["train_batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    with torch.no_grad():
        for source, targets in val_loader:
            source1, targets1 = source, targets
            break

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    print(f"ALLOCATED MEMORY ON CUDA: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    print(accelerator.state)
    if accelerator.state.deepspeed_plugin is not None:
        print("DeepSpeed Plugin is enabled and working.")
    else:
        print("DeepSpeed Plugin is not enabled.")
    model.train()

    step = 0  # Initialize the step variable
    epoch = 1
    if "num_training_steps" in train_parameters:
        max_steps = train_parameters["num_training_steps"]
    else:
        max_steps = train_parameters["num_epochs"]*len(train_loader)

    print(f"Training for {max_steps} steps")
    criterion = torch.nn.MSELoss()
    mean_loss = 0
    to_exit = False
    grad_mean = []

    while step < max_steps and not to_exit:
            print(f"EPOCH {epoch}:")
            done = False
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

                    optimizer.step()

                    # To do every logging step.
                    # Note that the ``step`` variable indicates each iteration in the training loop, while for logging steps
                    # only steps which update the gradients are considered
                    if (step + 1) % (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"]) == 0:
                        # Compute metrics for validation set if "validate_while_training" setting is enabled
                        if ("validate_while_training" in train_parameters and train_parameters["validate_while_training"] and not "val_all" in train_parameters) or ("val_all" in train_parameters and not done):
                            done = True
                            cpu_model = copy.deepcopy(model).cpu()
                            output = cpu_model(source1[0], source1[1])
                            print(output.detach(), targets1.detach())
                            eval_thread = threading.Thread(
                                target=__in_train_evaluate_model__, 
                                args=(cpu_model, val_loader, train_parameters, state, scaler, step+1)
                            )
                            eval_thread.start()
                        
                        state["last_train_mae_score"] = state["train_mae_score"]
                        state["train_r2_score"] = r2_train.compute()
                        state["train_mae_score"] = mae_train.compute()
                        state["train_pearson_r_score"] = pearson_r_train.compute()
                        mae_train.reset()
                        r2_train.reset()
                        pearson_r_train.reset()

                        # check whether division by gradient_steps is necessary or done automatically
                        mean_loss /= (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"])
                        print(f"Step {step + 1} - Loss: {mean_loss:.4f}, R2: {state['train_r2_score']:.4f}, MAE: {state['train_mae_score']:.4f}, Pearson R: {state['train_pearson_r_score']:.4f}")
                        
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

    __plot_metrics__(state)
    scores = evaluate(model, val_dataset, scaler)
    model = model.cpu()
    #model, optimizer = accelerator.free_memory(model, optimizer)
    
    # Evaluate best model
    print("Best R^2 recoded:", state["best_r2"])
    best_scores = evaluate(state["best_model"], val_dataset, scaler)
    save_finetuned_model(state["best_model"], train_parameters, train_dataset, val_dataset, scaler, suffix="_best")
    

    del train_dataset
    del val_dataset
    torch.cuda.empty_cache()

    return scores, model

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
    plt.savefig(f"{PLOT_DIR}/metricsplots_{current_time}.png")

def save_finetuned_model(finetune_model, train_parameters, train_dataset, val_dataset, scaler, suffix=""):
    """
        Saves the finetuned model, datasets and scaler to drive so that they can be easily loaded in any time.

        Parameters:
            finetune_model: huggingface finetuned model
            train_parameters (dict str -> obj): training parameters
            train_dataset (md.InterDataset): the train dataset used for finetuning.
            val_dataset (md.InterDataset): the validation dataset used for finetuning.
            scaler: scaler used for finetuning.
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


def crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode=False):
    """
        Executes finetuning with crossvalidation.

        Parameters:
            X (pandas.DataFrame): dataframe containing data used for prediction
            y (List): list containing labels to predict for each interaction (dissociation constant)
            n_split (int): number of cross validation folds to split the data into
            train_parameters (dict str -> obj): training parameters
            scaler: scaler to apply on labels
            classes (List): classes by which to stratify the folds
            short_mode (bool): if True, crossvalidation is not complete, meaning that only three rounds of crossvalidation are applied.
                Default: False
        Returns:
            The mean, the minimum and maximum validation scores of the runs, for each metric.
    """
    all_scores = dict()
    i = 0
    for train_dataset, val_dataset in get_crossvalidate_datasets(X, y, n_split, scaler, classes):
        accelerator, finetune_model = create_finetune_model(train_parameters)
        scores, finetune_model = finetune_and_evaluate(finetune_model, accelerator, train_parameters, train_dataset, val_dataset, scaler)
        del accelerator
        del finetune_model
        for score in scores:
            if not score[0] in all_scores:
                all_scores[score[0]] = []

            all_scores[score[0]].append(score[1])

        i+=1
        if short_mode:
            if i >= 3:
                break

    result = dict()
    for (metric, scores) in all_scores.items():
        result[metric] = ((sum(scores)/len(scores)), min(scores), max(scores))

    return result
    
    

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
        print(f"Fold {i+1}:")
        train_X = X.iloc[train_index]
        val_X = X.iloc[val_index]
        train_y = y[train_index]
        val_y = y[val_index]


        train_dataset, val_dataset = __prepare_train_val_datasets__(drug_tokenizer, target_tokenizer, train_X, val_X, train_y, val_y, scaler, plot)

        yield train_dataset, val_dataset

def deleteEncodingLayers(model, num_layers_to_remove):  # must pass in the full bert model
    """
    Removes layers from a RoBERTa model to get a smaller model.
    
    Parameters:
        model: a RoBERTa model.
        num_layers_to_remove (int): number of layers to remove.

    Returns:
        a copy of the original model with the number of layers specified removed.
    """
    oldModuleList = model.roberta.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, len(oldModuleList)-num_layers_to_remove):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.roberta.encoder.layer = newModuleList

    return copyOfModel
    
def evaluate(model, val_dataset, scaler, metrics = [R2Score(), MeanAbsoluteError(), PearsonCorrCoef()], device = "cuda"):
    """
        Evaluates the model against a list of metric ``metrics``.

        Parameters:
            model: the model to evaluate
            val_dataset (torch Dataset): Dataset containing validation examples
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation
            metrics: a list of metrics the model will be evaluated against
                Default: ``[R2Score(), MeanAbsoluteError(), PearsonCorrCoef()]``
            device: device where the operations will be computed on
                Default: "cuda"

            Returns:
                A list of pairs with metrics and their corresponding scores
    """

    # TODO: Device is not implemented
    model = model.cpu()
    model.eval()

    for id in range(len(metrics)):
        metrics[id] = metrics[id]

    val_loader = DataLoader(val_dataset, shuffle=True)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for source, targets in val_loader:
                        
            output = model(source[0], source[1])
            output = __unscale__(output, scaler)
            targets = __unscale__(targets, scaler)
            for metric in metrics:
                metric.update(output, targets.reshape(-1,1))

            all_outputs.extend(output.cpu().numpy().ravel())
            all_targets.extend(targets.cpu().numpy().ravel())
    #model = model.cpu()
    all_pairs = zip(all_targets, all_outputs)
    all_pairs = sorted(all_pairs, key=lambda x: x[0])
    sorted_targets, sorted_outputs = zip(*all_pairs)
    sorted_targets = list(sorted_targets)
    sorted_outputs = list(sorted_outputs)

    print("####### EVALUATION METRICS ########")

    scores = []
    for metric in metrics:
        metric_score = metric.compute()
        ig_, ax_ = metric.plot()
        metric_name = str(metric)[:-2]
        scores.append((metric_name, metric_score))

        print(f"{metric_name}: {metric_score}")
    
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

def load_RNABERTa(layers_to_remove):
    """
        Returns the target encoder with the specified number of encoder blocks removed.

        Returns:
            the target encoder.
    """
    target_tokenizer, _ = __get_tokenizers__()
    configuration = RobertaConfig(vocab_size=len(target_tokenizer),
                                hidden_size=384,
                                num_hidden_layers=12,  
                                num_attention_heads=12,  
                                intermediate_size=3072,
                                max_position_embeddings=514,
                                output_hidden_states=True)

    target_encoder = RobertaForMaskedLM(configuration)

    target_encoder = deleteEncodingLayers(target_encoder, layers_to_remove)
    return target_encoder    

def create_finetune_model(train_parameters, from_pretrained=None):
    """
        Load the pretrained models and prepare them for finetuning.
        LoRA is prepared for finetuning and LoftQ is applied for the target encoder.

        Parameters:
            train_parameters (dict str -> obj): training parameters
            from_pretrained (str): path of the pretrained model to load, if available. If not, create a new model. 
                Default: None
        Returns:
            a transformers accelerator and the model to be trained.

    """
    torch.cuda.empty_cache()

    drug_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", output_hidden_states=True)
    target_encoder = load_RNABERTa(train_parameters["layers_to_remove"])

    loftq_config = LoftQConfig(loftq_bits=8)           
    lora_config_target = LoraConfig(r=train_parameters["lora_r"],
                                lora_alpha=train_parameters["lora_alpha"], 
                                use_rslora=True, 
                                bias="none",
                                target_modules=["query", "key", "value", "dense"],
                                init_lora_weights="loftq", 
                                loftq_config=loftq_config,
                                task_type=TaskType.FEATURE_EXTRACTION,
                                lora_dropout=train_parameters["lora_dropout"])
    
    # Could add quantization for this too
    lora_config_drug = LoraConfig(r=train_parameters["lora_r"],
                                lora_alpha=train_parameters["lora_alpha"], 
                                use_rslora=True, 
                                bias="none",
                                target_modules=["query", "key", "value", "dense"],
                                task_type=TaskType.FEATURE_EXTRACTION,
                                lora_dropout=train_parameters["lora_dropout"])
        
    target_encoder = get_peft_model(target_encoder, lora_config_target)
    drug_encoder = get_peft_model(drug_encoder, lora_config_drug)

    torch.cuda.empty_cache()

    print_trainable_parameters("Target encoder:", target_encoder)
    print_trainable_parameters("Drug encoder:", drug_encoder)

    if not from_pretrained:
        config = md.InteractionModelATTNConfig(train_parameters["model_dropout"])
        model = md.InteractionModelATTNForRegression(config, target_encoder, drug_encoder)
    else:
        model = md.InteractionModelATTNForRegression.from_pretrained(from_pretrained, target_encoder, drug_encoder)

    deepspeed_plugin = None
    # This makes everything crash during cross_validation for some reason
    # To reproduce, move deepspeed_plugin + accelerator lines before model loading
    # Maybe conflict between LoftQ quantization and deepspeed moving everything to CPU
    #deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="ds_config.json")
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=train_parameters["gradient_accumulation_steps"], mixed_precision="bf16")

    return accelerator, model

def objective(trial, X, y, n_split, scaler, classes, short_mode):

    train_parameters = {
        "train_batch_size": 4,
        "device": "cuda",
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16])}

    gas = train_parameters["gradient_accumulation_steps"]

    train_parameters.update({
                        "learning_rate": trial.suggest_float("learning_rate", 2e-6, 2e-5, log=True)*np.sqrt(gas*train_parameters["train_batch_size"]),
                        "adam_epsilon": 1e-6,
                        "num_epochs":20,
                        "log_performance_every":5,
                        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
                        "model_dropout": trial.suggest_float("model_dropout", 0.1, 0.5),
                        "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
                        "lora_alpha": trial.suggest_categorical("lora_alpha", [4, 8, 16, 32, 64]),
                        "lora_dropout":trial.suggest_float("lora_dropout", 0, 0.5),
                        "max_norm":1,
                        })

    results = crossvalidate(X, y, n_split, train_parameters, scaler, classes, short_mode)
    return np.mean(results["R2Score"][:2])

def optimize(X, y, n_split, scaler, classes, short_mode):
    """
        Use Optuna TPESampler to optimize finetuning parameters.
        
        Parameters:
            X (pandas.DataFrame): dataframe containing data used for prediction
            y (List): list containing labels to predict for each interaction (dissociation constant)
            n_split (int): number of cross validation folds to split the data into
            scaler: scaler to apply on labels
            classes (List): classes by which to stratify the folds.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=14),
        study_name="finetuning_parameter_selection",
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, X, y, n_split, scaler, classes, short_mode), n_trials=20)

    return study.best_params
