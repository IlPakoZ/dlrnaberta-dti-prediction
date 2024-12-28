from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing
from torchmetrics.regression import R2Score, MeanAbsoluteError
import matplotlib.pyplot as plt

import torch
import threading
import copy
import model as md
import torch.nn as nn
from transformers import AutoModelForMaskedLM, RobertaForMaskedLM, AutoTokenizer, RobertaTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig
import optuna
import os
from datetime import datetime


PLOT_DIR = "plots"

def train_tokenizer(train_files):
    """Trains a RoBERTa tokenizer from the files ``train_files``.
    Uses a character tokenizer for interpretability of tokens.
    
    Parameters:
        train_files (list str): a list of paths from which to train the tokenizer.
    """

    roberta_base_tokenizer = CharBPETokenizer()

    # Customize training (change vocab size)
    roberta_base_tokenizer.train(train_files, vocab_size=1000, min_frequency=2, special_tokens=[
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
    
    eval_n = 0

    with torch.no_grad():
        for source, targets in data_loader:
            if eval_n > (train_parameters["gradient_accumulation_steps"]*train_parameters["log_performance_every"])//3:
                break
            targets = targets.reshape(-1, 1)
            output = cpu_model(source[0], source[1])
            
            targets = __unscale__(targets, scaler)
            output = __unscale__(output, scaler)

            r2_metric.update(output, targets)
            mae_metric.update(output, targets)

            eval_n += 1


    state["last_valid_mae_score"] = state["valid_mae_score"]
    state["valid_r2_score"] = r2_metric.compute()
    state["valid_mae_score"] = mae_metric.compute()
    state["val_maes"].append(state["valid_mae_score"])
    state["val_r2s"].append(state["valid_r2_score"])
    print(f"Step {thread_id} - R2 val: {state['valid_r2_score']:.4f}, MAE val: {state['valid_mae_score']:.4f}")
      

def __check_mae_score_exit__(train_parameters, counter, train=True):
    """
        Checks whether the mae score has converged. If the  ``counter`` is greater then 5, then the loss
        is stable and training is considered complete.

        Parameters:
            train_parameters (dict str -> obj): training parameters
            counter (int): number of consecutive logging steps in which loss hasn't decreased
            train (bool): whether we are checking the train mae. If false, validation mae is checked instead.
                Default: True
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
        Checks the additional exit conditions. Returns True if the training process is finished.
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

    state = {"valid_mae_score":0, "train_mae_score":0, "train_r2_score":0, "valid_r2_score":0, "counter_train":0, "counter_valid":0, "last_train_mae_score":0, "last_valid_mae_score":0, "val_maes":[], "val_r2s":[]}

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

                    optimizer.step()

                    # To do every logging step.
                    # Note that the ``step`` variable indicates each iteration in the training loop, while for logging steps
                    # only steps which update the gradients are considered
                    if (step + 1) % (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"]) == 0:

                        # Compute metrics for validation set if "validate_while_training" setting is enabled
                        if "validate_while_training" in train_parameters and train_parameters["validate_while_training"]:
                            
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
                        mae_train.reset()
                        r2_train.reset()
                        
                        # check whether division by gradient_steps is necessary or done automatically
                        mean_loss /= (train_parameters["log_performance_every"]*train_parameters["gradient_accumulation_steps"])
                        print(f"Step {step + 1} - Loss: {mean_loss:.4f}, R2: {state['train_r2_score']:.4f}, MAE: {state['train_mae_score']:.4f}")
                        
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
    return evaluate(model, val_dataset, scaler)

def __plot_metrics__(state):
    """
    Plots the validation metrics and saves them into a folder "plots" in current directory.

    Parameters:
        state (dict str -> float): state of the training process, used to memorize metrics and other important state variables
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    # Adjust layout
    plt.tight_layout()

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{PLOT_DIR}/metricsplots_{current_time}.png")
def crossvalidate_finetuning(model, dataset):
    raise NotImplementedError
    
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
    
def evaluate(model, val_dataset, scaler, metrics = [R2Score(), MeanAbsoluteError()], device = "cuda"):
    """
        Evaluates the model against a list of metric ``metrics``.

        Parameters:
            model: the model to evaluate
            val_dataset (torch Dataset): Dataset containing validation examples
            scaler: Scaler used to scale targets, will be used to scale them back to normal during metric calculation

            metrics: a list of metrics the model will be evaluated against
                Default: [R2Score(), MeanAbsoluteError()]
            device: device where the operations will be computed on
                Default: "cuda"
    """

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

    scores = []
    for metric in metrics:
        metric_score = metric.compute()
        ig_, ax_ = metric.plot()
        metric_name = str(metric)[:-2]
        scores.append((metric_name, metric_score))

        print("####### EVALUATION METRICS ########")
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

    #plt.grid()
    
    return scores


def create_model(train_parameters, layers_to_remove=0):
    """
        Load the pretrained models and prepare them for finetuning.
        LoRA is prepared for finetuning and LoftQ is applied for the target encoder.

        Parameters:
            train_parameters (dict str -> obj): training parameters
            layers_to_remove (int): number of layers to remove from the target encoder model.

    """
    torch.cuda.empty_cache()

    drug_encoder = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", output_hidden_states=True)
    drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    target_encoder = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
    target_tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer')
    target_encoder.resize_token_embeddings(len(target_tokenizer))
    target_encoder = deleteEncodingLayers(target_encoder, layers_to_remove)

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
    model = md.InteractionModelATTN(target_encoder, drug_encoder, train_parameters["model_dropout"])

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="ds_config.json")
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=train_parameters["gradient_accumulation_steps"], mixed_precision="bf16")

    return accelerator, model, target_tokenizer, drug_tokenizer

def optimize():
    """
    Not implemented
    """

    raise NotImplementedError
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=14),
        study_name="finetuning_parameter_selection",
        load_if_exists=True
    )

    
    #study.optimize(objective, n_trials=10)

