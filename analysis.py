import train
from tqdm import tqdm
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_target_model_infos():
    """
        Prints the number of parameters and the embedding dimension of the target model.
    """
    target_encoder = train.load_RNABERTa(0)
    total_params = sum(p.numel() for p in target_encoder.parameters())
    print("Total params:", total_params)
    print("Embedding dimension:", target_encoder.config.hidden_size)

def count_training_tokens(input_file):
    """
        Counts the number of tokens in the training dataset.
        
        Parameters:
            input_file (str): Path to the training dataset text file.
        Returns:
            The number of tokens in the training dataset.
    """
    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()

    with open(input_file, "r") as f:
        lines = f.readlines()
        tokens = 0
        for line in tqdm(lines):
            tokens += len(target_tokenizer(line, max_length=512, truncation=True)["input_ids"])

    print("Total tokens:", tokens)
    return tokens

def count_training_lines(input_file):
    """
        Counts the number of lines in the training dataset.

        Parameters:
            input_file (str): Path to the training dataset text file.
        Returns:
            The number of lines in the training dataset.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    print("Total lines:", len(lines))
    return len(lines)

def plot_overlapping_distributions(folder_path, column="pKd", num_bins=30, alpha=0.5):
    """
    Reads all CSV files in the specified folder, extracts the data from the given column,
    and plots the distribution as overlapping histograms.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        column (str): The column name whose distribution is to be plotted.
        num_bins (int): Number of bins to use for the histogram. 
            Default: 30.
        alpha (float): Transparency level for each histogram.
            Default: 0.5.
    """
    # Find all CSV files in the folder
    csv_files = [file for file in glob.glob(os.path.join(folder_path, "*.csv")) if os.path.basename(file) != "all.csv"]
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # First pass: compute global min and max for the column across all files
    global_min = np.inf
    global_max = -np.inf
    data_dict = {}  # To store data for each file
    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            continue

        if column not in df.columns:
            print(f"Column '{column}' not found in {file}.")
            continue

        # Drop missing values
        data = df[column].dropna()
        if data.empty:
            print(f"No valid data in column '{column}' for file {file}.")
            continue

        global_min = min(global_min, data.min())
        global_max = max(global_max, data.max())
        data_dict[file] = data

    if global_min == np.inf or global_max == -np.inf:
        print(f"No valid data found for column '{column}'.")
        return

    # Define common bins for all histograms
    bins = np.linspace(global_min, global_max, num_bins + 1)

    plt.figure(figsize=(10, 6))
    for file, data in data_dict.items():
        # Plot histogram with density normalization so that the areas sum to 1
        plt.hist(data, bins=bins, alpha=alpha, label=os.path.splitext(os.path.basename(file))[0], density=True)

    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Distribution of '{column}' between different RNA types")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pKd distribution.png")
    plt.show()

def count_finetuning_tokens(input_file):
    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
    total_tokens = 0


    df = pd.read_csv(input_file)
        

    for _, row in df.iterrows():
        sequence = row["Target_RNA_sequence"]
        tokens = len(target_tokenizer(sequence)["input_ids"])
        if tokens > 510:
            print("Warning: Sequence length exceeds 510 tokens:", sequence)
        print("Sequence length:", tokens)
        total_tokens += tokens

    print("Total tokens:", total_tokens)
    return total_tokens
