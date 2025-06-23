import train
from tqdm import tqdm
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_crossattention_weights(target_mask, drug_mask, target_tokenized, drug_tokenized, crossattention_weights, index, path="."):
    """
    Plots the cross-attention weights for a given drug-target pair, only considering unmasked tokens.
    
    Parameters:
        target_mask (np.ndarray): Boolean mask for target tokens.
        drug_mask (np.ndarray): Boolean mask for drug tokens.
        crossattention_weights (np.ndarray): The cross-attention weights.
    """
    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()

    tokens_input = target_tokenized["input_ids"][0][target_mask]

    target_token_str = target_tokenizer.convert_ids_to_tokens(tokens_input)

    tokens_input = drug_tokenized["input_ids"][0][drug_mask]
    drug_token_str = drug_tokenizer.convert_ids_to_tokens(tokens_input)

    subset = crossattention_weights[target_mask][:, drug_mask]
    height, width = subset.shape
    fig, ax = plt.subplots(
        figsize=(width * 0.2 + 2, height * 0.2 + 3),
        dpi=300
    )
    im = ax.imshow(subset, cmap='hot', interpolation='nearest')


    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, shrink=0.8)

    plt.title("Cross-Attention Weights")
    plt.xlabel("Drug Tokens")
    plt.ylabel("Target Tokens")
    vertical_labels = ['\n'.join(label) for label in drug_token_str]
    plt.xticks(ticks=np.arange(width), labels=vertical_labels)
    plt.yticks(ticks=np.arange(height), labels=target_token_str)
    max_val = subset.max()
    for i in range(height):
        for j in range(width):
            val = subset[i, j]
            if val > max_val / 2:
                # Extract just the digits after the decimal (no leading '0.')
                text = f"{val % 1:.2f}"[2:]
                plt.text(j, i, text,
                        ha='center', va='center',
                        color="black",
                        fontsize=6)
    plt.savefig(os.path.join(path, f"crossattention_weights_{index}.png"))
    

def plot_presum(tokenized_input, affinities, scaler, w, b, suffix, raw_affinities=False, path="."):
    """
    Generates and saves an annotated 1D heatmap of token-level contribution scores.

    Applies a linear transformation (w, b) to the raw affinities, rescales them back to original units,
    optionally enforces non-negativity, then arranges the per-token contributions into rows of up to 20 tokens.
    Each cell shows the token and its contribution; a colorbar indicates magnitude.

    Args:
        tokenized_input (dict): Output of a tokenizer with keys:
            - 'input_ids' (torch.Tensor): token ID sequences, shape (1, seq_len)
            - 'attention_mask' (torch.Tensor): mask indicating padding tokens
        affinities (torch.Tensor): Final layer summation affinity contributions from the model, shape (1, seq_len)
        scaler (object): Fitted scaler with `mean_` and `std_` attributes for inverse-transform.
        w (float): Weight applied to the summed affinities before bias.
        b (float): Bias added to the summed affinities.
        suffix (str): Filename suffix to append when saving the figure.
        raw_affinities (bool): If True, plot raw (signed) contributions on a blue–white–red scale.
            If False, enforce non-negative contributions and use a white–red scale.
            Default: False
        path (str): Directory in which to save the output PNG. Default: "."

    Raises:
        ValueError: If `sum(transformed_affinities) < 0` when `raw_affinities=False`.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (1.0, 0.95, 0.95),  
        (1.0, 0.5, 0.5),  
        (0.8, 0.0, 0.0)   
    ]

    custom_reds = LinearSegmentedColormap.from_list("CustomReds", colors)
    
    affinities = w*(affinities[0]) + b / len(affinities[0])
    affinities = (affinities*scaler.std_) + scaler.mean_/len(affinities)

    if sum(affinities) < 0 and not raw_affinities:
        raise ValueError("Cannot use non-raw affinities with negative binding affinity prediction")

    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()

    tokens_input = tokenized_input["input_ids"][0]
    token_str = target_tokenizer.convert_ids_to_tokens(tokens_input)


    pad_mask = tokenized_input["attention_mask"][0] == 0
    padding_affinities_sum = affinities[pad_mask].sum()
    non_padding_affinities = affinities[~pad_mask]
    processed_affinities = non_padding_affinities + padding_affinities_sum/len(non_padding_affinities)


    # Makes all affinities non-negative, while keeping the sum of the affinities the same
    # This is done by adding the sum of the negative affinities to the positive affinities
    # and dividing by the number of positive affinities

    # Those partial affinities should not be seen as affinities of each token, but rather their contribution
    # to the final affinity of the drug-target pair
    # Relative distances between positive affinities are kept the same
    if not raw_affinities:
        all_negative_non_paddings = processed_affinities[processed_affinities < 0]

        # Negative affinities are set to 0, and the sum of the negative affinities is split eavenly and added to the positive affinities
        while(len(all_negative_non_paddings) > 0):
            all_positive_non_paddings = processed_affinities[processed_affinities > 0]

            processed_affinities[processed_affinities < 0] = 0
            processed_affinities[processed_affinities > 0] = all_positive_non_paddings + all_negative_non_paddings.sum()/len(all_positive_non_paddings)
            all_negative_non_paddings = processed_affinities[processed_affinities < 0]

    # 1D Annotated heatmap plotting
    # Each row can handle a max of 20 tokens. Rows are not padded and are aligned to the left.
    # Each row is separated by some white space.
    # Red indicates strong contribution, while blue indicates low contribution in case of raw affinities;
    # otherwise, a white-red scale is used.
    # Contribution values are included inside each cell, and each cell is labeled with the token it corresponds
    # to (token_str[i]).

    max_per_row = 20

    n = len(processed_affinities)
    n_rows = int(np.ceil(n / max_per_row))
    grid = np.full((n_rows, max_per_row), np.nan)
    grid.flat[:n] = processed_affinities

    fig, ax = plt.subplots(
        figsize = (max_per_row * 1, n_rows * 1 + 2),
        dpi = 300
    )

    ax.set_xticks([])
    ax.set_yticks([])


    im = ax.imshow(
        grid,
        aspect='equal',
        cmap='bwr' if raw_affinities else custom_reds,
        vmin=np.nanmin(grid) if not raw_affinities else -max(abs(np.nanmin(grid)), abs(np.nanmax(grid))),
        vmax=np.nanmax(grid) if not raw_affinities else max(abs(np.nanmin(grid)), abs(np.nanmax(grid))),
    )

    def wrap_text(text, width=8):
        return '\n'.join(text[i:i+width] for i in range(0, len(text), width))

    for idx, val in enumerate(processed_affinities):
        r, c = divmod(idx, max_per_row)
        wrapped_token = wrap_text(token_str[idx], width=8)
        ax.text(c, r, f"{val:.2f}\n{wrapped_token}",
                ha='center', va='center', fontsize=8)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size=0.2, pad=0.3)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label("Contribution")
    print("Figure size (inches):", fig.get_size_inches())
    raw_suffix = "raw" if raw_affinities else ""
    plt.savefig(os.path.join(path, f"plotsum_pair_{raw_suffix}{suffix}.png"))

def check_overlap_training_test(training_file, test_file):
    """
    Computes and prints sequence-level and token-level overlap between training and test sets.

    Loads CSVs of training and test interactions, then:
      1. Measures percentage of identical RNA sequences and SMILES strings.
      2. Tokenizes all sequences/SMILES, computes unique token IDs, and reports
         what fraction of test tokens were seen during training.

    Args:
        training_file (str): Path to CSV file with columns 'Target_RNA_sequence' and 'SMILES'.
        test_file (str): Path to CSV file with same columns for the test set.
    """    
    # --- load data ---
    training_file = pd.read_csv(training_file)
    test_file     = pd.read_csv(test_file)

    target_training = training_file["Target_RNA_sequence"].tolist()
    target_test     = test_file["Target_RNA_sequence"].tolist()

    smiles_training = training_file["SMILES"].tolist()
    smiles_test     = test_file["SMILES"].tolist()

    # --- sequence‐level overlap (unchanged) ---
    target_training_set = set(target_training)
    smiles_training_set = set(smiles_training)

    matching_target = sum(seq in target_training_set for seq in target_test)
    matching_smiles = sum(seq in smiles_training_set for seq in smiles_test)

    target_match_percentage = matching_target / len(target_test) * 100
    smiles_match_percentage = matching_smiles / len(smiles_test) * 100

    # --- get your trained tokenizers ---
    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()

    # --- token‐level overlap using HF tokenizer APIs, batch mode ---

    # 1) batch‐encode all training & test sequences at once
    train_smiles_encodings = drug_tokenizer(
        smiles_training,
            truncation=False,
        padding=False,
    )["input_ids"]   # this is a List[List[int]]

    test_smiles_encodings = drug_tokenizer(
        smiles_test,
        truncation=False,
        padding=False,
    )["input_ids"]

    train_target_encodings = target_tokenizer(
        target_training,
        truncation=False,
        padding=False,
    )["input_ids"]

    test_target_encodings = target_tokenizer(
        target_test,
        truncation=False,
        padding=False,
    )["input_ids"]

    # 2) flatten each into a single stream of ints, then make sets
    train_smiles_token_ids = {tok for seq in train_smiles_encodings for tok in seq}
    test_smiles_token_ids  = {tok for seq in test_smiles_encodings  for tok in seq}

    train_target_token_ids = {tok for seq in train_target_encodings for tok in seq}
    test_target_token_ids  = {tok for seq in test_target_encodings  for tok in seq}

    # 3) compute intersections & percentages
    shared_smiles_tokens = test_smiles_token_ids & train_smiles_token_ids
    shared_target_tokens = test_target_token_ids & train_target_token_ids

    smiles_token_share = len(shared_smiles_tokens) / len(test_smiles_token_ids) * 100
    target_token_share = len(shared_target_tokens) / len(test_target_token_ids) * 100

    # --- report ---
    print(f"Sequence‐level overlap (100% the same):")
    print(f"  • Target RNA match percentage: {target_match_percentage:.2f}%")
    print(f"  •   SMILES match percentage: {smiles_match_percentage:.2f}%\n")

    print(f"Token‐level overlap (unique token IDs):")
    print(f"  • % of unique Target‐RNA tokens in test seen during training: {target_token_share:.2f}%")
    print(f"  •   % of unique SMILES tokens in test seen during training: {smiles_token_share:.2f}%")