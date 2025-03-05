import train
from tqdm import tqdm

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