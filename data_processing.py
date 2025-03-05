import os
import gzip
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

def __preprocess_fasta__(read):
    """
        Generate a mapping ID -> sequence from a FASTA file.

        Parameters:
            read (str): path 

        Returns:
            mapping (dict: str -> str): mapping from ID to its correspondent sequence.
    """
    mapping = dict()
    with gzip.open(read,'rt') as f:
        fasta_sequence = ""
        just_started = True
        while(True):
            line1 = f.readline()
            if not line1:
                break
            if line1[0] == ">":
                # If it's the first sequence of the file don't do anything other than setting the unid
                if not just_started:
                    mapping[unid] = fasta_sequence
                else:
                    just_started = False

                unid = line1.split(" ")[0][1:]
                fasta_sequence = ""
                continue
            fasta_sequence += line1.strip()
                
    mapping[unid] = fasta_sequence
   
    return mapping

def __save_to_csv__(sorted_mapping, write):
    """
    Saves a list of pairs (id, sequence) to a CSV file.

        Parameters:
            sorted_mapping (list of tuple): A list of tuples where each tuple contains an id and its corresponding sequence.
            write (str): CSV file to which the list wil be saved.   
    """

    last = None
    with open(write, "w") as fw:
        fw.write(f"id,fasta\n")
        for id,seq in sorted_mapping:
            # Don't write duplicate sequences
            if seq == last:
                continue
            fw.write(f"{id},{seq}\n")
            last = seq

def __save_fasta_database__(sorted_mapping, write):
    """
    Saves a FASTA file containing sequences to be used for the creation of a BLAST database. You probably shouldn't call this method on its own.
    This method also deletes duplicate sequences. Called internally by ``prepare_blast_database()``.

        Parameters:
            sorted_mapping (list of tuple): A list of tuples where each tuple contains an id and its corresponding sequence.
            write (str): FASTA file to which the list wil be saved.

    """
    last = None
    with open(write, "a") as fw:
        for id,seq in sorted_mapping:
            # Don't write duplicate sequences
            if seq == last:
                continue
            
            # Split sequence in chunks of 60 characters to resemble a FASTA file
            string_val = '\n'.join(seq[i:i+60] for i in range(0, len(seq), 60))+"\n"
            
            if string_val.strip():
                fw.write(f">{id}\n")
                fw.write(string_val)


def prepare_blast_database(directory, database_filename="./processed/database.fasta"):    
    """
    Merges all FASTA files in a directory into a single FASTA file that can be used for the creation of a BLAST database.
    This method also deletes duplicate sequences.

        Parameters:
            directory (str): the directory from which the raw FASTA sequences are read.
            database_filename (str): name of the output file.
                    Default: ``./processed/database.fasta``


    """

    with open(database_filename, "w"):
        pass    
        
    for root, _, files in os.walk(directory):
        for file in files:
            fastas = __preprocess_fasta__(f"{root}/{file}")
            sortv = sorted(fastas.items(), key=lambda i: i[1])
    
            __save_fasta_database__(sortv, database_filename)


def extract_rna_seq(interaction_file, rna_seq_file, target_column="Target_RNA_sequence"):
    """
    Extracts all interactions from an Excel format interaction file, and writes the unique sequences in a file.
    The RNA sequences will be extracted from the column ``target_column``.
    Those sequences can then be compared to the sequences in the BLAST database to remove sequences that appear too similar.
    This is to avoid the bias of pretraining on the sequences the model will be finetuned on.
    
        Parameters:
            interaction_file (str): the path to the Excel file containing the interactions.
            rna_seq_file (str): name of the output file, containing all the unique sequences.
            target_column (str): name  of the column in the interaction file that contains the RNA sequences.
                    Default: ``Target_RNA_sequence``
 
    """
    sequences = pd.read_excel(interaction_file)[target_column]
    with open(rna_seq_file, "w") as fw:
        for seq in np.unique(sequences.values):
            fw.write(f"{seq}\n")

def create_pretraining_files(directory, blast_results_file="./processed/blast_results.out", proportion_train=.9):
    """
        Create the files needed for the pretraining. It creates a train and test split, stratified by file.
        For example, if there are 5 different fasta files in ``directory``, ``proportion_train*100%`` of the sequences of each
        file are used to make a training set, and the remaining percent for a test set. 
        Here, a BLAST result file is read from the path in ``blast_result_file``. The file should be in format 6. 
        The sequences matched by the query are removed from the raw FASTA sequences before creating the train and test sets.
        The sequences are shuffled before splitting. 

        Parameters:
            directory (str): the directory containing the sequences to use to create the train and test sets.
            blast_result_file (str): the name of the file containing the blastn query results.
                    Default: ``./processed/blast_results.out``
            proportion_train (float): proportion of the files to be used to make the train set.
                    Default: ``0.9``

 
    """
    res = pd.read_csv(blast_results_file, 
                    names=["queryid", "seqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"], 
                    header=None,
                    sep="\t")
    
    processed_fasta_dir = "./processed/fastas/"

    if not os.path.exists(processed_fasta_dir):
        os.makedirs(processed_fasta_dir)

    for root, _, files in os.walk(directory):
        for file in files:
            parent_dir = os.path.basename(root)
            fastas = __preprocess_fasta__(f"{root}/{file}")
            
            # Deletes all sequences that have been found by the blastn query
            for el in res["seqid"]:
                if el in fastas:
                    del fastas[el]
            
            sortv = sorted(fastas.items(), key=lambda i: i[1])
            save_to = f"./processed/fastas/{parent_dir}.csv"
            __save_to_csv__(sortv, save_to)    
            subprocess.call(f"tar -cvf - {save_to} | gzip -9 > {parent_dir}.tar.gz", shell=True)

            # Removes uncompressed file
            Path.unlink(save_to)

    #__split_train_test__("./processed/fastas/", proportion_train)

def encode_interaction_inputs(path):
    inters = pd.read_csv(f"{path}/processed/interactions/all.csv")
    pretrained_model_path = f"{path}/pretrained"
    X_smiles = inters["SMILES"]
    X_targets = inters["Target_RNA_sequence"]

    from transformers import RobertaForMaskedLM, AutoModel
    import train
    import torch
    from tqdm import tqdm
    import h5py 
       
    target_encoder = RobertaForMaskedLM.from_pretrained(pretrained_model_path, output_hidden_states=True).to("cuda")
    drug_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", output_hidden_states=True).to("cuda")

    target_tokenizer, drug_tokenizer = train.__get_tokenizers__()
    with torch.no_grad():
        encoded_targets = []
        encoded_drugs = []
        masks = []

        for x1, x2 in tqdm(zip(X_targets, X_smiles)):
            smiles = drug_tokenizer(x1,
                                    padding="max_length", 
                                    truncation=True, 
                                    max_length=512,
                                    return_tensors="pt").to("cuda")
            targets = target_tokenizer(x2,
                                    padding="max_length", 
                                    truncation=True, 
                                    max_length=512,
                                    return_tensors="pt").to("cuda")
            
            encoded_target = target_encoder(**targets).hidden_states[-1]
            encoded_drug = drug_encoder(**smiles).hidden_states[-1]

            encoded_targets.append(encoded_target.cpu().numpy())
            encoded_drugs.append(encoded_drug.cpu().numpy())
            masks.append(smiles["attention_mask"].cpu().numpy())

    with h5py.File(f"{path}/processed/interactions/all_encoded.h5", 'w') as f:
        f.create_dataset(f'Target_RNA_sequence', data=encoded_targets)
        f.create_dataset(f'SMILES', data=encoded_drugs)
        f.create_dataset(f'attention_mask', data=masks)

def __split_train_test__(processed_fastas_directory, proportion_train):
    """
    Creates a train and test split for the pretraining task. This function is called internally by "create_pretraining_files()".
    The input files are the compressed CSV files created by the "create_pretraining_files()" function. 
    The outputs are uncompressed text files containing only sequences. Currently the output paths are hard-coded.
    Outputs are saved in the directory "./processed/dataset".
    If they don't exist, the directories will be created automatically.
    The sequences are shuffled before splitting. 

    Parameters:
        processed_fastas_directory (str): the path to the directory containing the processed FASTA sequences.
        proportion_train (float): the proportion of data to use in the training set. The rest will go to the test set.
 
    """          
    dataset_dir = "./processed/dataset/"
    train_dir = "./processed/dataset/train"
    test_dir = "./processed/dataset/test"
    descript_path = "./processed/dataset/descript.csv"
    train_path = "./processed/dataset/train/train.txt"
    test_path = "./processed/dataset/test/test.txt"

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    with open(descript_path, "w") as fd:
        fd.write("id,type\n")
        
    with open(train_path, "w"):
        pass

    with open(test_path, "w"):
        pass

    for root, _, files in os.walk(processed_fastas_directory):
        for file in files:
            with gzip.open(f"{root}/{file}", "rt") as fr:
                fr.readline()
                shuffled = fr.readlines()[:-1]
            
            # Shuffles the data before splitting it
            np.random.shuffle(shuffled)
            
            with open(descript_path, "a") as fwd:
                # Generates train set
                with open(train_path, "a") as fwt:
                    for i in range(int(proportion_train*len(shuffled))):
                        splt = shuffled[i].split(",")
                        fwt.write(splt[1])
                        fwd.write(f"{splt[0]},{file}\n")
                # Generates test set
                with open(test_path, "a") as fwt:
                    for i in range(int(proportion_train*len(shuffled)), len(shuffled)):
                        splt = shuffled[i].split(",")
                        fwt.write(splt[1])
                        fwd.write(f"{splt[0]},{file}\n")

def create_finetuning_files(interactions_path):
    """
        Processes the raw interactions. It saves a new file for each target category and a file containing each category
        at once. Each row will be the target's sequence, the SMILES of the drug and the pKd.
        Files are saved as CSVs.
        The ``Category`` column is added to the file containing all categories to allow for stratification during cross-validation.

        Parameters:
            interactions_path (str): path of the RAW interactions, saved in excel format.
    """
    excel = pd.read_excel(interactions_path)    
    
    if not os.path.exists("./processed/interactions"):
        os.makedirs("./processed/interactions")

    col_names = ["Target_RNA_sequence", "SMILES", "pKd"]
    for cat in excel.groupby("Category"):
        interactions = cat[1][col_names]
        interactions.to_csv(f"./processed/interactions/{cat[0]}", index = False)

    excel[col_names+["Category"]].to_csv("./processed/interactions/all.csv", index = False)

