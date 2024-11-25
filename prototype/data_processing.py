import pandas as pd
import requests
import time
import urllib.request
import gzip
import os.path
import numpy as np
from sklearn.model_selection import train_test_split

def __preprocess_fasta__(filename):
    mapping = dict()
    with gzip.open(filename,'rt') as f:
        fasta_sequence = ""
        just_started = True
        while(True):
            line1 = f.readline()
            if not line1:
                break
            if line1[0] == ">":
                if not just_started:
                    mapping[unid] = fasta_sequence
                else:
                    just_started = False

                unid = line1.split("|")[1]
                fasta_sequence = ""
                continue
            fasta_sequence += line1.strip()
            
    mapping[unid] = fasta_sequence
    return mapping


def __map_ens_to_seq__(ens_to_seq_mappings, uni_to_ens_mappings, filename):
    mapping = __preprocess_fasta__(filename)
    for k,v in mapping.items():
        ens_to_seq_mappings[uni_to_ens_mappings[k]] = v 
    return ens_to_seq_mappings

def get_sequences(ensembl_ids, job_id = None, filename = "result.fasta.gz", extra = "extra.fasta.gz"):
    
    if not job_id:

        url = "https://rest.uniprot.org/idmapping/run"
        data = {
            "from": "STRING",
            "to": "UniProtKB",
            "format": "json",
            "ids": ",".join(ensembl_ids)
        }
        
        # Submit mapping request
        response = requests.post(url, data=data)
        response.raise_for_status()
        job_id = response.json()["jobId"]
        # Check job status
        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        while True:
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            status = status_response.json()
            if "jobStatus" in status:
                if status["jobStatus"] == "FINISHED":
                    break
            time.sleep(0.1)

    results_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
    results_response = requests.get(results_url)
    results_response.raise_for_status()
    response = results_response.json()["results"]
    ens_to_seq_mappings = dict()
    uni_to_ens_mappings = dict()
    for res in response:
        uni_to_ens_mappings[res["to"]] = res["from"] 

    for id in ensembl_ids:
        ens_to_seq_mappings[id] = None

    if not os.path.exists(filename):
        fasta_url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job_id}?compressed=true&format=fasta"
        urllib.request.urlretrieve(fasta_url, filename)

    if not os.path.exists(extra):
        fasta_url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28%28taxonomy_id%3A9606%29%29"
        urllib.request.urlretrieve(fasta_url, extra)

    sequences = __map_ens_to_seq__(ens_to_seq_mappings, uni_to_ens_mappings, filename)
    new_sequences = dict()

    # Remove all "None" entries
    for k,v in sequences.items():
        if v:
            new_sequences[k] = v
    
    main_sequences = set(new_sequences.values())

    # Use external data (extra sequences that are not included in the main interaction problem)
    # Useful for having extra data for pretraining
    extra_sequences = {k: v for k,v in __preprocess_fasta__(extra).items() if not k in main_sequences}

    return new_sequences, extra_sequences


#Lax approach to get negative interactions in similar numbers of positive interactions
def __get_negative_inters__(inters, count_inters_per_protein):
    negative_samples = []
    already_added = set()

    for index, inter in inters.iterrows():
        prt1 = inter["protein1"] 
        prt2 = inter["protein2"]
        if prt2 < prt1:
            prt2, prt1 = prt1, prt2
        # Doesn't matter if there is a bit more negative samples than they should be, prt2 can be repeated
        if not prt1 in count_inters_per_protein:    
            if prt1 in already_added:
                continue                
            if prt2 in already_added:
                continue
            negative_samples.append(((prt1, prt2), 0))
            already_added.add(prt1)
            already_added.add(prt2)
            
            
        elif count_inters_per_protein[prt1] > 0:
            count_inters_per_protein[prt1] -= 1
            if prt2 in count_inters_per_protein:
                count_inters_per_protein[prt2] -= 1

                negative_samples.append(((prt1, prt2), 0))
    
    return negative_samples

def __split_data__(all_inters, unique_prot, extra_prot, train_size):
    train_prot, test_prot = train_test_split(unique_prot, train_size=train_size, random_state=15)
    train_prot, val_prot = train_test_split(train_prot, train_size=train_size, random_state=15)
    
    train_extra, test_extra = train_test_split(extra_prot, train_size=train_size, random_state=15)
    train_extra, val_extra = train_test_split(train_extra, train_size=train_size, random_state=15)

    set_train = set(train_prot)
    set_val = set(val_prot)
    set_test = set(test_prot)

    train_inter = []
    val_inter = []
    test_inter = []

    undeciseveness = 0.8
    rng = np.random.default_rng(15)

    
    for inter in all_inters:
        if inter[0][0] in set_train and inter[0][1] in set_train:
            train_inter.append(inter)
        elif inter[0][0] in set_train or inter[0][1] in set_train:
            if rng.random(1)[0] < undeciseveness:
                train_inter.append(inter)
            else:
                val_inter.append(inter)
            
        elif inter[0][0] in set_val or inter[0][1] in set_val:
            val_inter.append(inter)
        elif inter[0][0] in set_test and inter[0][1] in set_test:
            test_inter.append(inter)
        else:
            val_inter.append(inter) 

    return (train_prot, train_inter, train_extra),(val_prot, val_inter, val_extra),(test_prot, test_inter, test_extra)

def __save_data__(train, val, test, sequences, extra_sequences,
                  prot_indexes_file, 
                  train_prot_file, 
                  val_prot_file, 
                  test_prot_file, 
                  train_inters_file, 
                  val_inters_file, 
                  test_inters_file,
                  tokenizer_train_file):

    with open(prot_indexes_file, "w") as f:
        f.write("STRING_id,set\n")
        for line in train[0]:
            f.write(f"{line},train\n")
        for line in val[0]:
            f.write(f"{line},val\n")
        for line in test[0]:
            f.write(f"{line},test\n")
    
    with open(tokenizer_train_file, "w") as f:
        for line in train[0]:
            f.write(f"{sequences[line]}\n")
        for line in train[2]:
            f.write(f"{extra_sequences[line]}\n")

    with open(train_prot_file, "w") as f:
        f.write("STRING_id,fasta\n")
        for line in train[0]:
            f.write(f"{line},{sequences[line]}\n")
        for line in train[2]:
            f.write(f"{line},{extra_sequences[line]}\n")

    with open(val_prot_file, "w") as f:
        f.write("STRING_id,fasta\n")
        for line in val[0]:
            f.write(f"{line},{sequences[line]}\n")
        for line in val[2]:
            f.write(f"{line},{extra_sequences[line]}\n")

    with open(test_prot_file, "w") as f:
        f.write("STRING_id,fasta\n")
        for line in test[0]:
            f.write(f"{line},{sequences[line]}\n")
        for line in train[2]:
            f.write(f"{line},{extra_sequences[line]}\n")

    with open(train_inters_file, "w") as f:
        f.write("fasta1,fasta2,labels\n")
        for line in train[1]:
            f.write(f"{sequences[line[0][0]]},{sequences[line[0][1]]},{line[1]}\n")

    with open(val_inters_file, "w") as f:
        f.write("fasta1,fasta2,labels\n")
        for line in val[1]:
            f.write(f"{sequences[line[0][0]]},{sequences[line[0][1]]},{line[1]}\n")

    with open(test_inters_file, "w") as f:
        f.write("fasta1,fasta2,labels\n")
        for line in test[1]:
            f.write(f"{sequences[line[0][0]]},{sequences[line[0][1]]},{line[1]}\n")
    


def preprocess_data(file, thr=900, job_id=None, train_size = 0.8, 
                    sequence_filename="data/result.fasta.gz", 
                    extra_file = "data/extra.fasta.gz",
                    prot_indexes_file = "dataset/prot_indexes.csv", 
                    train_prot_file = "dataset/train/train_proteins.csv", 
                    val_prot_file = "dataset/val/val_proteins.csv", 
                    test_prot_file = "dataset/test/test_proteins.csv",
                    train_inters_file = "dataset/train/train_inters.csv", 
                    val_inters_file = "dataset/val/val_inters.csv", 
                    test_inters_file = "dataset/test/test_inters.csv",
                    tokenizer_train_file = "dataset/train/tokenizer_train.txt"):
    
    inters = pd.read_csv(file, sep=" ")
    unique_prot = list(set(inters.loc[:,["protein1", "protein2"]].values.reshape(-1)))
    unique_prot.sort()
    sequences, extra_sequences = get_sequences(unique_prot, job_id, filename=sequence_filename, extra = extra_file)
    
    inters = inters[inters["protein1"].isin(sequences.keys())]
    inters = inters[inters["protein2"].isin(sequences.keys())]
    unique_prot = [el for el in unique_prot if el in sequences]
    extra_prot = [el for el in set(extra_sequences.keys())]
    extra_prot.sort()

    positive_inters = inters[inters.loc[:, "combined_score"] > thr]

    list_positive_inters = [((row["protein1"], row["protein2"]), 1) for index, row in positive_inters.iterrows()]
    all_posit_proteins = list(set(positive_inters.loc[:,["protein1", "protein2"]].values.reshape(-1)))
    
    count_inters_per_protein = dict()
    for prot in positive_inters.loc[:,["protein1", "protein2"]].values.reshape(-1):
        if not prot in count_inters_per_protein:
            count_inters_per_protein[prot]=0
        count_inters_per_protein[prot]+=1
    
    all_posit_proteins.sort()
    negative_inters = inters[inters.loc[:, "combined_score"] < 200]
    
    list_negative_inters = __get_negative_inters__(negative_inters, count_inters_per_protein) 
    
    list_all_inters = list_positive_inters + list_negative_inters
    train, val, test = __split_data__(list_all_inters, unique_prot, extra_prot, train_size)
    __save_data__(train, val, test, sequences, extra_sequences, prot_indexes_file, train_prot_file, 
                  val_prot_file, test_prot_file, train_inters_file, val_inters_file, test_inters_file, tokenizer_train_file)

inters = preprocess_data("./data/9606.protein.links.v12.0.txt", job_id="18f99dfc70e1fa7599e18bd26014de8bc8708b55")