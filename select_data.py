import numpy as np

all_proteins = dict()
all_interactions = dict()
thresh = 900
with open("./data/9606.protein.links.v12.0.txt", "r") as f:
    with open(f"./data/9606.links.{thresh/1000}.txt", "w") as wr:
        f.readline()
        for line in f:
            
            arr = line.strip().split(" ")
            all_interactions[(arr[0],arr[1])] = 1
            if float(arr[2])>=thresh:
                wr.write(line)
                if not arr[0] in all_proteins:
                    all_proteins[arr[0]] = 0
                all_proteins[arr[0]] += 1

                if not arr[1] in all_proteins:
                    all_proteins[arr[1]] = 0
                all_proteins[arr[1]] += 1

with open(f"./data/9606.proteins.{thresh/1000}.txt", "w") as wr:        
    for protein in all_proteins.keys():
        wr.write(protein+"\n")

negative_samples = set()
all_proteins_keys = list(all_proteins.keys())

done = 0
for protein in all_proteins.keys():
    i=0
    number = all_proteins[protein]
    els = np.random.choice(all_proteins_keys, replace=False, size=len(all_proteins_keys))
    for el in els:
        if not ((protein, el) in all_interactions) and not ((el, protein) in all_interactions):
            if el == protein:
                continue
            if all_proteins[el] <= 0:
                continue
            if (protein, el) in negative_samples or (el, protein) in negative_samples:
                continue

            negative_samples.add((protein, el))
            i+=1

            all_proteins[el] -= 1

        if i >= number:
            break

    if i < number:
        els = np.random.choice(all_proteins_keys, replace=False, size=len(all_proteins_keys))
        for el in els:
            if not ((protein, el) in all_interactions) and not ((el, protein) in all_interactions):
                if el == protein:
                    continue
                if all_proteins[el] < 0:
                    continue
                if (protein, el) in negative_samples or (el, protein) in negative_samples:
                    continue

                negative_samples.add((protein, el))
                i+=1

                all_proteins[el] -= 1

            if i >= number:
                break
            
    all_proteins[protein] = 0

    if done % 100 == 0:
        print(f"Done {done} iterations.")
    done+=1

with open(f"./data/9606.negatives.{thresh/1000}.txt", "w") as wr:
    for prot1, prot2 in negative_samples:
        wr.write(str(prot1) + " " + str(prot2) +  "\n")
