import itertools
import RNA
import mordred.GeometricalIndex
import pandas as pd
import numpy as np

from rdkit import Chem
from collections import Counter
from mordred import Calculator, descriptors

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
def compute_psessc_v1(seqs, ids, names,
                      n_val=2, lambda_val=8, w_val=0.5):
    """
    Given lists of sequences, IDs and names, returns a DataFrame
    of the Pseudo‐structure‐status‐composition v1 features.
    """
    # 1. set up
    nfeatures = 10**n_val
    struct_status = ["A", "G", "C", "U",
                     "A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]
    free_energy = {"A": 0, "G": 0, "C": 0, "U": 0,
                   "A-U": -2, "U-A": -2,
                   "G-C": -3, "C-G": -3,
                   "G-U": -1, "U-G": -1}
    # all possible k‑tuple × status combos (e.g. "A,U", "G-C,G-U", …)
    feat_desc = [','.join(p) for p in itertools.product(struct_status, repeat=n_val)]
    
    # helper to compute the λ‐tier correlation
    def calc_corr(comp_seq, seq_len):
        corr = []
        for i in range(1, lambda_val+1):
            s = 0
            for j in range(seq_len - i):
                f1 = free_energy[comp_seq[j]]
                f2 = free_energy[comp_seq[j+i]]
                s += (f1 - f2)**2
            corr.append(s / (seq_len - i))
        return corr

    # build output rows
    rows = []
    for seq, sid, sname in zip(seqs, ids, names):
        # a) fold → dot‑bracket + MFE
        ss, mfe = RNA.fold(seq)
        # b) build comp_seq: for each position, either "." or the paired base
        comp_seq = []
        stack = []
        for i, ch in enumerate(ss):
            if ch == '(':
                stack.append(i)
                comp_seq.append('.')  # will fill on ) 
            elif ch == ')':
                j = stack.pop()
                comp_seq[j] = seq[i]
                comp_seq.append(seq[j])
            else:
                comp_seq.append('.')
        # c) build the “composition vector” of length = len(seq)-1
        comp2 = []
        for a,b in zip(seq, comp_seq):
            comp2.append(a if b=='.' else f"{a}-{b}")
        # d) normalize k‑tuple frequencies
        k_tuples = [','.join(pair) for pair in zip(comp2, comp2[1:])]
        freqs = Counter(k_tuples)
        L = len(k_tuples)
        comp_norm = [freqs.get(fd,0)/L for fd in feat_desc]
        # e) correlation vector
        corr = calc_corr(comp2, len(seq))
        # f) final feature vector of length nfeatures + λ
        denom = sum(comp_norm) + w_val * sum(corr)
        feat_vec = [
            round(cn/denom, 5) for cn in comp_norm
        ] + [
            round((w_val * c)/denom, 5) for c in corr
        ]

        rows.append([sid, sname] + feat_vec)

    # put into DataFrame
    cols = ["ID","Name"] + feat_desc + [f"PS_Feat_{i}" for i in range(nfeatures+1,
                                                                       nfeatures+1+lambda_val)]
    return pd.DataFrame(rows, columns=cols)

import itertools
from collections import Counter
import numpy as np

def calc_correlation_vector(seq, lambda_val, prop_df):
    mu_val = prop_df.shape[1] - 1
    seq_len = len(seq)
    corr_vector = []
    for i in range(1, lambda_val+1):
        corr_sum = 0
        for j in range(1, seq_len-(i+1)):
            dinucl1 = seq[j-1] + seq[j]
            dinucl2 = seq[j+i-1] + seq[j+i]
            prop_vec1 = prop_df.loc[prop_df['Dinucleotide']==dinucl1].iloc[0, 1:].astype(float).values
            prop_vec2 = prop_df.loc[prop_df['Dinucleotide']==dinucl2].iloc[0, 1:].astype(float).values
            # squared‐difference sum
            sq_diff = np.sum((prop_vec1 - prop_vec2)**2)
            corr_sum += sq_diff / mu_val
        corr_vector.append(corr_sum / (seq_len - (i+1)))
    return corr_vector

def compute_pseudo_dnc(seqs, prop_df, lambda_val=8, w_val=0.5):
    # all 16 dinucleotides
    bases = ["A","G","C","U"]
    di_strings = [''.join(p) for p in itertools.product(bases, repeat=2)]
    # name your DNC_Feat columns
    dnc_feat_names = [f"DNC_Feat_{i}" for i in range(17, 17+lambda_val)]

    # collect feature‐vectors for each seq
    all_feats = []
    for seq in seqs:
        # 1) compute normalized dinucleotide freqs
        dinucs = [seq[j:j+2] for j in range(len(seq)-1)]
        cnt = Counter(dinucs)
        norm_freq = [cnt[di]/len(seq) for di in di_strings]

        # 2) compute correlation vector
        corr = calc_correlation_vector(seq, lambda_val, prop_df)

        # 3) build combined feature‐vector
        denom = sum(norm_freq) + w_val*sum(corr)
        # first 16 features
        f1 = [round(f/denom, 5) for f in norm_freq]
        # next lambda_val features
        f2 = [round(w_val*c/denom, 5) for c in corr]

        all_feats.append(f1 + f2)

    # assemble into a DataFrame
    import pandas as pd
    feat_df = pd.DataFrame(all_feats, columns=di_strings + dnc_feat_names)

    print(feat_df)
    return feat_df

def evaluate_mvrm_aptamer(to_predict_csv):
    # —————— 1. load and split out aptamers ——————
    df = pd.read_csv(to_predict_csv)
    apt = df[df["Dataset"].str.contains("Aptamer", na=False)].copy()
    tg_seq = apt["Target_RNA_sequence"].tolist()
    tg_name = apt["RNA_Target"].tolist()
    seqs = apt["Target_RNA_sequence"].astype(str)

    # —————— 2. Mordred descriptors ——————
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in apt["SMILES"]]
    mordred_df = calc.pandas(mols)
    wanted = ["AATSC2i","nG12FRing","GATS5i","SMR_VSA9","MATS3v","ATSC8i","JGI5"]
    mordred_df = mordred_df[wanted].reset_index(drop=True)
    
    # —————— 3. PseSSC_v1 descriptors ——————
    # make sure your CSV has a "Sequence" column
    pssc = compute_psessc_v1(
        seqs  = tg_seq,
        ids   = tg_seq,
        names = tg_name
    ).reset_index(drop=True)

    subset = pssc[["ID", "Name", "G-C,A", "C,G-C"]]   
    # —————— 4. combine and return ——————
    out = pd.concat([apt.reset_index(drop=True),
                     mordred_df, subset.drop(columns=["ID","Name"])], axis=1)
    
    
    # ———— 5. Compute CCA, ACU, GGC frequencies ————
    # We’ll define frequency = count(substring) / (total_possible_windows)
    L = seqs.str.len()

    for tri in ["CCA","ACU","GGC"]:
        # count non‑overlapping occurrences; if you want overlapping, you can use a sliding window
        counts = seqs.str.count(tri)
        out[f"freq_{tri}"] = counts / L  # avoid div0

    # (optional) fill NaN→0 for ultra‑short sequences
    out[["freq_CCA","freq_ACU","freq_GGC"]] = out[["freq_CCA","freq_ACU","freq_GGC"]].fillna(0)

    # —————— 6. Model predictions ——————
    coefficients = {"G-C,A": 2.754e+02, 
                    "AATSC2i": 1.796e+00, 
                    "nG12FRing": 5.241e-01, 
                    "GATS5i": 2.972e+00, 
                    "SMR_VSA9": 2.547e-02, 
                    "freq_ACU": 2.295e+01, 
                    "freq_GGC": 9.870e+00, 
                    "freq_CCA": -2.363e+01, 
                    "C,G-C": 1.857e+02, 
                    "ATSC8i": -1.138e-02, 
                    "JGI5": -2.885e+01, 
                    "MATS3v": -2.595e+00}
    bias = 2.315

    # turn dict → pandas Series, indexed by column name
    coef_ser = pd.Series(coefficients)
    out["predicted_pKd"] = out[coef_ser.index].dot(coef_ser) + bias

    # —————— 7. Print results ——————


    print(out["predicted_pKd"].tolist()[:100])
    """
    for _, row in out.iterrows():
        print(
            f"ID: {row[id_col]}, "
            f"Name: {row[name_col]}, "
            f"SMILES: {row[smiles_col]}, "
            f"RNA Seq: {row[seq_col]}, "
            f"predicted_pKd: {row['predicted_pKd']:.3f}, "
            f"pKd:" f"{row['pKd']:.3f}"
        )
    """
    out["y_pred"] = out["predicted_pKd"] >= 4
    out["y_true"] = out["pKd"] == 1

    y_true   = out["y_true"]
    y_pred   = out["y_pred"]
    y_scores = out["predicted_pKd"]  # for AUROC

    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    auroc  = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AUROC    : {auroc:.3f}")
    print()
    print("Confusion Matrix")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")


def evaluate_mvrm_mirna(to_predict_csv):
    # —————— 1. load and split out aptamers ——————
    df = pd.read_csv(to_predict_csv)
    mirna = df[df["Dataset"].str.contains("miRNA", na=False)].copy()
    tg_seq = mirna["Target_RNA_sequence"].tolist()
    tg_name = mirna["RNA_Target"].tolist()
    seqs = mirna["Target_RNA_sequence"].astype(str)
    seqs = seqs.reset_index(drop=True)

    # —————— 2. Mordred descriptors ——————
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in mirna["SMILES"]]
    mordred_df = calc.pandas(mols)
    wanted = ["PEOE_VSA7","GATS3v","ATSC3i","JGI2"]
    mordred_df = mordred_df[wanted].reset_index(drop=True)
    
    # —————— 3. PseSSC_v1 descriptors ——————
    pssc = compute_psessc_v1(
        seqs  = tg_seq,
        ids   = tg_seq,
        names = tg_name
    ).reset_index(drop=True)
    subset = pssc[["ID", "Name", "C,G-C"]]

    # —————— 4. combine so far ——————
    out = pd.concat([mirna.reset_index(drop=True),
                     mordred_df,
                     subset.drop(columns=["ID","Name"])],
                    axis=1)
    
    # ———— 5. Compute CCA, ACU, GGC frequencies ————
    L = seqs.str.len()
    for tri in ["ACU", "GUCC", "CAUU"]:
        counts = seqs.str.count(tri)
        out[f"freq_{tri}"] = counts / L


    out[["freq_ACU","freq_GUCC","freq_CAUU"]] = \
        out[["freq_ACU","freq_GUCC","freq_CAUU"]].fillna(0)

    # ———— 5b. Compute A... frequency from secondary structure ————
    def freq_A_dotdotdot(seq):
        ss, mfe = RNA.fold(seq)
        ss = ss.replace(")", "(")
        # sliding window count of middle-base A & ss-pattern "..."
        count = sum(
            1
            for j in range(len(seq)-2)
            if seq[j+1] == "A" and ss[j:j+3] == "..."
        )
        total = max(len(seq)-2, 1)
        return count / total

    out["A..."] = seqs.apply(freq_A_dotdotdot)
    # —————— 6. Model predictions ——————
    coefficients = {
        "freq_GUCC":      -6.842e+01,
        "PEOE_VSA7":  1.833e-02,
        "GATS3v":    -6.473e+00,
        "ATSC3i":    -2.429e-02,
        "freq_CAUU":       7.564e+01,
        "freq_ACU":        4.240e+01,
        "A...":       1.805e+01,
        "JGI2":      -1.668e+01
    }
    bias = 10.676

    coef_ser = pd.Series(coefficients)
    out["predicted_pKd"] = out[coef_ser.index].dot(coef_ser) + bias

    # —————— 7. Print & evaluate ——————
    out["y_pred"] = out["predicted_pKd"] >= 4
    out["y_true"] = out["pKd"] == 1

    y_true   = out["y_true"]
    y_pred   = out["y_pred"]
    y_scores = out["predicted_pKd"]  # for AUROC

    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    auroc  = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AUROC    : {auroc:.3f}")
    print()
    print("Confusion Matrix")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")
    return out


from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import mordred

def evaluate_mvrm_general_viral(to_predict_csv):
    # 1. load & filter riboswitches
    df    = pd.read_csv(to_predict_csv)
    mirna = df[df["Dataset"].str.contains("Ribos", na=False)].copy()
    
    # 2. pull raw sequences & clean them → seqs_new
    raw_seqs = mirna["Target_RNA_sequence"].astype(str)
    seq_repl = {"X":"A","N":"A"," ":"","R":"G","Y":"C","K":"G",
                "M":"A","S":"G","W":"A","B":"G","D":"G","H":"A","V":"G"}
    seqs_new = []
    for s in raw_seqs:
        s = s.replace("T","U")
        for ch, rep in seq_repl.items():
            s = s.replace(ch, rep)
        seqs_new.append(s)
    seqs = pd.Series(seqs_new).reset_index(drop=True)
    tg_seq  = seqs.tolist()
    tg_name = mirna["RNA_Target"].tolist()
    
    # 3. Mordred descriptors + GeometricalShapeIndex
    calc       = Calculator(descriptors, ignore_3D=True)
    mols       = [Chem.MolFromSmiles(smi) for smi in mirna["SMILES"]]
    mordred_df = calc.pandas(mols)
    wanted     = ["SssS","nAcid","GATS3p","GATS3i","SMR_VSA6","SlogP_VSA10", "SssNH"]
    mordred_df = mordred_df[wanted].reset_index(drop=True)

    # 4. PseSSC_v1 descriptors
    pssc   = compute_psessc_v1(seqs=tg_seq, ids=tg_seq, names=tg_name).reset_index(drop=True)
    subset = pssc[["ID","Name","A,A-U","G-C,A"]]
    
    # 5. combine all features
    out = pd.concat([
        mirna.reset_index(drop=True),
        mordred_df,
        subset.drop(columns=["ID","Name"])
    ], axis=1)
    
    # 6. tri‐nuc freqs
    L = seqs.str.len()
    for tri in ["AGA"]:
        out[f"freq_{tri}"] = seqs.str.count(tri) / L
    out[["freq_AGA"]] = out[["freq_AGA"]].fillna(0)

    # Report before cleaning
    before_n = out.shape[0]
    print(f"Rows before removing NaNs: {before_n}")
    # drop any rows with NaNs in features used for prediction
    features = ["freq_AGA","SlogP_VSA10","SssNH","SssS","nAcid",
                "GATS3p","GATS3i","SMR_VSA6"]
    out_clean = out.dropna(subset=features).reset_index(drop=True)
    after_n = out_clean.shape[0]
    print(f"Rows after removing NaNs:  {after_n}")

    # 9. prediction
    coefficients = {
        "freq_AGA":   -14.238,
        "SlogP_VSA10":  -0.066,
        "SssNH":     0.093,
        "SssS":    0.752,
        "nAcid":    1.990,
        "GATS3p":    3.131,
        "GATS3i": -3.214,
        "SMR_VSA6":   0.015   }
    bias = 5.458
    coef_ser = pd.Series(coefficients)
    out_clean["predicted_pKd"] = out_clean[coef_ser.index].dot(coef_ser) + bias

    # 10. evaluate
    out_clean["y_pred"] = out_clean["predicted_pKd"] >= 4
    out_clean["y_true"] = out_clean["pKd"] == 1
    y_true, y_pred, y_scores = out_clean["y_true"], out_clean["y_pred"], out_clean["predicted_pKd"]
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    auroc  = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AUROC    : {auroc:.3f}")
    print("\nConfusion Matrix")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")

    return out_clean
#
def evaluate_mvrm_riboswitches(to_predict_csv, prop_data_tsv):
    # 1. load & filter riboswitches
    df    = pd.read_csv(to_predict_csv)
    mirna = df[df["Dataset"].str.contains("Ribos", na=False)].copy()
    
    # 2. pull raw sequences & clean them → seqs_new
    raw_seqs = mirna["Target_RNA_sequence"].astype(str)
    seq_repl = {"X":"A","N":"A"," ":"","R":"G","Y":"C","K":"G",
                "M":"A","S":"G","W":"A","B":"G","D":"G","H":"A","V":"G"}
    seqs_new = []
    for s in raw_seqs:
        s = s.replace("T","U")
        for ch, rep in seq_repl.items():
            s = s.replace(ch, rep)
        seqs_new.append(s)
    seqs = pd.Series(seqs_new).reset_index(drop=True)
    tg_seq  = seqs.tolist()
    tg_name = mirna["RNA_Target"].tolist()
    
    # 3. Mordred descriptors + GeometricalShapeIndex
    calc       = Calculator(descriptors, ignore_3D=True)
    mols       = [Chem.MolFromSmiles(smi) for smi in mirna["SMILES"]]
    mordred_df = calc.pandas(mols)
    wanted     = ["Xp-6d","PEOE_VSA13","ATSC7d","AATS2dv","AATSC6i","GATS2i"]
    mordred_df = mordred_df[wanted].reset_index(drop=True)

    # calculate 3D shape index
    geom_calc = descriptors.GeometricalIndex.GeometricalShapeIndex()
    shape_vals = []
    for smi in mirna["SMILES"]:
        m2d = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m2d)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(m, params) != 0:
            print(f"⚠️ Embed failed for {smi}, skipping shape index")
            shape_vals.append(float('nan'))
            continue
        AllChem.MMFFOptimizeMolecule(m)
        shape_vals.append(geom_calc(m))
    mordred_df["GeomShapeIndex"] = shape_vals
    
    # 4. PseSSC_v1 descriptors
    pssc   = compute_psessc_v1(seqs=tg_seq, ids=tg_seq, names=tg_name).reset_index(drop=True)
    subset = pssc[["ID","Name","A,A-U","G-C,A"]]
    
    # 5. combine all features
    out = pd.concat([
        mirna.reset_index(drop=True),
        mordred_df,
        subset.drop(columns=["ID","Name"])
    ], axis=1)
    
    # 6. tri‐nuc freqs
    L = seqs.str.len()
    for tri in ["GAA","CUG"]:
        out[f"freq_{tri}"] = seqs.str.count(tri) / L
    out[["freq_GAA","freq_CUG"]] = out[["freq_GAA","freq_CUG"]].fillna(0)

    # 7. U(.. feature
    def freq_U_lpar_dotdot(seq):
        ss, mfe = RNA.fold(seq)
        ss = ss.replace(")","(")
        count = sum(1 for j in range(len(seq)-2) if seq[j+1]=="U" and ss[j:j+3]=="(..")
        return count / max(len(seq)-2, 1)
    out["U(.."] = seqs.apply(freq_U_lpar_dotdot)

    # 8. DNC features
    prop_df = pd.read_csv(prop_data_tsv, sep='\t', header=0)
    dnc_df  = compute_pseudo_dnc(seqs_new, prop_df, lambda_val=8, w_val=0.5)
    out["DNC_Feat_17"] = dnc_df["DNC_Feat_17"].values

    # Report before cleaning
    before_n = out.shape[0]
    print(f"Rows before removing NaNs: {before_n}")

    # drop any rows with NaNs in features used for prediction
    feature_cols = [
        "freq_GAA","PEOE_VSA13","ATSC7d","AATS2dv","Xp-6d","freq_CUG",
        "U(..","DNC_Feat_17","GATS2i","GeomShapeIndex","G-C,A","A,A-U","AATSC6i"
    ]
    out_clean = out.dropna(subset=feature_cols).reset_index(drop=True)
    after_n = out_clean.shape[0]
    print(f"Rows after removing NaNs:  {after_n}")

    # 9. prediction
    coefficients = {
        "freq_GAA":   1.158e+02,
        "Xp-6d": 5.837e-01,
        "PEOE_VSA13":  1.363e-01,
        "ATSC7d":     1.065e-01,
        "AATS2dv":    3.396e-01,
        "freq_CUG":    -1.315e+02,
        "U(..":        1.445e+02,
        "DNC_Feat_17": -1.342e+02,
        "AATSC6i": -2.525e+00,
        "GATS2i":    2.644e+00,
        "G-C,A": 1.3402e+03,
        "A,A-U": -8.762e+02,
        "GeomShapeIndex": 5.431e+00,
    }
    bias = 5.631
    # 1. build coefficient Series
    coef_ser = pd.Series(coefficients)

    # 2. select numeric features and drop rows with any NaNs
    feats = out_clean[coef_ser.index] \
            .apply(pd.to_numeric, errors='coerce') \
            .astype(float) \
            .dropna(axis=0, subset=coef_ser.index)

    # 3. subset out_clean to exactly those rows (still using the old index)
    out_clean = out_clean.loc[feats.index]

    # 4. compute and assign predictions while indices still line up
    out_clean["predicted_pKd"] = feats.dot(coef_ser) + bias

    # 5. now reset the index
    out_clean = out_clean.reset_index(drop=True)

    # 6. evaluate
    out_clean["y_pred"] = out_clean["predicted_pKd"] >= 4
    out_clean["y_true"] = out_clean["pKd"] == 1

    # 7. pull out your arrays
    y_true, y_pred = out_clean["y_true"], out_clean["y_pred"]
    y_scores = out_clean["predicted_pKd"]
        # Print the number of NaNs in y_pred and y_true
    print(f"NaNs in y_pred: {y_pred.isna().sum()}")
    print(f"NaNs in y_true: {y_true.isna().sum()}")
    print(f"NaNs in y_scores: {y_scores.isna().sum()}")

    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    auroc  = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AUROC    : {auroc:.3f}")
    print("\nConfusion Matrix")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")

    return out_clean

"""
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

mol = Chem.SDMolSupplier('raw.sdf', removeHs=False)[3]
# Explicitly set up UFF and fix random seed

# RDKit doesn’t have a built-in “Geometric Shape Index,” but you could
# compute an analogous surface‑based descriptor with rdMolDescriptors.CalcPBF(mol)
gsi = mordred.GeometricalIndex.GeometricalShapeIndex()
gsi_value = gsi(mol)/1.09
print("Geometric Shape Index (GSI) value:", gsi_value)


"""


evaluate_mvrm_riboswitches("/media/pako/z/Master-Thesis/RNA-Based-DTI/to_predict.csv", "data/author-performance/Physicochemical_indices_RNA.csv")
evaluate_mvrm_mirna("/media/pako/z/Master-Thesis/RNA-Based-DTI/to_predict.csv")
evaluate_mvrm_aptamer("/media/pako/z/Master-Thesis/RNA-Based-DTI/to_predict.csv")
evaluate_mvrm_general_viral("/media/pako/z/Master-Thesis/RNA-Based-DTI/to_predict.csv")
