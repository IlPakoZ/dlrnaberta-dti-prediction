import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

def calculate_metrics():
    df = pd.read_csv("test_predictions.csv")
    all_predicts = pd.read_csv("to_predict.csv")
    
    # Merge on "SMILES", "Target_RNA_sequence"
    df = df.merge(all_predicts[["SMILES", "Target_RNA_sequence", "Dataset"]], on=["SMILES", "Target_RNA_sequence"], how="left")

    results = []

    threshold = 4.0
    for dataset, group in df.groupby("Dataset"):
        y_true = (group["true_pKd"]).astype(int)
        y_pred = (group["predicted_pKd"] >= threshold).astype(int)

        y_score = group["predicted_pKd"]  # required for AUROC

        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_score)

        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = recall_score(y_true, y_pred)

        results.append({
            "Dataset": dataset,
            "Precision": precision,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "Recall": recall,
            "AUROC": auroc,
            "F1 Score": f1
        })

        print(results[-1])  # Print the latest result

    return pd.DataFrame(results)

calculate_metrics()