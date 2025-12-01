#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on SUN NOV 30 11:22:09 2025

@author: ENDOUC - An EndoCV2025 @IEEE ISBI Challenge

"""
import pandas as pd
import argparse 
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def metrics_endouc(y_true, y_pred):
    # ---------------------------------------------------------
    # Sensitivity, Specificity, PPV per class
    # ---------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)  # 4x4 matrix
    num_classes = cm.shape[0]

    sensitivity = {}
    specificity = {}
    ppv = {}

    for cls in range(num_classes):
        TP = cm[cls, cls]
        FN = cm[cls, :].sum() - TP
        FP = cm[:, cls].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity[cls] = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0
    return sensitivity, specificity, ppv


def read_json(jsonFile):
    with open(jsonFile) as json_data:
        data = json.load(json_data)
        return data
    

def get_args():
    parser = argparse.ArgumentParser(description="EndoUC - An EndoCV2025 @IEEE ISBI Challenge")
    parser.add_argument("-gtFile", type=str, default="examples/gt.csv", help="ground truth file in csv")
    parser.add_argument("-predFile", type=str, default="examples/pred.csv", help="prediction file in csv")
    parser.add_argument("-caseType", type=int, default=3, help="MES classification only (1), UCEIS classification only (2), MES classification and " \
    "captioning (3) ")
    parser.add_argument("-csvFileName", type=str, default="EndoUC25_Classification_Metrics_", help="all evaluation scores used for grading")
    args = parser.parse_args()
    return args
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    valArgs = get_args()

    "Read GT file"
    df_gt = pd.read_csv(valArgs.gtFile)

    "Read image ids and classification for MES/UCEIS scoring and captioning"
    id_col_gt = "id"                 # ID column name in ground truth file
    label_col_gt_MES = "mes_scoring_0_3"        # Ground truth label column (0,1,2,3)
    label_col_gt_UCEIS = "uceis_score" 
    label_col_gt_Captions = "captions" 

    """ case: MES classification """

    if (valArgs.caseType == 1) or (valArgs.caseType == 3):
       num_classes = 3

       df_gt = df_gt[[id_col_gt, label_col_gt_MES]]
       df_gt['mes_scoring_0_3'] = df_gt['mes_scoring_0_3'].replace({'MES-0': 0, 'MES-1': 1, 'MES-2': 2, 'MES-3': 3})

       print(df_gt)

       exists = os.path.isfile(valArgs.predFile) 

       if exists:
            df_pred = pd.read_csv(valArgs.predFile)
            print("Prediction file columns:", df_pred.columns.tolist())

            id_col_pred = "id"
            label_col_pred_MES = "mes_scoring_0_3" 
            
            df_pred = df_pred[[id_col_pred, label_col_pred_MES]]
            # Map MES-0, MES-1, MES-2, MES-3 to 0, 1, 2, 3
            df_pred[label_col_pred_MES] = df_pred[label_col_pred_MES].replace({'MES-0': 0, 'MES-1': 1, 'MES-2': 2, 'MES-3': 3})
            print(df_pred)  

            # # Merge using the ID column
            # df = pd.merge(df_gt, df_pred,
            #             left_on=id_col_gt,
            #             right_on=id_col_pred,
            #             how="inner")
            # print(df)

            y_true = df_gt[label_col_pred_MES].astype(int)
            y_pred = df_pred[label_col_pred_MES].astype(int)

    """ case: UCEIS classification """
    if valArgs.caseType == 2:
       
       num_classes = 8 #0-7
       label_col = "uceis_score"
       df_gt = df_gt[[id_col_gt, label_col]]
       df_gt[label_col] = df_gt[label_col].replace({'UCEIS-0': 0, 'UCEIS-1': 1, 'UCEIS-2': 2, 'UCEIS-3': 3, 'UCEIS-4': 4, 'UCEIS-5': 5, 'UCEIS-6': 6, 'UCEIS-7': 7, 'UCEIS-8': 8})
       exists = os.path.isfile(valArgs.predFile) 

       if exists:
            df_pred = pd.read_csv(valArgs.predFile)
            print("Prediction file columns:", df_pred.columns.tolist())
            id_col_pred = "id"
            df_pred = df_pred[[id_col_pred, label_col]] 
            df_pred[label_col] = df_pred[label_col].replace({'UCEIS-0': 0, 'UCEIS-1': 1, 'UCEIS-2': 2, 'UCEIS-3': 3, 'UCEIS-4': 4, 'UCEIS-5': 5, 'UCEIS-6': 6, 'UCEIS-7': 7, 'UCEIS-8': 8})
            print(df_pred)  
    
            # Merge using the ID column UCEIS-1
            # df = pd.merge(df_gt, df_pred,
            #             left_on=id_col_gt,
            #             right_on=id_col_pred,
            #             how="inner")
            
            y_true = df_gt[label_col].astype(int)
            y_pred = df_pred[label_col].astype(int)


    """ case: MES classification and captioning """
    if valArgs.caseType == 3:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
        
        df_gt = pd.read_csv(valArgs.gtFile)

        # pass
        # To be implemented
        label_col_Captions = "captions"
        df_gt = df_gt[["id", label_col_Captions]]
       
        # Read prediction file
        df_pred = pd.read_csv(valArgs.predFile)
        pred_caption_col = "predictions" #explicitly set prediction for caption header!!!
        df_pred = df_pred[["id", pred_caption_col]]

        # Merge on ID
        df = pd.merge(df_gt, df_pred, left_on="id", right_on="id", suffixes=("_gt", "_pred"))
        
        # print(df)
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        # Store results
        results = []   
        for _, row in df.iterrows():
            print(str(row[label_col_Captions]).strip())
            gt = str(row[label_col_Captions]).strip()
            pred = str(row[pred_caption_col]).strip()
            # -----------------------------
            # Cosine similarity (TF-IDF)
            # -----------------------------
            vect = TfidfVectorizer().fit([gt, pred])
            tfidf = vect.transform([gt, pred])
            cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

            print(cos_sim)
            
            # -----------------------------
            # BLEU scores (BLEU-1 → BLEU-4)
            # pip install --upgrade nltk
            # -----------------------------
            # print([gt.split()])
            # print(pred.split())

            # BLEU-1 (unigram)
            bleu1 = sentence_bleu([gt.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)

            # # BLEU-2 (bigram)
            bleu2 = sentence_bleu([gt.split()], pred.split(),
                                weights=(0.5, 0.5, 0, 0),
                                smoothing_function=SmoothingFunction().method1)

            # BLEU-3 (trigram)
            bleu3 = sentence_bleu([gt.split()], pred.split(),
                                weights=(0.33, 0.33, 0.33, 0),
                                smoothing_function=SmoothingFunction().method1)

            # BLEU-4 (standard BLEU)
            bleu4 = sentence_bleu([gt.split()], pred.split(),
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=SmoothingFunction().method1)

            # -----------------------------
            # METEOR score
            # -----------------------------
            meteor = meteor_score([gt.split()], pred.split())
            # print(meteor)

            # -----------------------------
            # ROUGE scores
            # -----------------------------
            rouge = scorer.score(gt, pred)
            rouge1 = rouge['rouge1'].fmeasure
            rougeL = rouge['rougeL'].fmeasure

            # Append all results
            results.append({
                "id": row[label_col_Captions],
                "cosine_similarity": cos_sim,
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4,   # Standard BLEU
                "meteor": meteor,
                "rouge1": rouge1,
                "rougeL": rougeL
            })

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Print results
        print("\n===== Individual Caption Metrics =====\n")
        print(df_results)

        print("\n===== Average Metrics =====\n")
        mean_metrics = df_results.mean(numeric_only=True)
        print('caption mean metric values:', mean_metrics)
        mean_metrics.to_csv("Captioning_Metrics_Output_mean.csv", header=False)

        # output_file_mean = "Captioning_Metrics_Output_mean.csv"
        # mean_metrics.to_csv(output_file_mean, index=False)

        output_file = "Captioning_Metrics_Output.csv"
        df_results.to_csv(output_file, index=False)

        print(f"\nMetrics saved to: {output_file}")




# ---------------------------------------------------------
# 1. TOP-1 ACCURACY
# ---------------------------------------------------------
top1_accuracy = accuracy_score(y_true, y_pred)

# ---------------------------------------------------------
# 2. F1 SCORE (macro)
# ---------------------------------------------------------
f1_macro = f1_score(y_true, y_pred, average="macro")

# ---------------------------------------------------------
# 3. MULTICLASS AUC (macro)
# ---------------------------------------------------------
# 1. Define the full list of required columns (e.g., [0, 1, 2, 3] for MES)
all_classes = list(range(num_classes))

# 2. Convert to one-hot, then reindex to ensure all columns exist.
#    fill_value=0 is crucial to set missing class columns to 0.
y_true_oh = pd.get_dummies(y_true).reindex(columns=all_classes, fill_value=0)
y_pred_oh = pd.get_dummies(y_pred).reindex(columns=all_classes, fill_value=0)

# 3. Ensure the column order is identical and numerical for roc_auc_score
#    (The reindex step already aligns the columns, but this step confirms the data type)
y_true_oh = y_true_oh[all_classes]
y_pred_oh = y_pred_oh[all_classes]

try:
    auc_macro = roc_auc_score(y_true_oh, y_pred_oh, average="macro")
except ValueError:
    auc_macro = float('nan')

# 4. Calculate AUC (y_true_oh and y_pred_oh now have the exact same columns)
sensitivity, specificity, ppv = metrics_endouc(y_true, y_pred)

# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------
print("\n===== MULTICLASS METRICS =====")
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"AUC (Macro): {auc_macro:.4f}")

print("\n--- Per-Class Metrics (0,1,2,3) ---")

for cls in range(num_classes):
    print(f"\nClass {cls}:")
    try:
        sens = sensitivity[cls]
        spec = specificity[cls]
        p = ppv[cls]

        print(f"  Sensitivity (Recall): {sens:.4f}")
        print(f"  Specificity:          {spec:.4f}")
        print(f"  PPV (Precision):      {p:.4f}")

    except KeyError:

        print("  ↳ Metrics missing for this class.")

# ---------------------------------------------------------
# SAVE METRICS TO CSV FILES
# ---------------------------------------------------------

# Overall metrics
overall_df = pd.DataFrame({
    "Metric": ["Top1_Accuracy", "F1_Macro", "AUC_Macro"],
    "Value": [top1_accuracy, f1_macro, auc_macro]
})

fileName = valArgs.csvFileName+str(valArgs.caseType)+'.csv'
overall_df.to_csv(fileName, index=False)

# Per-class metrics
per_class_df = pd.DataFrame({
    "Class": list(range(num_classes)),
    "Sensitivity": [sensitivity.get(c, np.nan) for c in range(num_classes)],
    "Specificity": [specificity.get(c, np.nan) for c in range(num_classes)],
    "PPV": [ppv.get(c, np.nan) for c in range(num_classes)]
})
per_class_df.to_csv("per_class_"+ fileName, index=False)

print("\nCSV files successfully saved:")
print(fileName)
print("per_class_"+ fileName)