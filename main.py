import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold

from msi.msi import MSI


def load_features(feature_file_path, idx2node):
    features = []
    with open(feature_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        for line in file:
            parts = line.strip().split()
            node_idx = int(parts[0][2:])
            node = idx2node[node_idx]
            feature = [float(x) for x in parts[1:]]
            features.append((node, feature))
    return features


def concatenate_features(drug_features, indication_features, samples):
    concatenated_features = []
    drug_dict = dict(drug_features)
    indication_dict = dict(indication_features)
    for drug, indication in samples:
        if drug in drug_dict and indication in indication_dict:
            concatenated_features.append(
                np.concatenate([drug_dict[drug], indication_dict[indication]])
            )
    return concatenated_features


def generate_negative_samples(positive_samples, num_negative_samples, drugs, indications):
    negative_samples = []
    existing = set(positive_samples)
    rng = np.random.default_rng(1037)
    while len(negative_samples) < num_negative_samples:
        negative_pair = (rng.choice(drugs), rng.choice(indications))
        if negative_pair not in existing:
            existing.add(negative_pair)
            negative_samples.append(negative_pair)
    return negative_samples


def build_feature_sets(idx2node, positive_samples, negative_samples, feature_paths):
    features_list = []
    labels_list = []
    for path in feature_paths:
        drug_feats = load_features(path, idx2node)
        indication_feats = load_features(path, idx2node)
        pos_features = concatenate_features(drug_feats, indication_feats, positive_samples)
        neg_features = concatenate_features(drug_feats, indication_feats, negative_samples)
        features = np.vstack((pos_features, neg_features))
        labels = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])
        features_list.append(features)
        labels_list.append(labels)
    return features_list, labels_list


def train_xgb_max(features_list, labels_list, params, random_seed=12, num_round=100):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fpr_list": [],
        "tpr_list": [],
        "precision_curves": [],
        "recall_curves": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features_list[0], labels_list[0])):
        y_preds = []
        for feats, labels in zip(features_list, labels_list):
            X_train, X_val = feats[train_idx], feats[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            booster = xgb.train(
                params,
                xgb.DMatrix(X_train, label=y_train),
                num_boost_round=num_round,
            )
            y_preds.append(booster.predict(xgb.DMatrix(X_val)))

        y_pred_max = np.max(np.vstack(y_preds), axis=0)
        y_val = labels_list[0][val_idx]

        fpr, tpr, _ = roc_curve(y_val, y_pred_max)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_max)

        metrics["roc_auc"].append(roc_auc_score(y_val, y_pred_max))
        metrics["pr_auc"].append(auc(recall_curve, precision_curve))
        metrics["accuracy"].append(accuracy_score(y_val, np.round(y_pred_max)))
        metrics["precision"].append(precision_score(y_val, np.round(y_pred_max)))
        metrics["recall"].append(recall_score(y_val, np.round(y_pred_max)))
        metrics["f1"].append(f1_score(y_val, np.round(y_pred_max)))
        metrics["fpr_list"].append(fpr)
        metrics["tpr_list"].append(tpr)
        metrics["precision_curves"].append(precision_curve)
        metrics["recall_curves"].append(recall_curve)

    return metrics


def main():
    random_seed = 12

    msi = MSI()
    msi.load()
    idx2node = msi.idx2node

    df = pd.read_csv("data/6_drug_indication_df.tsv", sep="\t")
    positive_samples = list(zip(df["drug"], df["indication"]))

    result_files = glob.glob(
        #"results/*.npy"#
    )

    drug_features = []
    indication_features = []
    for file_path in result_files:
        features = np.load(file_path)
        node_name = file_path.split("\\")[-1].split("_")[0]
        if node_name.startswith("DB"):
            drug_features.append((node_name, features))
        else:
            indication_features.append((node_name, features))

    drugs = [node for node, _ in drug_features]
    indications = [node for node, _ in indication_features]
    negative_samples = generate_negative_samples(
        positive_samples, len(positive_samples), drugs, indications
    )

    feature_paths = [
        "feature/d2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
        "feature/d2p2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
        "feature/d2p2f2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
        "feature/d2p2f2f2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
    ]
    features_list, labels_list = build_feature_sets(
        idx2node, positive_samples, negative_samples, feature_paths
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 12,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "gamma": 0.1,
    }

    metrics = train_xgb_max(features_list, labels_list, params, random_seed=random_seed)

    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        print(f"{key.upper()} Mean: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}")


if __name__ == "__main__":
    main()
