import numpy as np
import pandas as pd
import xgboost as xgb

from msi.msi import MSI


FEATURE_PATH = "feature/d2p2i_1000_100_++_s128_w5_n5_ts1e9.txt"
INDICATION_ID = "C0002395" # Alzheimer's disease
MODEL_PATH = "model/max_xgb.model"
ASSOCIATION_PATH = "data/6_drug_indication_df.tsv"


def load_features(feature_file_path, idx2node):
    features = {}
    with open(feature_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        for line in file:
            parts = line.strip().split()
            node_idx = int(parts[0][2:])
            node = idx2node[node_idx]
            features[node] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return features


def main():
    msi = MSI()
    msi.load()
    idx2node = msi.idx2node

    df = pd.read_csv(ASSOCIATION_PATH, sep="\t")
    drug_list = df["drug"].unique()

    feature_map = load_features(FEATURE_PATH, idx2node)
    target_feature = feature_map.get(INDICATION_ID)
    if target_feature is None:
        raise ValueError(f"Missing feature vector for indication {INDICATION_ID}")

    feature_matrix = []
    retained_drugs = []
    for drug in drug_list:
        drug_feature = feature_map.get(drug)
        if drug_feature is None:
            continue
        feature_matrix.append(np.concatenate([drug_feature, target_feature]))
        retained_drugs.append(drug)

    if not feature_matrix:
        raise ValueError("No drug features were found.")

    dtest = xgb.DMatrix(np.vstack(feature_matrix))
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    predictions = booster.predict(dtest)

    results = (
        pd.DataFrame({"Drug": retained_drugs, "Prediction": predictions})
        .sort_values("Prediction", ascending=False)
        .reset_index(drop=True)
    )
    pd.set_option("display.max_rows", None)
    print(results)


if __name__ == "__main__":
    main()
