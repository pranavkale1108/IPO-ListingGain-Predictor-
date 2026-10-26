"""
train_real_models.py
--------------------
End-to-end retraining pipeline for the real-world Indian IPO dataset.

Fits:
1. IPOFeaturePipeline (standardization & sector encoding)
2. K-Means Clustering for archetype categorization
3. k-NN Peer Similarity Engine (strict prior-date retrieval)
4. Regression Suite (Linear, Polynomial, Ridge, Gradient Boosting)
5. Classification Suite (Logistic Regression, Gradient Boosting, SVM)
6. Leaderboard and Metrics Export
"""

from typing import Dict, Any, List
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.preprocessor import IPOFeaturePipeline
from src.similar_ipos import IPOSimilarityFinder


def run_pipeline():
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    FIG_DIR = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load Chronological Splits
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"Loaded Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # 2. Fit Preprocessor Pipeline
    pipeline = IPOFeaturePipeline()
    train_prep = pipeline.fit_transform(train_df)
    test_prep = pipeline.transform(test_df)
    pipeline.save(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    base_feature_cols = pipeline.final_feature_names

    # 3. K-Means Archetype Clustering (k=3)
    km = KMeans(n_clusters=3, random_state=42, n_init=15)
    train_clusters = km.fit_predict(train_prep[base_feature_cols])
    test_clusters = km.predict(test_prep[base_feature_cols])

    train_prep["cluster_id"] = train_clusters
    test_prep["cluster_id"] = test_clusters

    # Name clusters based on average GMP & gain
    cluster_profiles = train_df.groupby(train_clusters)["listing_gain_pct"].mean()
    cluster_names = {}
    sorted_clusters = cluster_profiles.sort_values()
    cluster_names[sorted_clusters.index[0]] = "Compressed & Discount Debuts"
    cluster_names[sorted_clusters.index[1]] = "Steady Mid-Range Offerings"
    cluster_names[sorted_clusters.index[2]] = "High-Momentum GMP Leaders"

    train_prep["cluster_name"] = train_prep["cluster_id"].map(cluster_names)
    test_prep["cluster_name"] = test_prep["cluster_id"].map(cluster_names)

    for i in range(3):
        train_prep[f"cluster_{i}"] = (train_prep["cluster_id"] == i).astype(float)
        test_prep[f"cluster_{i}"] = (test_prep["cluster_id"] == i).astype(float)

    all_feature_cols = base_feature_cols + [f"cluster_{i}" for i in range(3)]

    # Save Preprocessed Files
    train_prep.to_csv(os.path.join(DATA_DIR, "train_preprocessed.csv"), index=False)
    test_prep.to_csv(os.path.join(DATA_DIR, "test_preprocessed.csv"), index=False)
    train_df["cluster_id"] = train_clusters
    train_df["cluster_name"] = train_prep["cluster_name"]
    test_df["cluster_id"] = test_clusters
    test_df["cluster_name"] = test_prep["cluster_name"]
    train_df.to_csv(os.path.join(DATA_DIR, "train_clustered.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_clustered.csv"), index=False)

    # 4. Peer Similarity Finder
    sim_finder = IPOSimilarityFinder(
        historical_df=train_df,
        feature_matrix=train_prep,
        feature_cols=all_feature_cols,
    )
    joblib.dump(sim_finder, os.path.join(MODELS_DIR, "knn_similarity.joblib"))

    # Save KMeans Artifact
    joblib.dump({
        "model": km,
        "k": 3,
        "cluster_names": cluster_names,
        "feature_cols": base_feature_cols,
    }, os.path.join(MODELS_DIR, "kmeans_model.joblib"))

    # 5. Train & Evaluate Regressors
    X_train_df = train_prep[all_feature_cols]
    y_train_gain = train_prep["listing_gain_pct"].values
    X_test_df = test_prep[all_feature_cols]
    y_test_gain = test_prep["listing_gain_pct"].values

    reg_models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=30, max_depth=2, learning_rate=0.08, random_state=42),
    }

    metrics_rows = []

    for name, model in reg_models.items():
        model.fit(X_train_df, y_train_gain)
        pred_train = model.predict(X_train_df)
        pred_test = model.predict(X_test_df)

        test_mse = mean_squared_error(y_test_gain, pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_gain, pred_test)
        test_r2 = r2_score(y_test_gain, pred_test)

        metrics_rows.append({
            "Model": name,
            "Type": "Regressor",
            "Train_MSE": round(mean_squared_error(y_train_gain, pred_train), 2),
            "Test_MSE": round(test_mse, 2),
            "Test_RMSE": round(test_rmse, 2),
            "Test_MAE": round(test_mae, 2),
            "Test_R2": round(test_r2, 4),
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "AUC": None,
        })

    # Pick Best Regressor (GBR)
    best_gbr = reg_models["GradientBoostingRegressor"]
    best_gbr.feature_names_in_ = np.array(all_feature_cols)
    joblib.dump(best_gbr, os.path.join(MODELS_DIR, "gradient_boosting_regressor.joblib"))
    joblib.dump(reg_models["LinearRegression"], os.path.join(MODELS_DIR, "linear_regression_multi.joblib"))

    # 6. Train & Evaluate Classifiers
    y_train_pos = train_prep["positive_listing"].values
    y_test_pos = test_prep["positive_listing"].values

    clf_models = {
        "LogisticRegression": LogisticRegression(C=1.0, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.08, random_state=42),
        "kNN_Classifier": KNeighborsClassifier(n_neighbors=3),
    }

    for name, model in clf_models.items():
        model.fit(X_train_df, y_train_pos)
        pred_test = model.predict(X_test_df)
        prob_test = model.predict_proba(X_test_df)[:, 1] if hasattr(model, "predict_proba") else pred_test

        acc = accuracy_score(y_test_pos, pred_test)
        prec = precision_score(y_test_pos, pred_test, zero_division=0)
        rec = recall_score(y_test_pos, pred_test, zero_division=0)
        f1 = f1_score(y_test_pos, pred_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test_pos, prob_test)
        except Exception:
            auc = 0.5

        metrics_rows.append({
            "Model": name,
            "Type": "Classifier",
            "Train_MSE": None,
            "Test_MSE": None,
            "Test_RMSE": None,
            "Test_MAE": None,
            "Test_R2": None,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
        })

    best_gbc = clf_models["GradientBoostingClassifier"]
    joblib.dump(best_gbc, os.path.join(MODELS_DIR, "gradient_boosting_classifier.joblib"))
    joblib.dump(clf_models["LogisticRegression"], os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # 7. Train SVM for Verdict
    y_train_verdict = train_prep["verdict"].values
    y_test_verdict = test_prep["verdict"].values
    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_train_df, y_train_verdict)
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm_verdict.joblib"))
    svm_acc = accuracy_score(y_test_verdict, svm.predict(X_test_df))

    metrics_rows.append({
        "Model": "SVM_Verdict",
        "Type": "Classifier (Multi-class)",
        "Train_MSE": None,
        "Test_MSE": None,
        "Test_RMSE": None,
        "Test_MAE": None,
        "Test_R2": None,
        "Accuracy": round(svm_acc, 4),
        "Precision": None,
        "Recall": None,
        "F1": None,
        "AUC": None,
    })

    # Save Metrics & Leaderboard
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "metrics.csv"), index=False)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel Evaluation Leaderboard:")
    print(metrics_df.to_string(index=False))

    # Ablation Study on Real Data
    pred_full = best_gbr.predict(X_test_df)
    mse_full = mean_squared_error(y_test_gain, pred_full)
    r2_full = r2_score(y_test_gain, pred_full)

    # Without GMP
    no_gmp_cols = [c for c in all_feature_cols if not c.startswith("gmp")]
    gbr_no_gmp = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42)
    gbr_no_gmp.fit(X_train_df[no_gmp_cols], y_train_gain)
    pred_no_gmp = gbr_no_gmp.predict(X_test_df[no_gmp_cols])
    mse_no_gmp = mean_squared_error(y_test_gain, pred_no_gmp)
    r2_no_gmp = r2_score(y_test_gain, pred_no_gmp)

    # Without Clusters/Sector (GMP & Price Only)
    only_gmp_cols = ["issue_price", "gmp_last_day", "gmp_implied_gain_pct"]
    gbr_pure_gmp = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42)
    gbr_pure_gmp.fit(X_train_df[only_gmp_cols], y_train_gain)
    pred_pure_gmp = gbr_pure_gmp.predict(X_test_df[only_gmp_cols])
    mse_pure_gmp = mean_squared_error(y_test_gain, pred_pure_gmp)
    r2_pure_gmp = r2_score(y_test_gain, pred_pure_gmp)

    ablation_df = pd.DataFrame([
        {"Ablation Configuration": "Full Model (All Features)", "Feature Count": len(all_feature_cols), "Test MSE": round(mse_full, 2), "Test RMSE": round(np.sqrt(mse_full), 2), "Test R2": round(r2_full, 4)},
        {"Ablation Configuration": "Without GMP (No gmp_last_day / implied)", "Feature Count": len(no_gmp_cols), "Test MSE": round(mse_no_gmp, 2), "Test RMSE": round(np.sqrt(mse_no_gmp), 2), "Test R2": round(r2_no_gmp, 4)},
        {"Ablation Configuration": "GMP & Price Only (No Sector/Clusters)", "Feature Count": len(only_gmp_cols), "Test MSE": round(mse_pure_gmp, 2), "Test RMSE": round(np.sqrt(mse_pure_gmp), 2), "Test R2": round(r2_pure_gmp, 4)},
    ])
    ablation_df.to_csv(os.path.join(REPORTS_DIR, "ablation_study.csv"), index=False)
    print("\nAblation Study Results:")
    print(ablation_df.to_string(index=False))

    # Export Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ablation_df["Ablation Configuration"], ablation_df["Test RMSE"], color=["#10b981", "#ef4444", "#3b82f6"])
    ax.set_ylabel("Test RMSE (%)")
    ax.set_title("Ablation Study: Real Indian IPO Market Data (Test RMSE)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "19_ablation_study.png"), dpi=200)
    plt.close()

    # Model Leaderboard Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    reg_df = metrics_df[metrics_df["Type"] == "Regressor"]
    ax.bar(reg_df["Model"], reg_df["Test_RMSE"], color="#6366f1")
    ax.set_ylabel("Test RMSE (%)")
    ax.set_title("Regression Model Leaderboard on Real Test IPOs")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "18_model_comparison_leaderboard.png"), dpi=200)
    plt.close()

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    importances = pd.Series(best_gbr.feature_importances_, index=all_feature_cols).sort_values(ascending=True)
    importances.tail(8).plot(kind="barh", color="#10b981", ax=ax)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importances on Real Market IPOs (GBR)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "17_boosting_feature_importances.png"), dpi=200)
    plt.close()

    print("[SUCCESS] All models retrained, artifacts saved, and reports generated.")


if __name__ == "__main__":
    run_pipeline()
