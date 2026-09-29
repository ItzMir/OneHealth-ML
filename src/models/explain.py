import os, json, numpy as np, joblib
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "../models"
PROCESSED_DIR = "../data/processed"
REPORT_DIR = "../reports/figures"
os.makedirs(REPORT_DIR, exist_ok=True)

DISEASE_MAP = {
    "diabetes": "label_diabetes",
    "hypertension": "label_hypertension",
    "ckd": "label_ckd",
    "dyslipidemia": "label_dyslipidemia",
    "cvd": "label_cvd",
    "metabolic_syndrome": "label_metabolic_syndrome",
    "liver_disease": "label_liver_disease",
}

def load_feature_names(disease):
    with open(os.path.join(PROCESSED_DIR, "feature_names.json")) as f:
        mapping = json.load(f)
    return mapping[DISEASE_MAP[disease]]

def explain_model(disease, model_type="xgboost", max_display=15, sample_size=200):
    model_path = os.path.join(MODEL_DIR, f"{disease}_{model_type}.pkl")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return
    model = joblib.load(model_path)
    X_test = np.load(os.path.join(PROCESSED_DIR, f"{disease}_X_test.npy"))
    feature_names = load_feature_names(disease)
    # Ensure feature_names length matches
    feature_names = feature_names[:X_test.shape[1]]
    
    # Subsample
    if X_test.shape[0] > sample_size:
        idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test

    # SHAP explainer
    if model_type in ["xgboost", "lightgbm", "random_forest"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"{disease}_{model_type}_summary_dot.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                          plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"{disease}_{model_type}_summary_bar.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # For logistic regression, use KernelExplainer (slower)
        explainer = shap.KernelExplainer(model.predict_proba, X_explain[:50])
        shap_values = explainer.shap_values(X_explain[:50])
        shap.summary_plot(shap_values, X_explain[:50], feature_names=feature_names,
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"{disease}_{model_type}_summary_dot.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"SHAP plots saved for {disease} ({model_type}).")

if __name__ == "__main__":
    # Generate SHAP for the selected best models
    best = {
        "diabetes": "lightgbm",
        "hypertension": "xgboost",
        "ckd": "xgboost",
        "dyslipidemia": "logistic_regression",
        "cvd": "logistic_regression",
        "metabolic_syndrome": "xgboost",
        "liver_disease": "xgboost"
    }
    for disease, model_type in best.items():
        explain_model(disease, model_type)