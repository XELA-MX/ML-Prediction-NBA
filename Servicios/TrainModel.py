"""
NBA Predictor — Entrenamiento del Modelo ML
Entrena sobre el dataset histórico y guarda el mejor modelo.

Uso:
    python Servicios/TrainModel.py

Requiere haber ejecutado primero BuildDataset.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Differential features selected via backward elimination + new features
DIFF_COLS = ["win_rate_10j", "pt_diff_10j", "pts_allowed_10j", "season_def_rtg", "schedule_strength"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Builds the final feature matrix from the raw dataset.
    Features expanded to include:
      - Schedule strength differential
      - Travel factor for away team
      - Home rest days
    Returns (feature_df, feature_col_names).
    """
    feat = pd.DataFrame(index=df.index)

    # Differential features: home - away
    for col in DIFF_COLS:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in df.columns and away_col in df.columns:
            feat[f"diff_{col}"] = df[home_col] - df[away_col]

    # Context features
    feat["away_rest_days"]  = df["away_rest_days"]
    feat["away_fatigue_7d"] = df["away_fatigue_7d"]
    feat["home_b2b"]        = df["home_b2b"]
    
    # New: home rest days
    if "home_rest_days" in df.columns:
        feat["home_rest_days"] = df["home_rest_days"]
    
    # New: travel factor (affects away team)
    if "travel_factor" in df.columns:
        feat["travel_factor"] = df["travel_factor"]

    # H2H
    feat["h2h_ratio"] = df["h2h_ratio"]

    feature_cols = list(feat.columns)
    return feat, feature_cols


def train() -> dict:
    dataset_path = os.path.join(DATA_DIR, "dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset no encontrado en {dataset_path}\n"
            "Ejecuta primero:  python Servicios/BuildDataset.py"
        )

    df = pd.read_csv(dataset_path)
    print(f"Dataset: {len(df)} partidos  |  temporadas: {sorted(df['season'].unique())}")
    print(f"Win rate local en el dataset: {df['home_win'].mean():.3f}\n")

    feat_df, feature_cols = build_features(df)
    X = feat_df.values
    y = df["home_win"].values

    print(f"Features ({len(feature_cols)}): {feature_cols}\n")

    # Split temporal: train all seasons before 2024-25, test on 2024-25
    mask_train = df["season"] != "2024-25"
    mask_test  = df["season"] == "2024-25"
    X_train, y_train = X[mask_train], y[mask_train]
    X_test,  y_test  = X[mask_test],  y[mask_test]
    print(f"Train: {len(X_train)} partidos  |  Test (2024-25): {len(X_test)} partidos\n")

    # ── Modelo 1: HistGradientBoosting ───────────────────────
    print("─ HistGradientBoostingClassifier...")
    gbm = HistGradientBoostingClassifier(
        max_iter=600,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=40,
        l2_regularization=0.5,
        max_features=0.8,
        random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_probs = gbm.predict_proba(X_test)[:, 1]
    gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
    gbm_auc = roc_auc_score(y_test, gbm_probs)
    gbm_brier = brier_score_loss(y_test, gbm_probs)
    print(f"  Accuracy: {gbm_acc:.3f}  |  AUC-ROC: {gbm_auc:.3f}  |  Brier: {gbm_brier:.4f}")

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(gbm, X, y, cv=cv, scoring="roc_auc")
    print(f"  AUC CV 5-fold: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # ── Modelo 2: Logistic Regression (baseline) ─────────────────
    print("\n─ Logistic Regression (baseline)...")
    lr = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_brier = brier_score_loss(y_test, lr_probs)
    print(f"  Accuracy: {lr_acc:.3f}  |  AUC-ROC: {lr_auc:.3f}  |  Brier: {lr_brier:.4f}")

    # ── Elegir el mejor ─────────────────────────────────────────
    if gbm_auc >= lr_auc:
        best_model, best_name, best_auc, best_acc = gbm, "HistGradientBoosting", gbm_auc, gbm_acc
    else:
        best_model, best_name, best_auc, best_acc = lr,  "LogisticRegression",   lr_auc,  lr_acc

    # ── Calibración de probabilidades ─────────────────────────
    print(f"\n─ Calibrando modelo ({best_name})...")
    cal_model = CalibratedClassifierCV(
        best_model, cv=5, method="isotonic"
    )
    cal_model.fit(X_train, y_train)
    cal_probs = cal_model.predict_proba(X_test)[:, 1]
    cal_acc   = accuracy_score(y_test, cal_model.predict(X_test))
    cal_auc   = roc_auc_score(y_test, cal_probs)
    cal_brier = brier_score_loss(y_test, cal_probs)
    print(f"  Calibrado → Accuracy: {cal_acc:.3f}  |  AUC-ROC: {cal_auc:.3f}  |  Brier: {cal_brier:.4f}")

    # Usar calibrado si mejora Brier sin destruir AUC
    best_brier = gbm_brier if best_name == "HistGradientBoosting" else lr_brier
    if cal_brier < best_brier and cal_auc >= best_auc * 0.98:
        final_model = cal_model
        final_name  = f"{best_name}+Calibrated"
        final_auc, final_acc = cal_auc, cal_acc
        print(f"  Usando modelo calibrado (Brier mejorado)")
    else:
        final_model = best_model
        final_name  = best_name
        final_auc, final_acc = best_auc, best_acc
        print(f"  Calibración no mejoró — usando modelo sin calibrar")

    print(f"\nModelo final: {final_name}  (AUC={final_auc:.3f}  Acc={final_acc:.3f})")

    # ── Permutation importances ───────────────────────────────
    print("\n─ Permutation importances (test set):")
    perm = permutation_importance(
        final_model, X_test, y_test,
        n_repeats=10, random_state=42, scoring="roc_auc"
    )
    pairs = sorted(zip(feature_cols, perm.importances_mean),
                   key=lambda x: x[1], reverse=True)
    for name, imp in pairs:
        bar = "█" * int(imp * 200)
        print(f"  {name:<35} {imp:+.4f}  {bar}")

    # ── Guardar ─────────────────────────────────────────
    artifact = {
        "model":         final_model,
        "feature_cols":  feature_cols,
        "model_name":    final_name,
        "test_auc":      round(final_auc, 4),
        "test_accuracy": round(final_acc, 4),
    }
    model_path = os.path.join(DATA_DIR, "model.pkl")
    joblib.dump(artifact, model_path)
    print(f"\nModelo guardado en: {model_path}")

    return artifact


if __name__ == "__main__":
    print("=" * 56)
    print("  NBA PREDICTOR — Entrenamiento del modelo ML")
    print("=" * 56 + "\n")
    train()
