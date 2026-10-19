from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    dataset_path = base_dir / "datasets" / "pool_water_quality_augmented_dataset.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    feature_columns = ["pH", "Temperature", "Chlorine", "Turbidity", "TDS"]
    target_column = "Status"

    missing = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[feature_columns]
    y = df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Model: XGBoost (XGBClassifier)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, models_dir / "water_quality_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

    metrics = {
        "model": "XGBClassifier",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classes": label_encoder.classes_.tolist(),
        "feature_columns": feature_columns,
    }
    (models_dir / "training_metrics_xgboost.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nSaved model to: {models_dir / 'water_quality_model.pkl'}")
    print(f"Saved scaler to: {models_dir / 'scaler.pkl'}")
    print(f"Saved label encoder to: {models_dir / 'label_encoder.pkl'}")
    print(f"Saved metrics to: {models_dir / 'training_metrics_xgboost.json'}")


if __name__ == "__main__":
    main()
