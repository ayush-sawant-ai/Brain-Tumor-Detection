import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"E:\SEM5\ERA\ERA_Project\Brain_Tumor"
)

TRAIN_DIR = PROJECT_DIR / "data" / "Training"
TEST_DIR = PROJECT_DIR / "data" / "Testing"

OUTPUT_DIR = PROJECT_DIR / "diagnostic_outputs" / "metadata_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# 2. EXTRACT IMAGE METADATA
# ============================================================

def extract_image_features(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        width, height = image.size

        img = np.asarray(image, dtype=np.float32)

        # RGB statistics
        rgb_mean = img.mean(axis=(0, 1))
        rgb_std = img.std(axis=(0, 1))

        # Grayscale brightness statistics
        gray = np.asarray(image.convert("L"), dtype=np.float32)

        brightness_mean = gray.mean()
        brightness_std = gray.std()

        # Border brightness
        border_size = max(1, int(min(width, height) * 0.05))

        border = np.concatenate([
            gray[:border_size, :].ravel(),
            gray[-border_size:, :].ravel(),
            gray[:, :border_size].ravel(),
            gray[:, -border_size:].ravel()
        ])

        border_brightness = border.mean()

        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height,

            "red_mean": rgb_mean[0],
            "green_mean": rgb_mean[1],
            "blue_mean": rgb_mean[2],

            "red_std": rgb_std[0],
            "green_std": rgb_std[1],
            "blue_std": rgb_std[2],

            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "border_brightness": border_brightness
        }


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_metadata_dataset(directory):

    records = []

    class_names = sorted([
        folder.name
        for folder in directory.iterdir()
        if folder.is_dir()
    ])

    for class_name in class_names:

        class_dir = directory / class_name

        image_paths = [
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"Processing {class_name}: {len(image_paths)} images")

        for image_path in image_paths:

            try:
                features = extract_image_features(image_path)

                features["label"] = class_name
                features["image_path"] = str(image_path)

                records.append(features)

            except Exception as e:
                print(f"Skipping {image_path}: {e}")

    return pd.DataFrame(records)


print("Extracting training metadata...")
train_df = load_metadata_dataset(TRAIN_DIR)

print("\nExtracting testing metadata...")
test_df = load_metadata_dataset(TEST_DIR)


# ============================================================
# 4. VERIFY DATASET
# ============================================================

print("\nTraining shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_df["label"].value_counts())

train_classes = set(train_df["label"].unique())
test_classes = set(test_df["label"].unique())

assert train_classes == test_classes, (
    f"Class mismatch! Train: {train_classes}, Test: {test_classes}"
)

assert len(train_df) > 0 and len(test_df) > 0, "Dataset is empty!"

# Save extracted metadata
train_df.to_csv(
    OUTPUT_DIR / "training_metadata.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "testing_metadata.csv",
    index=False
)


# ============================================================
# 5. PREPARE FEATURES AND LABELS
# ============================================================

FEATURE_COLUMNS = [
    "width",
    "height",
    "aspect_ratio",

    "red_mean",
    "green_mean",
    "blue_mean",

    "red_std",
    "green_std",
    "blue_std",

    "brightness_mean",
    "brightness_std",
    "border_brightness"
]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["label"]

print("\nFeatures used:", FEATURE_COLUMNS)


# ============================================================
# 6. DEFINE METADATA-ONLY MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}


# ============================================================
# 7. TRAIN-ONLY CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy"
        },
        n_jobs=-1,
        return_train_score=True
    )

    print("\nTRAINING CROSS-VALIDATION RESULTS")

    print(
        "CV Accuracy:",
        f"{cv_results['test_accuracy'].mean():.4f}",
        "+/-",
        f"{cv_results['test_accuracy'].std():.4f}"
    )

    print(
        "CV Balanced Accuracy:",
        f"{cv_results['test_balanced_accuracy'].mean():.4f}",
        "+/-",
        f"{cv_results['test_balanced_accuracy'].std():.4f}"
    )

    print(
        "Mean Training Accuracy:",
        f"{cv_results['train_accuracy'].mean():.4f}"
    )

    # Fit using the complete training set
    model.fit(X_train, y_train)

    # Evaluate on untouched test set
    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)

    test_balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    print("\nUNTOUCHED TEST RESULTS")

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(train_classes),
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "CV Accuracy": cv_results["test_accuracy"].mean(),
        "CV Accuracy Std": cv_results["test_accuracy"].std(),
        "CV Balanced Accuracy": cv_results["test_balanced_accuracy"].mean(),
        "Test Accuracy": test_accuracy,
        "Test Balanced Accuracy": test_balanced_accuracy
    })

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=sorted(train_classes)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(train_classes)
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Metadata-only: {model_name}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 8. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "metadata_model_comparison.csv",
    index=False
)


# ============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = models["Random Forest"]

rf_classifier = rf_model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": rf_classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "metadata_feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Metadata Feature")
plt.title("Random Forest - Metadata Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "metadata_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("Results saved at:")
print(OUTPUT_DIR)

print("\nImportant:")
print("The test set was not used during cross-validation or model fitting.")
print("Only image metadata was used; no tumor-image features were used.")