import sys
import pickle
from pathlib import Path
import numpy as np
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Ensure we can import from the project root
sys.path.append(str(Path(__file__).parent.parent))
from data.format_data import PlasticIRDataset
from preprocess import preprocess, PreprocessConfig


def main():
    print("Loading and preprocessing data...")
    # Extract + align all spectra onto the shared wavenumber grid (format_data.py),
    # then apply the same per-spectrum, leakage-free preprocessing used in the
    # CNN pipeline.  X is already 2-D (n_spectra, n_points) -- no channel axis to
    # strip, since Random Forest works directly on the flat spectra.
    data_dir = Path(__file__).parent.parent / "data"
    ds = PlasticIRDataset(
        ftir_c4_path=str(data_dir / "FTIR_PLASTIC_c4.csv"),
        ftir_c8_path=str(data_dir / "FTIR_PLASTIC_c8.csv"),
        openspecy_dataset_path=str(data_dir / "openspecy_polymer_dataset.csv"),
        openspecy_metadata_path=str(data_dir / "openspecy_polymer_metadata.csv"),
        openspecy_wavenumbers_path=str(data_dir / "openspecy_wavenumbers.csv"),
    )
    ds.process()
    formatted, wn = ds.get_formatted_data()
    X = np.array([e["intensities"] for e in formatted], dtype=float)
    y = np.array([e["label_int"] for e in formatted], dtype=int)
    X = preprocess(X, PreprocessConfig(normalize="minmax"), wavenumbers=wn)[0]
    
    print(f"Data shape for Random Forest: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Set up directories
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 5-fold Stratified K-Fold to match the CNN evaluation scheme
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    accuracies = []
    importances = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Training Fold {fold} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Instantiate Random Forest Classifier
        # n_estimators=100 is standard and fast; n_jobs=-1 runs on all CPU cores
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        
        # Fit the model
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        acc = skm.accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        
        # Store feature importances for this fold
        importances.append(rf.feature_importances_)
        
        # Save model using pickle
        model_path = output_dir / f"rf_fold_{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        print(f"Saved model to {model_path}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\n--- Cross-Validation Summary ---")
    print(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # Compute mean feature importances across all folds
    mean_importances = np.mean(importances, axis=0)
    importance_path = output_dir / "rf_feature_importances.npy"
    np.save(importance_path, mean_importances)
    print(f"Saved mean feature importances to {importance_path}")

    # Save summary metrics to text file
    summary_path = output_dir / "rf_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Random Forest 5-Fold Stratified Cross-Validation Summary\n")
        f.write("=====================================================\n")
        for fold, acc in enumerate(accuracies, 1):
            f.write(f"Fold {fold} Accuracy: {acc:.4f}\n")
        f.write(f"\nMean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
