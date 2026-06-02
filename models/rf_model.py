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


def main():
    print("Initializing and loading PlasticIRDataset...")
    # Instantiate dataset with path prefixes relative to root directory
    ds = PlasticIRDataset(
        ftir_c4_path="data/FTIR_PLASTIC_c4.csv",
        ftir_c8_path="data/FTIR_PLASTIC_c8.csv",
        openspecy_dataset_path="data/openspecy_polymer_dataset.csv",
        openspecy_metadata_path="data/openspecy_polymer_metadata.csv",
        openspecy_wavenumbers_path="data/openspecy_wavenumbers.csv"
    )
    
    # Process and align all datasets onto a single wavenumber grid
    ds.process()
    formatted_data, target_wavenumbers = ds.get_formatted_data()
    
    print("Filtering and formatting data for Random Forest...")
    # Keep only samples with valid labels (class index != -1)
    valid_data = [item for item in formatted_data if item["label_int"] != -1]
    
    X_raw = np.array([item["intensities"] for item in valid_data])
    y = np.array([item["label_int"] for item in valid_data])
    
    print(f"Total valid spectra: {len(X_raw)}")
    print(f"Wavenumber features: {X_raw.shape[1]}")

    # Per-spectrum min-max normalization, handling any NaNs introduced by interpolation
    print("Normalizing spectral intensities...")
    X = []
    for row in X_raw:
        if np.all(np.isnan(row)):
            X.append(np.zeros_like(row))
            continue
            
        min_val = np.nanmin(row)
        max_val = np.nanmax(row)
        
        if max_val - min_val == 0:
            scaled = np.zeros_like(row)
        else:
            scaled = (row - min_val) / (max_val - min_val)
            
        # Fill any NaNs with 0.0
        scaled = np.nan_to_num(scaled, nan=0.0)
        X.append(scaled)
        
    X = np.array(X)
    print("Normalization complete.")

    # Set up output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 5-fold Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    accuracies = []
    importances = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Training Fold {fold} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Instantiate Random Forest (n_estimators=100, use all CPU cores)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        
        # Train model
        rf.fit(X_train, y_train)
        
        # Validate model
        y_pred = rf.predict(X_test)
        acc = skm.accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} Validation Accuracy: {acc:.4f}")
        
        # Store feature importances for this fold
        importances.append(rf.feature_importances_)
        
        # Save fold model to pickle file
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
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Random Forest 5-Fold Stratified Cross-Validation Summary (Full Dataset)\n")
        f.write("========================================================================\n")
        for fold, acc in enumerate(accuracies, 1):
            f.write(f"Fold {fold} Accuracy: {acc:.4f}\n")
        f.write(f"\nMean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
