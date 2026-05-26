import sys
import pickle
from pathlib import Path
import numpy as np
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Ensure we can import from models
sys.path.append(str(Path(__file__).parent.parent))
from models.cnn_model_draft import load_and_preprocess


def main():
    print("Loading and preprocessing data...")
    # load_and_preprocess returns absorbance_values: (6000, 1868, 1) and labels: (6000,)
    absorbance_values, labels = load_and_preprocess()
    
    # Flatten the channel dimension for tabular Random Forest: shape (6000, 1868)
    X = np.squeeze(absorbance_values, axis=-1)
    y = labels
    
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
