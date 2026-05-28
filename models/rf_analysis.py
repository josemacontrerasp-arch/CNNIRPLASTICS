import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from the project root
sys.path.append(str(Path(__file__).parent.parent))
from data.format_data import PlasticIRDataset


def main():
    importances_path = Path("output") / "rf_feature_importances.npy"
    if not importances_path.exists():
        print(f"Error: {importances_path} not found. Please run models/rf_model.py first.")
        return

    print("Initializing and loading PlasticIRDataset for wavenumbers...")
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
    _, target_wavenumbers = ds.get_formatted_data()

    print("Loading feature importances...")
    importances = np.load(importances_path)
    wavenumbers = target_wavenumbers

    if len(importances) != len(wavenumbers):
        print(f"Warning: size mismatch! Importances count: {len(importances)}, Wavenumbers count: {len(wavenumbers)}")
        # Truncate to match smallest size just in case
        min_len = min(len(importances), len(wavenumbers))
        importances = importances[:min_len]
        wavenumbers = wavenumbers[:min_len]

    # Combine into a list of tuples and sort by importance descending
    features = list(zip(wavenumbers, importances))
    features_sorted = sorted(features, key=lambda x: x[1], reverse=True)

    print("\nTop 20 Most Important Wavenumbers:")
    print("==================================")
    print(f"{'Rank':<6}{'Wavenumber (cm-1)':<20}{'Gini Importance':<15}")
    for rank, (wn, imp) in enumerate(features_sorted[:20], 1):
        print(f"{rank:<6}{wn:<20.2f}{imp:<15.6f}")

    # Save textual summary
    summary_path = Path("output") / "rf_importance_analysis.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Random Forest Feature Importance Analysis (Full Dataset)\n")
        f.write("========================================================\n\n")
        f.write("Top 50 Wavenumbers by Gini Importance:\n\n")
        f.write(f"{'Rank':<6}{'Wavenumber (cm-1)':<20}{'Gini Importance':<15}\n")
        for rank, (wn, imp) in enumerate(features_sorted[:50], 1):
            f.write(f"{rank:<6}{wn:<20.2f}{imp:<15.6f}\n")
    print(f"\nSaved importance list to {summary_path}")

    # Plot importances vs wavenumber
    print("Generating feature importance plot...")
    plt.figure(figsize=(12, 6))
    
    # Standard FTIR spectra plots have inverted X-axis (4000 down to 400 cm-1)
    plt.plot(wavenumbers, importances, color="#2ca02c", linewidth=1.5, label="Gini Importance")
    plt.gca().invert_xaxis()
    
    # Label the top 5 wavenumbers directly on the plot
    for rank, (wn, imp) in enumerate(features_sorted[:5], 1):
        plt.axvline(x=wn, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        plt.text(wn, imp + (max(importances) * 0.02), f"{wn:.1f}", 
                 rotation=90, verticalalignment='bottom', horizontalalignment='center',
                 fontsize=8, color="red", alpha=0.8)

    plt.title("Random Forest Feature Importance across FTIR Spectrum (Full Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Wavenumber (cm-1)", fontsize=12)
    plt.ylabel("Gini Importance Score", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(4000, 400)  # Standard range for commodity plastic spectra
    plt.ylim(0, max(importances) * 1.2)
    plt.legend()
    plt.tight_layout()

    plot_path = Path("output") / "rf_feature_importances.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")


if __name__ == "__main__":
    main()
