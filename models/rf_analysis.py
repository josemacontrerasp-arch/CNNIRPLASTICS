import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_wavenumbers():
    dbs2_path = Path("data") / "FTIR_PLASTIC_c8.csv"
    with open(dbs2_path, newline="", encoding="utf-8") as dbs:
        dbs_reader = csv.reader(dbs)
        dbs_reader.__next__()  # skip header
        first_row = dbs_reader.__next__()
        
        # Wavenumbers start at index 6 and alternate:
        # 6, 8, 10, etc. are wavenumbers (x); 7, 9, 11, etc. are intensities (y)
        data_slice = first_row[6:len(first_row) - 2]
        wavenumbers = [float(data_slice[j]) for j in range(0, len(data_slice), 2)]
        return np.array(wavenumbers)


def main():
    importances_path = Path("output") / "rf_feature_importances.npy"
    if not importances_path.exists():
        print(f"Error: {importances_path} not found. Please run models/rf_model.py first.")
        return

    print("Loading feature importances and wavenumbers...")
    importances = np.load(importances_path)
    wavenumbers = load_wavenumbers()

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
        f.write("Random Forest Feature Importance Analysis\n")
        f.write("=========================================\n\n")
        f.write("Top 50 Wavenumbers by Gini Importance:\n\n")
        f.write(f"{'Rank':<6}{'Wavenumber (cm-1)':<20}{'Gini Importance':<15}\n")
        for rank, (wn, imp) in enumerate(features_sorted[:50], 1):
            f.write(f"{rank:<6}{wn:<20.2f}{imp:<15.6f}\n")
    print(f"\nSaved importance list to {summary_path}")

    # Plot importances vs wavenumber
    print("Generating feature importance plot...")
    plt.figure(figsize=(12, 6))
    
    # Standard FTIR spectra plots have inverted X-axis (4000 down to 400 cm-1)
    plt.plot(wavenumbers, importances, color="#1f77b4", linewidth=1.5, label="Gini Importance")
    plt.gca().invert_xaxis()
    
    # Label the top 5 wavenumbers directly on the plot
    for rank, (wn, imp) in enumerate(features_sorted[:5], 1):
        plt.axvline(x=wn, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        plt.text(wn, imp + (max(importances) * 0.02), f"{wn:.1f}", 
                 rotation=90, verticalalignment='bottom', horizontalalignment='center',
                 fontsize=8, color="red", alpha=0.8)

    plt.title("Random Forest Feature Importance across FTIR Spectrum", fontsize=14, fontweight="bold", pad=15)
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
