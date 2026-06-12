from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# ROOTS
# -----------------------------
BASE_DIR = Path(__file__).parent.parent

DATASETS = {
    "SLoPP": BASE_DIR / "SLoPP",
    "SLoPP-E": BASE_DIR / "SLoPP-E"
}

# -----------------------------
# LABEL MAP (shared)
# -----------------------------
LABEL_MAP = {
    "Polyethylene Terephthalate": "PET",
    "Polypropylene": "PP",
    "Polystyrene": "PS",
    "Polyvinyl Chloride": "PVC",
}

# -----------------------------
# FIXED GRID (1983 points)
# -----------------------------
common_x = np.linspace(400, 4000, 1983)

def process_dataset(root_path: Path):
    rows = []

    for folder_name, label in LABEL_MAP.items():
        folder = root_path / folder_name

        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        for file in folder.glob("*.txt"):

            x_vals, y_vals = [], []

            try:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 2:
                            continue
                        try:
                            x, y = map(float, parts)
                            x_vals.append(x)
                            y_vals.append(y)
                        except ValueError:
                            continue

                x_vals = np.array(x_vals)
                y_vals = np.array(y_vals)

                if len(x_vals) < 10:
                    continue

                # sort by wavenumber
                idx = np.argsort(x_vals)
                x_vals = x_vals[idx]
                y_vals = y_vals[idx]

                # interpolate to fixed grid
                y_interp = np.interp(common_x, x_vals, y_vals)

                row = {
                    "polymer": label,
                    "sample": file.stem
                }

                # interleave x and y
                for i in range(len(common_x)):
                    row[f"x{i}"] = common_x[i]
                    row[f"y{i}"] = y_interp[i]

                rows.append(row)

            except Exception as e:
                print(f"[ERROR] {file}: {e}")

    return pd.DataFrame(rows)

# -----------------------------
# PROCESS BOTH DATASETS
# -----------------------------
for name, path in DATASETS.items():

    print(f"\nProcessing {name} ...")

    df = process_dataset(path)

    output_file = f"{name.lower()}_dataset.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved {output_file}")
    print("Shape:", df.shape)
    print(df["polymer"].value_counts())