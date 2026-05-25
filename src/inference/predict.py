import sys
import csv
import numpy as np
from pathlib import Path


LABELS = ['HDPE', 'LDPE', 'PP', 'PS', 'PVC', 'PET']
EXPECTED_POINTS = 1868


def load_spectrum(filepath):
    """Load absorbance from a CSV file.

    Accepts:
    - Two columns (wavenumber, absorbance) — only absorbance is used
    - Single column of absorbance values
    Non-numeric rows (headers) are skipped automatically.
    """
    values = []
    with open(filepath, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                values.append(float(row[1]) if len(row) >= 2 else float(row[0]))
            except ValueError:
                continue  # skip header or malformed rows
    return np.array(values, dtype=np.float64)


def predict(spectrum_path):
    from tensorflow import keras

    output_dir = Path("output/models")
    models = [
        keras.models.load_model(output_dir / f"fold_{k}.keras")
        for k in range(1, 6)
        if (output_dir / f"fold_{k}.keras").exists()
    ]

    if not models:
        print("No saved models found in output/models/. Run train_keras.py first.")
        sys.exit(1)

    spectrum = load_spectrum(spectrum_path)

    if len(spectrum) != EXPECTED_POINTS:
        print(f"Error: expected {EXPECTED_POINTS} absorbance points, got {len(spectrum)}.")
        print("The input spectrum must match the wavenumber grid used during training.")
        sys.exit(1)

    # Apply the same per-spectrum min-max normalisation used during training
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

    # Shape: (1, 1868, 1) — batch of one, 1868 timesteps, 1 channel
    x = spectrum[np.newaxis, :, np.newaxis]

    # Collect one vote per fold model, then take the majority
    votes = np.zeros(len(LABELS), dtype=int)
    for model in models:
        votes[np.argmax(model.predict(x, verbose=0)[0])] += 1

    predicted = LABELS[np.argmax(votes)]
    print(f"Prediction: {predicted}")
    print(f"Votes ({len(models)} fold model(s)): { {LABELS[i]: votes[i] for i in range(len(LABELS))} }")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inference/predict.py <spectrum.csv>")
        sys.exit(1)
    predict(sys.argv[1])
