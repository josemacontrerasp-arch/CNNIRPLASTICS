import os
import csv
import argparse
import glob

# ─────────────────────────────────────────────────────────────────
# Label lookup
# Maps filename (without extension) to polymer class.
# ─────────────────────────────────────────────────────────────────

LABELS = {
    "bluecap1":       "HDPE",
    "bluecap2":       "HDPE",
    "bluecap3":       "HDPE",
    "cddisk1":        "PS",
    "cddisk2":        "PS",
    "cddisk3":        "PS",
    "cup1":           "PP",
    "cup2":           "PP",
    "cup3":           "PP",
    "foam1":          "PS",
    "foam2":          "PS",
    "foam3":          "PS",
    "fruitbag1":      "HDPE",
    "fruitbag2":      "HDPE",
    "fruitbag3":      "HDPE",
    "greencap":       "HDPE",
    "greencap2":      "HDPE",
    "greencap3":      "HDPE",
    "gum1":           "HDPE",
    "gum2":           "HDPE",
    "gum3":           "HDPE",
    "ibuprofen1":     "PVC",
    "ibuprofen2":     "PVC",
    "ibuprofen3":     "PVC",
    "milk1":          "HDPE",
    "milk2":          "HDPE",
    "milk3":          "HDPE",
    "petbluebottle1": "PET",
    "petbluebottle2": "PET",
    "petbluebottle3": "PET",
    "pvcgrey":        "PVC",
    "pvcgrey2":       "PVC",
    "pvcgrey3":       "PVC",
    "pvcwhite":       "PVC",
    "pvcwhite2":      "PVC",
    "pvcwhite3":      "PVC",
    "ziploc1":        "LDPE",
    "ziploc2":        "LDPE",
    "ziploc3":        "LDPE",
}

HEADER_LINES = 4  # number of ##... metadata lines before data begins


def parse_txt(path: str) -> list[tuple[float, float]]:
    """
    Reads one IR spectrum .txt file and returns a list of (wavenumber, intensity)
    pairs. Skips the first HEADER_LINES lines. Handles Windows CRLF endings.
    """
    points = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < HEADER_LINES:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Unexpected line format in {path!r}: {line!r}")
            points.append((float(parts[0]), float(parts[1])))
    return points


def convert_folder(folder: str, output_path: str) -> None:
    """
    Reads every .txt file in `folder`, converts each to a row in a wide CSV,
    and writes the result to `output_path`.

    Output columns:
        filename, label, Data(x), Data(y), Data(x), Data(y), ...
                                  ↑ repeated once per data point ↑
    """
    txt_files = sorted(glob.glob(os.path.join(folder, "*.txt")))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder!r}")

    # Parse all files first so we can validate consistency before writing
    parsed: list[tuple[str, list]] = []
    expected_n_points: int | None = None

    for path in txt_files:
        filename = os.path.basename(path)
        stem     = os.path.splitext(filename)[0]

        if stem not in LABELS:
            raise KeyError(
                f"No label defined for '{filename}'. "
                f"Add it to the LABELS dict before running."
            )

        points = parse_txt(path)

        if expected_n_points is None:
            expected_n_points = len(points)
        elif len(points) != expected_n_points:
            raise ValueError(
                f"'{filename}' has {len(points)} data points; "
                f"expected {expected_n_points} (from the first file)."
            )

        parsed.append((filename, stem, points))

    n_points = expected_n_points

    # Build header: filename, label, then Data(x)/Data(y) repeated n_points times
    header = ["filename", "label"]
    for _ in range(n_points):
        header += ["Data(x)", "Data(y)"]

    print(f"Found {len(parsed)} files  |  {n_points} data points each  |  "
          f"Output columns: {len(header)}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filename, stem, points in parsed:
            label = LABELS[stem]
            row   = [filename, label]
            for x, y in points:
                row += [x, y]
            writer.writerow(row)
            print(f"  {filename:<25s}  →  {label}")

    print(f"\nDone. Written to: {output_path}")
    print(f"Rows: {len(parsed) + 1} (1 header + {len(parsed)} spectra)")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of IR spectrum .txt files to a single wide CSV."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing .txt files.",
    )
    parser.add_argument(
        "--output",
        default="ftir_real_world.csv",
        help="Output CSV filename/path (default: ftir_real_world.csv).",
    )
    args = parser.parse_args()

    convert_folder(
        folder=os.path.abspath(args.folder),
        output_path=args.output,
    )