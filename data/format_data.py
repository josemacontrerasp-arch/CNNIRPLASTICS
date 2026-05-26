import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# File paths
# Adjust these to match your directory layout.
# ─────────────────────────────────────────────────────────────────

FTIR_C4_PATH              = "FTIR_PLASTIC_c4.csv"
FTIR_C8_PATH              = "FTIR_PLASTIC_c8.csv"
OPENSPECY_DATASET_PATH    = "openspecy_polymer_dataset.csv"
OPENSPECY_METADATA_PATH   = "openspecy_polymer_metadata.csv"
OPENSPECY_WAVENUMBERS_PATH = "openspecy_wavenumbers.csv"

# SLoPP/SLoPP-E is excluded: it is a Raman spectral library, not FTIR.
# SLOPP_PATH = "slopp_and_slopp_e/slopp_and_slopp_e.zip"

# ─────────────────────────────────────────────────────────────────
# Label definitions
# ─────────────────────────────────────────────────────────────────

POLYMER_CLASSES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
LABEL_TO_INT    = {label: i for i, label in enumerate(POLYMER_CLASSES)}


# ─────────────────────────────────────────────────────────────────

class PlasticIRDataset:
    """
    Aggregates IR spectra from FTIR c4, FTIR c8, and OpenSpecy (FTIR-only)
    onto a single shared wavenumber grid, ready for CNN training.

    Typical usage:
        ds = PlasticIRDataset()
        ds.process()
        formatted, wavenumbers = ds.get_formatted_data()
    """

    def __init__(
        self,
        ftir_c4_path:               str = FTIR_C4_PATH,
        ftir_c8_path:               str = FTIR_C8_PATH,
        openspecy_dataset_path:     str = OPENSPECY_DATASET_PATH,
        openspecy_metadata_path:    str = OPENSPECY_METADATA_PATH,
        openspecy_wavenumbers_path: str = OPENSPECY_WAVENUMBERS_PATH,
    ):
        # ── Paths ────────────────────────────────────────────────
        self.ftir_c4_path               = ftir_c4_path
        self.ftir_c8_path               = ftir_c8_path
        self.openspecy_dataset_path     = openspecy_dataset_path
        self.openspecy_metadata_path    = openspecy_metadata_path
        self.openspecy_wavenumbers_path = openspecy_wavenumbers_path

        # ── Raw data ─────────────────────────────────────────────
        # Populated by load_raw(). Keys: 'ftir_c4', 'ftir_c8', 'openspecy'.
        # Each value is a dict; structure differs by source (see _load_ftir /
        # _load_openspecy for exact keys). Intentionally unexposed until
        # get_raw_data() is called, so callers know loading has occurred.
        self._raw_data: dict   = {}
        self._raw_loaded: bool = False

        # ── Analysis results ──────────────────────────────────────
        # Filled by process() before alignment decisions are made.
        # Keyed by dataset name; each value is a stats dict (range, step, count…).
        self._dataset_stats: dict = {}

        # ── Target grid ───────────────────────────────────────────
        # 1-D array of wavenumber values (cm⁻¹) that every formatted spectrum
        # is aligned to. Decided during process() as the coarsest grid within
        # the shared range of all datasets.
        self.target_wavenumbers: Optional[np.ndarray] = None

        # ── Formatted data ────────────────────────────────────────
        # Empty until process() completes. Each entry is a dict:
        #   {
        #     'label':       str          — polymer class, e.g. 'PET'
        #     'label_int':   int          — integer index per LABEL_TO_INT
        #     'intensities': np.ndarray   — aligned to target_wavenumbers
        #     'source':      str          — 'ftir_c4' | 'ftir_c8' | 'openspecy'
        #   }
        self.formatted_data: list = []

    # ─────────────────────────────────────────────────────────────
    # Private — loading
    # ─────────────────────────────────────────────────────────────

    def _load_ftir(self, path: str, key: str) -> None:
        """
        Parses one of the FTIR CSVs (c4 or c8).

        Layout: 6 metadata columns (IDE, Polymer, Technic, Sample, BR, RST)
        followed by interleaved x/y pairs — Data(x), Data(y), Data(x), Data(y), …

        Wavenumbers are stored as a 2-D array (n_spectra × n_points) because
        the spectrometer wavenumber positions vary very slightly between
        measurements. In practice the variance is tiny, but we preserve it here
        so _align_spectrum can use each spectrum's own x-axis exactly.
        """
        df = pd.read_csv(path, header=0)

        data_cols = df.columns[6:]          # everything after the 6 metadata cols
        x_cols    = data_cols[0::2]         # columns 0, 2, 4, … → wavenumbers
        y_cols    = data_cols[1::2]         # columns 1, 3, 5, … → intensities

        wavenumbers = df[x_cols].to_numpy(dtype=float)   # (n_spectra, n_points)
        intensities = df[y_cols].to_numpy(dtype=float)   # (n_spectra, n_points)
        labels      = df["Polymer"].to_numpy()

        self._raw_data[key] = {
            "wavenumbers": wavenumbers,  # 2-D; use row i for spectrum i
            "intensities": intensities,
            "labels":      labels,
            "n_spectra":   int(len(labels)),
            "n_points":    int(wavenumbers.shape[1]),
        }

    def _load_openspecy(self) -> None:
        """
        Loads OpenSpecy and drops all Raman spectra.

        The three files relate as follows:
          - metadata row i  ↔  dataset row i  (aligned by construction in the R script)
          - dataset column j (V1…V1983) ↔  wavenumber file row j

        After filtering, 'wavenumbers' is a shared 1-D axis (all spectra identical).
        """
        metadata   = pd.read_csv(self.openspecy_metadata_path)
        dataset    = pd.read_csv(self.openspecy_dataset_path)
        wavenumbers = (
            pd.read_csv(self.openspecy_wavenumbers_path)["wavenumber"]
            .to_numpy(dtype=float)
        )

        # Drop Raman — keep only FTIR
        ftir_mask       = metadata["spectrum_type"].str.lower().str.strip() == "ftir"
        n_raman_dropped = int((~ftir_mask).sum())

        metadata_ftir = metadata[ftir_mask].reset_index(drop=True)
        dataset_ftir  = dataset[ftir_mask].reset_index(drop=True)

        intensity_cols = [c for c in dataset_ftir.columns if c != "label"]
        intensities    = dataset_ftir[intensity_cols].to_numpy(dtype=float)
        labels         = dataset_ftir["label"].to_numpy()

        self._raw_data["openspecy"] = {
            "wavenumbers":     wavenumbers,          # 1-D shared axis
            "intensities":     intensities,
            "labels":          labels,
            "n_spectra":       int(len(labels)),
            "n_points":        int(len(wavenumbers)),
            "n_raman_dropped": n_raman_dropped,
        }

    # ─────────────────────────────────────────────────────────────
    # Private — analysis & alignment
    # ─────────────────────────────────────────────────────────────

    def _analyze_dataset(self, key: str) -> dict:
        """
        Computes per-dataset wavenumber statistics used to decide the target grid.
        For FTIR (2-D wavenumber array) the mean axis across spectra is used;
        the per-position standard deviation is reported as a sanity check.
        """
        raw = self._raw_data[key]
        wn  = raw["wavenumbers"]

        if wn.ndim == 2:
            mean_wn  = wn.mean(axis=0)
            # Average std across all wavenumber positions — should be near zero
            wn_jitter = float(wn.std(axis=0).mean())
        else:
            mean_wn   = wn
            wn_jitter = 0.0

        steps = np.diff(mean_wn)
        return {
            "n_spectra":  raw["n_spectra"],
            "n_points":   raw["n_points"],
            "wn_min":     float(mean_wn.min()),
            "wn_max":     float(mean_wn.max()),
            "mean_step":  float(steps.mean()),
            "min_step":   float(steps.min()),
            "max_step":   float(steps.max()),
            "mean_wn":    mean_wn,      # representative 1-D axis for this dataset
            "wn_jitter":  wn_jitter,    # FTIR only: how much wn varies between spectra
        }

    def _decide_target_wavenumbers(self) -> np.ndarray:
        """
        Picks the target wavenumber grid:
          - Range:      intersection of all dataset ranges (no extrapolation)
          - Resolution: coarsest dataset's actual grid points within that range

        Using the coarsest grid means that dataset needs zero interpolation;
        finer-resolution datasets are only downsampled (evaluated at fewer
        points within their existing range), which introduces no artificial data.
        """
        stats = self._dataset_stats

        shared_min = max(s["wn_min"]  for s in stats.values())
        shared_max = min(s["wn_max"]  for s in stats.values())

        coarsest_key = max(stats, key=lambda k: stats[k]["mean_step"])
        coarsest_wn  = stats[coarsest_key]["mean_wn"]

        mask   = (coarsest_wn >= shared_min) & (coarsest_wn <= shared_max)
        target = coarsest_wn[mask]

        print(f"\nTarget wavenumber grid decision:")
        print(f"  Shared range:      {shared_min:.1f} – {shared_max:.1f} cm⁻¹")
        print(f"  Reference dataset: '{coarsest_key}'"
              f"  (step ≈ {stats[coarsest_key]['mean_step']:.2f} cm⁻¹)")
        print(f"  Grid points:       {len(target)}")

        return target

    def _align_spectrum(
        self,
        wn_src:     np.ndarray,
        intensities: np.ndarray,
        target_wn:  np.ndarray,
    ) -> np.ndarray:
        """
        Resamples a single spectrum from its source grid onto target_wn using
        linear interpolation. Points outside the source range become NaN rather
        than extrapolated — handle NaNs downstream as needed.
        """
        f = interp1d(
            wn_src, intensities,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return f(target_wn)

    # ─────────────────────────────────────────────────────────────
    # Public — loading
    # ─────────────────────────────────────────────────────────────

    def load_raw(self) -> None:
        """
        Reads all source files into _raw_data without any alignment or processing.
        Idempotent — calling it a second time is a no-op.
        """
        if self._raw_loaded:
            return
        print("Loading FTIR c4 …")
        self._load_ftir(self.ftir_c4_path, "ftir_c4")
        print("Loading FTIR c8 …")
        self._load_ftir(self.ftir_c8_path, "ftir_c8")
        print("Loading OpenSpecy …")
        self._load_openspecy()
        self._raw_loaded = True
        dropped = self._raw_data["openspecy"]["n_raman_dropped"]
        print(f"Raw loading complete.  ({dropped} Raman spectra discarded from OpenSpecy)")

    # ─────────────────────────────────────────────────────────────
    # Public — processing
    # ─────────────────────────────────────────────────────────────

    def process(self) -> None:
        """
        Full pipeline:
          1. Load raw data (if not already done)
          2. Analyse each dataset — range, resolution, count
          3. Decide target wavenumber grid
          4. Align every spectrum to the target grid
          5. Populate self.formatted_data and self.target_wavenumbers
        """
        self.load_raw()

        # Step 2
        print("\nAnalysing datasets …")
        for key in self._raw_data:
            self._dataset_stats[key] = self._analyze_dataset(key)
            s = self._dataset_stats[key]
            print(
                f"  {key:<12s}  {s['n_spectra']:5d} spectra  "
                f"{s['n_points']:5d} pts  "
                f"range {s['wn_min']:.0f}–{s['wn_max']:.0f} cm⁻¹  "
                f"step ≈ {s['mean_step']:.2f} cm⁻¹"
                + (f"  wn-jitter ≈ {s['wn_jitter']:.4f}" if s["wn_jitter"] > 0 else "")
            )

        # Step 3
        self.target_wavenumbers = self._decide_target_wavenumbers()

        # Steps 4 + 5
        print("\nAligning spectra …")
        self.formatted_data = []

        for key in self._raw_data:
            raw       = self._raw_data[key]
            per_row_wn = raw["wavenumbers"].ndim == 2  # True for FTIR, False for OpenSpecy

            for i in range(raw["n_spectra"]):
                wn_src = raw["wavenumbers"][i] if per_row_wn else raw["wavenumbers"]
                aligned = self._align_spectrum(
                    wn_src, raw["intensities"][i], self.target_wavenumbers
                )
                label = str(raw["labels"][i])
                self.formatted_data.append({
                    "label":       label,
                    "label_int":   LABEL_TO_INT.get(label, -1),
                    "intensities": aligned,
                    "source":      key,
                })

        # Summary
        print(f"\nProcessing complete — {len(self.formatted_data)} spectra total")
        label_counts: dict = {}
        for entry in self.formatted_data:
            label_counts[entry["label"]] = label_counts.get(entry["label"], 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

    # ─────────────────────────────────────────────────────────────
    # Public — accessors
    # ─────────────────────────────────────────────────────────────

    def get_raw_data(self, key: Optional[str] = None) -> dict:
        """
        Returns raw data. Pass a key ('ftir_c4', 'ftir_c8', 'openspecy') to get
        one dataset's dict, or omit to get the full _raw_data dict.
        Triggers load_raw() automatically if not yet done.
        """
        self.load_raw()
        if key is not None:
            if key not in self._raw_data:
                raise KeyError(
                    f"Unknown key '{key}'. Valid keys: {list(self._raw_data.keys())}"
                )
            return self._raw_data[key]
        return self._raw_data

    def get_formatted_data(self) -> tuple[list, np.ndarray]:
        """
        Returns (formatted_data, target_wavenumbers).
          formatted_data     — list of dicts (see class docstring for structure)
          target_wavenumbers — 1-D np.ndarray of wavenumber values (cm⁻¹)
        Raises RuntimeError if process() has not been called yet.
        """
        if not self.formatted_data or self.target_wavenumbers is None:
            raise RuntimeError("Call process() before get_formatted_data().")
        return self.formatted_data, self.target_wavenumbers


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = PlasticIRDataset()
    ds.process()
    data, wn = ds.get_formatted_data()
    print(f"\nFinal: {len(data)} spectra × {len(wn)} wavenumber points")