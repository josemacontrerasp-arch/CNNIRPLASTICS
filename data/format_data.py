import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional

# -----------------------------------------------------------------
# File paths
# Adjust these to match your directory layout.
# -----------------------------------------------------------------

FTIR_C4_PATH               = "FTIR_PLASTIC_c4.csv"
FTIR_C8_PATH               = "FTIR_PLASTIC_c8.csv"
OPENSPECY_DATASET_PATH     = "openspecy_polymer_dataset.csv"
OPENSPECY_METADATA_PATH    = "openspecy_polymer_metadata.csv"
OPENSPECY_WAVENUMBERS_PATH = "openspecy_wavenumbers.csv"

# SLoPP/SLoPP-E is excluded: it is a Raman spectral library, not FTIR.
# SLOPP_PATH = "slopp_and_slopp_e/slopp_and_slopp_e.zip"

# -----------------------------------------------------------------
# Label definitions
# -----------------------------------------------------------------

# NOTE: PLA and PHA are bioplastics. They are only populated once the OpenSpecy
# dataset is regenerated with the extended keyword list in
# data/data_processing/openspecy_dataset_builder.R (which requires R + a re-download
# of the raw OpenSpecy library). Until then no training spectrum carries these
# labels, so they remain empty classes. n_classes is derived from len() of this
# list everywhere, so adding them here automatically resizes the CNN softmax / RF.
POLYMER_CLASSES = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET", "PLA", "PHA"]
LABEL_TO_INT    = {label: i for i, label in enumerate(POLYMER_CLASSES)}

LAB_LABELS = {
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

LAB_HEADER_LINES = 4  # ##TITLE, ##DATA TYPE, ##XUNITS, ##YUNITS


# -----------------------------------------------------------------

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
        # -- Paths --------------------------------------------------
        self.ftir_c4_path               = ftir_c4_path
        self.ftir_c8_path               = ftir_c8_path
        self.openspecy_dataset_path     = openspecy_dataset_path
        self.openspecy_metadata_path    = openspecy_metadata_path
        self.openspecy_wavenumbers_path = openspecy_wavenumbers_path

        # Path to the real-world lab .txt folder, set by add_lab_data().
        # If set before process(), lab data is included in the wavenumber
        # range/resolution decision. If set after, it is aligned immediately.
        self._lab_folder_path: Optional[str] = None

        # Tracks whether lab data has been loaded into _raw_data, so that
        # load_raw() stays idempotent even if called multiple times.
        self._lab_loaded: bool = False

        # -- Raw data -----------------------------------------------
        # Populated by load_raw(). Keys: 'ftir_c4', 'ftir_c8', 'openspecy'.
        # Each value is a dict; structure differs by source (see _load_ftir /
        # _load_openspecy for exact keys). Intentionally unexposed until
        # get_raw_data() is called, so callers know loading has occurred.
        self._raw_data: dict   = {}
        self._raw_loaded: bool = False

        # -- Analysis results ---------------------------------------
        # Filled by process() before alignment decisions are made.
        # Keyed by dataset name; each value is a stats dict (range, step, count).
        self._dataset_stats: dict = {}

        # -- Target grid --------------------------------------------
        # 1-D array of wavenumber values (cm^-1) that every formatted spectrum
        # is aligned to. Decided during process() as the coarsest grid within
        # the shared range of all datasets.
        self.target_wavenumbers: Optional[np.ndarray] = None

        # -- Formatted data -----------------------------------------
        # Empty until process() completes. Each entry is a dict:
        #   {
        #     'label':       str          -- polymer class, e.g. 'PET'
        #     'label_int':   int          -- integer index per LABEL_TO_INT
        #     'intensities': np.ndarray   -- aligned to target_wavenumbers
        #     'source':      str          -- 'ftir_c4' | 'ftir_c8' | 'openspecy' | 'lab'
        #   }
        self.formatted_data: list = []

    # -----------------------------------------------------------------
    # Private -- loading
    # -----------------------------------------------------------------

    def _load_ftir(self, path: str, key: str) -> None:
        """
        Parses one of the FTIR CSVs (c4 or c8).

        Layout: 6 metadata columns (IDE, Polymer, Technic, Sample, BR, RST)
        followed by interleaved x/y pairs -- Data(x), Data(y), Data(x), Data(y), ...

        Wavenumbers are stored as a 2-D array (n_spectra x n_points) because
        the spectrometer wavenumber positions vary very slightly between
        measurements. In practice the variance is tiny, but we preserve it here
        so _align_spectrum can use each spectrum's own x-axis exactly.
        """
        df = pd.read_csv(path, header=0)

        data_cols = df.columns[6:]       # everything after the 6 metadata cols
        x_cols    = data_cols[0::2]      # columns 0, 2, 4, ... -> wavenumbers
        y_cols    = data_cols[1::2]      # columns 1, 3, 5, ... -> intensities

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
          - metadata row i  <->  dataset row i  (aligned by the R builder script)
          - dataset column j (V1...V1983) <-> wavenumber file row j

        After filtering, 'wavenumbers' is a shared 1-D axis (all spectra identical).

        NaN handling:
          OpenSpecy intensities contain large leading and trailing NaN blocks (the
          instrument/library did not record data outside a per-spectrum range) plus
          scattered intermediate NaN values.  We compute the conservative valid
          wavenumber range -- the intersection of every spectrum's first-to-last
          valid point -- and store it as 'effective_wn_min' / 'effective_wn_max'.
          _analyze_dataset uses these instead of the raw axis endpoints (102-11994)
          so that the target grid is anchored to where data actually exists.
          The NaN values themselves are left in place here; _align_spectrum strips
          them per-spectrum before interpolating.
        """
        metadata    = pd.read_csv(self.openspecy_metadata_path, low_memory=False)
        dataset     = pd.read_csv(self.openspecy_dataset_path,  low_memory=False)
        wavenumbers = (
            pd.read_csv(self.openspecy_wavenumbers_path)["wavenumber"]
            .to_numpy(dtype=float)
        )

        # Drop Raman -- keep only FTIR
        ftir_mask       = metadata["spectrum_type"].str.lower().str.strip() == "ftir"
        n_raman_dropped = int((~ftir_mask).sum())

        dataset_ftir = dataset[ftir_mask].reset_index(drop=True)

        intensity_cols = [c for c in dataset_ftir.columns if c != "label"]
        intensities    = dataset_ftir[intensity_cols].to_numpy(dtype=float)
        labels         = dataset_ftir["label"].to_numpy()

        # -- Compute effective valid wavenumber range ---------------
        # For each spectrum find the index of its first and last non-NaN value.
        # The conservative intersection (max of firsts, min of lasts) gives the
        # range where every single spectrum has at least one valid data point --
        # i.e. the floor for safe interpolation with no out-of-bounds NaN output.
        valid_mask = ~np.isnan(intensities)

        # argmax on a bool array returns the index of the first True
        first_valid_idx = np.argmax(valid_mask, axis=1)
        # reverse-axis trick for last True
        last_valid_idx  = (valid_mask.shape[1] - 1
                           - np.argmax(valid_mask[:, ::-1], axis=1))

        effective_wn_min = float(wavenumbers[first_valid_idx].max())
        effective_wn_max = float(wavenumbers[last_valid_idx].min())

        n_at_floor = int((wavenumbers[first_valid_idx] == effective_wn_min).sum())
        n_above_floor = int((wavenumbers[first_valid_idx] > effective_wn_min).sum())
        print(f"  OpenSpecy effective range: "
              f"{effective_wn_min:.0f}-{effective_wn_max:.0f} cm^-1  "
              f"({n_at_floor} spectra start at floor, "
              f"{n_above_floor} start above it and will be dropped after alignment)")

        self._raw_data["openspecy"] = {
            "wavenumbers":      wavenumbers,   # 1-D shared axis (full, 102-11994)
            "intensities":      intensities,
            "labels":           labels,
            "n_spectra":        int(len(labels)),
            "n_points":         int(len(wavenumbers)),
            "n_raman_dropped":  n_raman_dropped,
            # Conservative valid range -- used by _analyze_dataset instead of raw axis
            "effective_wn_min": effective_wn_min,
            "effective_wn_max": effective_wn_max,
        }

    def _load_lab_data(self) -> None:
        """
        Reads all .txt files from self._lab_folder_path.
        Each file has LAB_HEADER_LINES metadata lines followed by
        tab-separated (wavenumber, intensity) pairs.
        Wavenumbers are stored as 2-D (n_spectra x n_points) to mirror
        the FTIR format, even though in practice they are identical
        across all lab files.
        """
        import glob

        folder = self._lab_folder_path
        paths  = sorted(glob.glob(os.path.join(folder, "*.txt")))

        if not paths:
            raise FileNotFoundError(f"No .txt files found in lab folder: {folder!r}")

        all_wavenumbers = []
        all_intensities = []
        labels          = []

        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem not in LAB_LABELS:
                raise KeyError(
                    f"No label defined for '{os.path.basename(path)}'. "
                    f"Add it to LAB_LABELS before running."
                )

            xs, ys = [], []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i < LAB_HEADER_LINES:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(
                            f"Unexpected line in {path!r}: {line!r}"
                        )
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))

            all_wavenumbers.append(xs)
            all_intensities.append(ys)
            labels.append(LAB_LABELS[stem])

        wavenumbers = np.array(all_wavenumbers, dtype=float)  # (n_spectra, n_points)
        intensities = np.array(all_intensities, dtype=float)

        self._raw_data["lab"] = {
            "wavenumbers": wavenumbers,
            "intensities": intensities,
            "labels":      np.array(labels),
            "n_spectra":   int(len(labels)),
            "n_points":    int(wavenumbers.shape[1]),
        }
        self._lab_loaded = True

    # -----------------------------------------------------------------
    # Private -- analysis & alignment
    # -----------------------------------------------------------------

    def _analyze_dataset(self, key: str) -> dict:
        """
        Computes per-dataset wavenumber statistics used to decide the target grid.
        For FTIR (2-D wavenumber array) the mean axis across spectra is used;
        the per-position standard deviation is reported as a sanity check.

        For OpenSpecy the raw wavenumber axis spans 102-11994 cm^-1, but large
        leading/trailing NaN blocks in the intensity matrix mean that range is
        misleading.  If the raw_data entry carries 'effective_wn_min' /
        'effective_wn_max' (set by _load_openspecy), those are used for wn_min /
        wn_max so that _decide_target_wavenumbers anchors to real data.
        """
        raw = self._raw_data[key]
        wn  = raw["wavenumbers"]

        if wn.ndim == 2:
            mean_wn   = wn.mean(axis=0)
            # Average std across all wavenumber positions -- should be near zero
            wn_jitter = float(wn.std(axis=0).mean())
        else:
            mean_wn   = wn
            wn_jitter = 0.0

        steps = np.diff(mean_wn)

        # Use pre-computed effective range if available (OpenSpecy), otherwise
        # fall back to the raw axis min/max (using nanmin/nanmax for safety).
        wn_min = raw.get("effective_wn_min", float(np.nanmin(mean_wn)))
        wn_max = raw.get("effective_wn_max", float(np.nanmax(mean_wn)))

        return {
            "n_spectra":  raw["n_spectra"],
            "n_points":   raw["n_points"],
            "wn_min":     wn_min,
            "wn_max":     wn_max,
            "mean_step":  float(steps.mean()),
            "min_step":   float(steps.min()),
            "max_step":   float(steps.max()),
            "mean_wn":    mean_wn,     # representative 1-D axis for this dataset
            "wn_jitter":  wn_jitter,   # FTIR only: how much wn varies between spectra
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

        shared_min = max(s["wn_min"] for s in stats.values())
        shared_max = min(s["wn_max"] for s in stats.values())

        coarsest_key = max(stats, key=lambda k: stats[k]["mean_step"])
        coarsest_wn  = stats[coarsest_key]["mean_wn"]

        mask   = (coarsest_wn >= shared_min) & (coarsest_wn <= shared_max)
        target = coarsest_wn[mask]

        print(f"\nTarget wavenumber grid decision:")
        print(f"  Shared range:      {shared_min:.1f} - {shared_max:.1f} cm^-1")
        print(f"  Reference dataset: '{coarsest_key}'"
              f"  (step ~= {stats[coarsest_key]['mean_step']:.2f} cm^-1)")
        print(f"  Grid points:       {len(target)}")

        return target

    def _align_spectrum(
        self,
        wn_src:      np.ndarray,
        intensities: np.ndarray,
        target_wn:   np.ndarray,
    ) -> np.ndarray:
        """
        Resamples a single spectrum from its source grid onto target_wn using
        linear interpolation.

        NaN handling:
          Before building the interpolator, all NaN values are stripped from the
          (wn_src, intensities) pair.  This has two effects:
            - Leading/trailing NaN blocks are excluded, so the interpolator's
              valid range is the spectrum's actual measured range.
            - Intermediate NaN values are bridged: the interpolator connects the
              last valid point before the gap to the first valid point after it,
              producing a linearly interpolated fill rather than propagating NaN.

          Any target point that falls outside the spectrum's valid range (i.e. the
          spectrum genuinely did not cover that wavenumber) is returned as NaN.
          process() filters those entries out after alignment and reports the count.
        """
        valid = ~np.isnan(intensities)
        if valid.sum() < 2:
            # Degenerate spectrum -- not enough points to interpolate
            return np.full(len(target_wn), np.nan)

        f = interp1d(
            wn_src[valid], intensities[valid],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return f(target_wn)

    # -----------------------------------------------------------------
    # Public -- loading
    # -----------------------------------------------------------------

    def load_raw(self) -> None:
        """
        Reads all source files into _raw_data without any alignment or processing.
        Idempotent -- calling it a second time is a no-op (except that it will
        still pick up lab data if add_lab_data() was called after the first load).
        """
        if self._raw_loaded:
            # Still check for lab data if it was registered after the initial load
            if self._lab_folder_path and not self._lab_loaded:
                print("Loading lab data ...")
                self._load_lab_data()
            return

        print("Loading FTIR c4 ...")
        self._load_ftir(self.ftir_c4_path, "ftir_c4")
        print("Loading FTIR c8 ...")
        self._load_ftir(self.ftir_c8_path, "ftir_c8")
        print("Loading OpenSpecy ...")
        self._load_openspecy()

        if self._lab_folder_path:
            print("Loading lab data ...")
            self._load_lab_data()

        self._raw_loaded = True
        dropped = self._raw_data["openspecy"]["n_raman_dropped"]
        print(f"Raw loading complete.  ({dropped} Raman spectra discarded from OpenSpecy)")

    def add_lab_data(self, folder_path: str) -> None:
        """
        Registers the real-world lab data folder for inclusion in the dataset.

        Behaviour depends on whether process() has already been called:

        Before process():
            Stores the path. The lab data will be loaded during process() and
            included in the wavenumber range/resolution decision alongside the
            other datasets.

        After process():
            Loads and parses the lab files immediately, aligns each spectrum to
            the already-decided target_wavenumbers, and appends the entries to
            formatted_data. The target grid is not re-evaluated.
        """
        self._lab_folder_path = os.path.abspath(folder_path)

        if not self.formatted_data:
            # process() hasn't run yet -- just register the path and return.
            # load_raw() / process() will pick it up.
            print("Lab folder registered. It will be included in process().")
            return

        # process() has already run -- load and align now.
        if not self._lab_loaded:
            print("Loading lab data ...")
            self._load_lab_data()
        else:
            print("Lab data already loaded; re-aligning to existing target grid.")

        raw        = self._raw_data["lab"]
        per_row_wn = raw["wavenumbers"].ndim == 2

        n_before = len(self.formatted_data)
        for i in range(raw["n_spectra"]):
            wn_src  = raw["wavenumbers"][i] if per_row_wn else raw["wavenumbers"]
            aligned = self._align_spectrum(wn_src, raw["intensities"][i],
                                           self.target_wavenumbers)
            label = str(raw["labels"][i])
            self.formatted_data.append({
                "label":       label,
                "label_int":   LABEL_TO_INT.get(label, -1),
                "intensities": aligned,
                "source":      "lab",
            })

        n_added = len(self.formatted_data) - n_before
        print(f"Lab data added. {n_added} spectra appended "
              f"(total now: {len(self.formatted_data)})")

    # -----------------------------------------------------------------
    # Public -- processing
    # -----------------------------------------------------------------

    def process(self) -> None:
        """
        Full pipeline:
          1. Load raw data (if not already done)
          2. Analyse each dataset -- range, resolution, count
          3. Decide target wavenumber grid
          4. Align every spectrum to the target grid
          5. Filter out any spectrum with remaining NaN intensities
          6. Populate self.formatted_data and self.target_wavenumbers
        """
        self.load_raw()

        # Step 2
        print("\nAnalysing datasets ...")
        for key in self._raw_data:
            self._dataset_stats[key] = self._analyze_dataset(key)
            s = self._dataset_stats[key]
            jitter_str = (f"  wn-jitter ~= {s['wn_jitter']:.4f}"
                          if s["wn_jitter"] > 0 else "")
            print(
                f"  {key:<12s}  {s['n_spectra']:5d} spectra  "
                f"{s['n_points']:5d} pts  "
                f"range {s['wn_min']:.0f}-{s['wn_max']:.0f} cm^-1  "
                f"step ~= {s['mean_step']:.2f} cm^-1"
                + jitter_str
            )

        # Step 3
        self.target_wavenumbers = self._decide_target_wavenumbers()

        # Steps 4 + 5
        print("\nAligning spectra ...")
        self.formatted_data = []

        for key in self._raw_data:
            raw        = self._raw_data[key]
            per_row_wn = raw["wavenumbers"].ndim == 2  # True for FTIR/lab, False for OpenSpecy

            for i in range(raw["n_spectra"]):
                wn_src  = raw["wavenumbers"][i] if per_row_wn else raw["wavenumbers"]
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

        # Step 5: remove any spectrum that still contains NaN after alignment.
        # This happens when a spectrum's valid range does not fully cover the
        # target grid (e.g. an OpenSpecy spectrum that starts at 600 cm^-1 while
        # the target starts at ~402 cm^-1).  These cannot be salvaged without
        # extrapolation, which we avoid.
        n_before = len(self.formatted_data)
        self.formatted_data = [
            e for e in self.formatted_data
            if not np.isnan(e["intensities"]).any()
        ]
        n_dropped = n_before - len(self.formatted_data)
        if n_dropped:
            print(f"\n  Dropped {n_dropped} spectra whose valid range did not "
                  f"fully cover the target grid "
                  f"({n_before} -> {len(self.formatted_data)})")

        # Hard guarantee -- this should never fire
        for entry in self.formatted_data:
            if np.isnan(entry["intensities"]).any():
                raise RuntimeError(
                    f"NaN survived filtering in a '{entry['source']}' / "
                    f"'{entry['label']}' entry -- this is a bug."
                )

        # Summary
        print(f"\nProcessing complete -- {len(self.formatted_data)} spectra total")
        label_counts: dict = {}
        for entry in self.formatted_data:
            label_counts[entry["label"]] = label_counts.get(entry["label"], 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

    # -----------------------------------------------------------------
    # Public -- accessors
    # -----------------------------------------------------------------

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
          formatted_data     -- list of dicts (see class docstring for structure)
          target_wavenumbers -- 1-D np.ndarray of wavenumber values (cm^-1)
        Raises RuntimeError if process() has not been called yet.
        """
        if not self.formatted_data or self.target_wavenumbers is None:
            raise RuntimeError("Call process() before get_formatted_data().")
        return self.formatted_data, self.target_wavenumbers

    def get_data_lists(self) -> list:
        """
        Returns [labels, intensities] where labels[i] and intensities[i]
        correspond to the same spectrum.
        labels      -- list of str, e.g. ['PET', 'HDPE', ...]
        intensities -- list of np.ndarray, each aligned to target_wavenumbers
        Raises RuntimeError if process() has not been called yet.
        """
        if not self.formatted_data or self.target_wavenumbers is None:
            raise RuntimeError("Call process() before get_data_lists().")

        labels      = [e["label"]       for e in self.formatted_data]
        intensities = [e["intensities"] for e in self.formatted_data]

        return [labels, intensities]


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------

if __name__ == "__main__":
    ds = PlasticIRDataset()
    ds.process()
    data, wn = ds.get_formatted_data()
    print(f"\nFinal: {len(data)} spectra x {len(wn)} wavenumber points")
