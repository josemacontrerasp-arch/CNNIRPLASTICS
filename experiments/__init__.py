"""Experiment harness for the CNNIRPLASTICS polymer-classification study.

This package sits ON TOP of the existing, validated modules
(data/format_data.py, preprocess.py, models/cnn_model_draft.py, models/rf_model.py)
and adds the orchestration + reporting needed to answer the project questions:

  1. Does including OpenSpecy help or hurt?
  2. Which preprocessing config is best?
  3. CNN vs RF head-to-head on the chosen config/data.
  4. Weathered-plastic (FLOPP-e) generalization.
  5. Bioplastic / chosen-material false positives (BLoP + NIST probes).
  6. Optimal confidence threshold for an "other / unknown" reject option,
     plus alternative OOD strategies.

Nothing here trains a 2D CNN; the 1D CNN and Random Forest are the scope.

Run the stage entrypoints from the project root, e.g.:
    python -m experiments.stage1_summary
"""
