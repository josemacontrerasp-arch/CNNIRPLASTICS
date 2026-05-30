"""
Cosine library-match classifier -- the no-training floor.

This represents "what the instrument's built-in library search does": match an
unknown spectrum to the most similar reference under cosine similarity. There is
NO training/optimization loop. `fit` only builds reference templates; `predict`
assigns the label of the highest-similarity reference.

It is the floor that any real ML model (RF, CNN, ...) must beat. If a learned
model cannot outperform plain cosine matching, it is not earning its complexity.

    mode="centroid" -> one mean spectrum per class (compact library).
    mode="nn"       -> keep every training spectrum (1-NN cosine).

Usage:
    clf = CosineLibraryClassifier(mode="centroid").fit(X_train, y_train)
    y_pred = clf.predict(X_test)
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CosineLibraryClassifier:
    def __init__(self, mode: str = "centroid"):
        if mode not in ("centroid", "nn"):
            raise ValueError(f"unknown mode '{mode}' (use 'centroid' or 'nn')")
        self.mode = mode
        self.templates_: np.ndarray | None = None
        self.template_labels_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CosineLibraryClassifier":
        """Build the reference library from the TRAIN FOLD ONLY.

        This is the only place training data touches the model -- keeping it
        inside the CV train split is what makes the evaluation leakage-free.
        """
        X = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0)
        y = np.asarray(y_train)
        self.classes_ = np.unique(y)

        if self.mode == "centroid":
            self.templates_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
            self.template_labels_ = self.classes_.copy()
        else:  # nn
            self.templates_ = X
            self.template_labels_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.templates_ is None:
            raise RuntimeError("call fit() before predict()")
        sims = self.decision_function(X)
        best = sims.argmax(axis=1)
        return self.template_labels_[best]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarity of each input row to every template."""
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        return cosine_similarity(X, self.templates_)

    def predict_with_score(self, X: np.ndarray):
        """Return (labels, best_similarity) -- useful for the --samples hook."""
        sims = self.decision_function(X)
        best = sims.argmax(axis=1)
        return self.template_labels_[best], sims[np.arange(len(best)), best]


if __name__ == "__main__":
    # Two well-separated synthetic classes -> perfect cosine separation.
    rng = np.random.RandomState(0)
    a = rng.rand(20, 50) + np.r_[np.ones(25), np.zeros(25)]
    b = rng.rand(20, 50) + np.r_[np.zeros(25), np.ones(25)]
    X = np.vstack([a, b])
    y = np.array(["A"] * 20 + ["B"] * 20)
    clf = CosineLibraryClassifier("centroid").fit(X, y)
    print("centroid acc:", (clf.predict(X) == y).mean())
    lbl, score = clf.predict_with_score(X[:3])
    print("sample preds:", list(zip(lbl, np.round(score, 3))))
