import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from cnn_model_draft import load_and_preprocess
import time

# ===== SPEED OPTIMISATIONS =====
tf.config.optimizer.set_jit(True)  # XLA acceleration


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "output" / "fold_1.keras"

    # ===== LOAD MODEL =====
    model = keras.models.load_model(model_path)

    # ===== FAST INFERENCE WRAPPER =====
    @tf.function
    def infer(x):
        return model(x, training=False)

    # ===== WARM-UP (VERY IMPORTANT) =====
    dummy = tf.zeros((1, 1868, 1), dtype=tf.float32)
    infer(dummy)

    # ===== LOAD DATA =====
    absorbance_values, labels = load_and_preprocess()

    class_names = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]

    # ===== PICK RANDOM SAMPLE =====
    sample_index = np.random.randint(0, len(absorbance_values))

    sample = absorbance_values[sample_index:sample_index + 1].astype(np.float32)

    x = tf.convert_to_tensor(sample, dtype=tf.float32)

    # ===== FAST BENCHMARK (OPTION B) =====
    times = []

    for _ in range(100):
        start = time.perf_counter()
        pred = infer(x)
        end = time.perf_counter()
        times.append(end - start)

    # ===== FINAL PREDICTION (reuse last run) =====
    predicted_class = tf.argmax(pred, axis=1).numpy()[0]
    confidence = tf.reduce_max(pred).numpy()

    true_class = labels[sample_index]

    # ===== RESULTS =====
    print("\n===== FAST SINGLE TEST (BENCHMARK MODE) =====")
    print(f"Sample index: {sample_index}")
    print(f"True label:      {class_names[true_class]}")
    print(f"Predicted label: {class_names[predicted_class]}")
    print(f"Confidence:      {float(confidence):.4f}")

    print("\n===== SPEED RESULTS =====")
    print(f"Avg inference time: {np.mean(times)*1000:.3f} ms")
    print(f"Min inference time: {np.min(times)*1000:.3f} ms")
    print(f"Max inference time: {np.max(times)*1000:.3f} ms")


if __name__ == "__main__":
    main()