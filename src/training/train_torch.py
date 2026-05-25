import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from models.cnn_torch import Spectral1DCNN
from utils.losses import get_loss_function
from utils.augmentations import SpectralAugmentor
import numpy as np

# Global Pipeline Configuration
USE_WBCE = True
INPUT_LENGTH = 600
NUM_CLASSES = 5  # Example: [PET, PE, PP, PS, PLA]
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

def run_pipeline():
    print("==================================================")
    print("   CNN IR SPECTROSCOPY - PLASTICS SORT PIPELINE   ")
    print("==================================================")

    # 1. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Spectral1DCNN(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES).to(device)
    print(f"[INIT] Model architecture loaded. Target: {NUM_CLASSES} classes.")
    print(f"[INIT] Execution Device: {device}")

    # 2. Configure Loss and Minority Weights
    pos_weights = None
    if USE_WBCE:
        # Assigning heavier penalty to minority bioplastics (e.g., at indices 3 and 4)
        weights = [1.0, 1.0, 1.0, 5.0, 8.0]
        pos_weights = torch.tensor(weights).to(device)
        print(f"[LOSS] Weighted BCE Active. Penalty Matrix: {weights}")
    else:
        print("[LOSS] Standard BCE Active.")

    criterion = get_loss_function(use_wbce=USE_WBCE, pos_weights=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Simulate Data Flow (Validation/Dummy Step)
    print("[EXEC] Initializing training orchestration...")

    # Dummy spectral batch: (Batch, Channels=1, L=600)
    dummy_input = torch.randn(BATCH_SIZE, 1, INPUT_LENGTH).to(device)
    dummy_targets = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float().to(device)

    # Forward pass
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[STATUS] Initial loss calculated: {loss.item():.4f}")
    print("[SUCCESS] Pipeline initialized and validated.")

if __name__ == "__main__":
    run_pipeline()
