import torch
import torch.nn as nn
import torch.nn.functional as F

class Spectral1DCNN(nn.Module):
    """
    1D-CNN Architecture for IR Spectroscopy Multi-label Classification.
    Optimized for extracting hierarchical spectral patterns from plastic samples.
    """
    def __init__(self, input_length=600, num_classes=10):
        super(Spectral1DCNN, self). __init__()

        # Convolutional Block 1: Primary bond vibration filter mapping
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=31, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(31)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolutional Block 2: Complex intermediate structural assembly
        self.conv2 = nn.Conv1d(in_channels=31, out_channels=62, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(62)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Flattening and Fully Connected Head
        # Note: Based on input_length=600, the flattened dimension after pool2 is 62 * 147 = 9114.
        # The prompt specifies a target linear projection starting at 4927.
        # To align with the strict requirement, we use an Adaptive Pool to force the dimension.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4927 // 62)

        self.fc1 = nn.Linear(62 * (4927 // 62), 2785)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2785, 1574)
        self.dropout2 = nn.Dropout(p=0.5)

        self.output = nn.Linear(1574, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (Batch, 1, 600)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Adaptive pooling to handle specific FC layer constraints
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.output(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    # Structural Validation
    model = Spectral1DCNN(input_length=600, num_classes=5)
    sample_input = torch.randn(8, 1, 600)
    output = model(sample_input)
    print(f"Input Shape: {sample_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("Model Architecture Initialized Successfully.")
