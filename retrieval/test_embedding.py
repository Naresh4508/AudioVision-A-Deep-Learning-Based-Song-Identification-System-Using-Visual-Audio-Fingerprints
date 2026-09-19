import torch
import numpy as np

from models.cnn import AudioCNN


MODEL_PATH = "models/cnn_best.pth"
SAMPLE_PATH = "spectrograms/Sample1/segment_0000.npy"


device = torch.device("cpu")


# Load model
model = AudioCNN(num_classes=7)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# Load spectrogram
spectrogram = np.load(SAMPLE_PATH)

# Normalize
spectrogram = (
    spectrogram + 80.0
) / 80.0

spectrogram = np.clip(
    spectrogram,
    0.0,
    1.0
)

# Convert to tensor
x = torch.tensor(
    spectrogram,
    dtype=torch.float32
)

# Add batch and channel dimensions
x = x.unsqueeze(0).unsqueeze(0)


# Extract fingerprint
with torch.no_grad():

    embedding = model.get_embedding(x)


print("Input shape:", x.shape)
print("Fingerprint shape:", embedding.shape)
print("Fingerprint:")
print(embedding)

print(
    "Fingerprint norm:",
    torch.norm(embedding).item()
)
