import os
import glob
import numpy as np
import torch

from models.metric_cnn import MetricCNN

DEVICE = torch.device("cpu")
MODEL_PATH = "models/metric_cnn.pth"
SPEC_DIR = "spectrograms"
OUT_PATH = "fingerprints/metric_fingerprints.npy"
META_PATH = "fingerprints/metric_metadata.txt"

os.makedirs("fingerprints", exist_ok=True)

model = MetricCNN(embedding_dim=128).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

files = sorted(glob.glob(f"{SPEC_DIR}/*/*.npy"))

fingerprints = []
metadata = []

with torch.no_grad():
    for i, file in enumerate(files):
        spec = np.load(file)

        # Normalize spectrogram
        spec = (spec + 80.0) / 80.0
        spec = np.clip(spec, 0, 1)

        x = torch.tensor(spec, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

        embedding = model(x)
        embedding = embedding.cpu().numpy()[0]

        fingerprints.append(embedding)
        metadata.append(file)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)}")

fingerprints = np.array(fingerprints)

np.save(OUT_PATH, fingerprints)

with open(META_PATH, "w") as f:
    for item in metadata:
        f.write(item + "\n")

print("\nMetric fingerprint database created.")
print("Shape:", fingerprints.shape)
print("Saved:", OUT_PATH)
print("Metadata:", META_PATH)
