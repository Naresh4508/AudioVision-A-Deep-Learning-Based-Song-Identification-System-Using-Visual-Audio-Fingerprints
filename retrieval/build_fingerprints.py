import os
import numpy as np
import torch

from models.cnn import AudioCNN


SPECTROGRAM_DIR = "spectrograms"
MODEL_PATH = "models/cnn_best.pth"
OUTPUT_DIR = "fingerprints"

DEVICE = torch.device("cpu")


os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = AudioCNN(num_classes=7)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


print("CNN model loaded.")


# --------------------------------------------------
# Find spectrograms
# --------------------------------------------------

samples = []

for song_name in sorted(
    os.listdir(SPECTROGRAM_DIR)
):

    song_dir = os.path.join(
        SPECTROGRAM_DIR,
        song_name
    )

    if not os.path.isdir(song_dir):
        continue

    for filename in sorted(
        os.listdir(song_dir)
    ):

        if filename.endswith(".npy"):

            path = os.path.join(
                song_dir,
                filename
            )

            samples.append(
                (song_name, filename, path)
            )


print(
    "Spectrograms found:",
    len(samples)
)


# --------------------------------------------------
# Generate fingerprints
# --------------------------------------------------

fingerprints = []
metadata = []


for index, (song, filename, path) in enumerate(samples):

    spectrogram = np.load(path)

    # Normalize spectrogram
    spectrogram = (
        spectrogram + 80.0
    ) / 80.0

    spectrogram = np.clip(
        spectrogram,
        0.0,
        1.0
    )

    tensor = torch.tensor(
        spectrogram,
        dtype=torch.float32
    )

    # Add batch and channel dimensions
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)


    with torch.no_grad():

        embedding = model.get_embedding(
            tensor
        )


    embedding = embedding.squeeze(0)

    fingerprints.append(
        embedding.cpu().numpy()
    )

    metadata.append(
        f"{song}/{filename}"
    )


    if (index + 1) % 50 == 0:

        print(
            f"Processed {index + 1}/{len(samples)}"
        )


# --------------------------------------------------
# Save database
# --------------------------------------------------

fingerprints = np.array(
    fingerprints,
    dtype=np.float32
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "fingerprints.npy"
    ),
    fingerprints
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.txt"
    ),
    "w"
) as file:

    for item in metadata:
        file.write(item + "\n")


print()
print("=" * 50)
print("Fingerprint database created")
print("=" * 50)

print(
    "Fingerprint matrix:",
    fingerprints.shape
)

print(
    "Saved:",
    "fingerprints/fingerprints.npy"
)

print(
    "Saved:",
    "fingerprints/metadata.txt"
)
