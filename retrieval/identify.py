import os
import sys
import tempfile

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from models.cnn import AudioCNN


MODEL_PATH = "models/cnn_best.pth"
DATABASE_PATH = "fingerprints/fingerprints.npy"
METADATA_PATH = "fingerprints/metadata.txt"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 5

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load CNN
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


# --------------------------------------------------
# Load fingerprint database
# --------------------------------------------------

database = np.load(DATABASE_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = [
        line.strip()
        for line in f
        if line.strip()
    ]


# --------------------------------------------------
# Convert audio to spectrogram
# --------------------------------------------------

def audio_to_spectrogram(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# --------------------------------------------------
# Generate fingerprint
# --------------------------------------------------

def generate_fingerprint(audio):

    required_samples = (
        SAMPLE_RATE * SEGMENT_DURATION
    )

    if len(audio) < required_samples:

        audio = np.pad(
            audio,
            (
                0,
                required_samples - len(audio)
            )
        )

    else:

        audio = audio[:required_samples]


    spectrogram = audio_to_spectrogram(
        audio
    )


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

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)


    with torch.no_grad():

        fingerprint = model.get_embedding(
            tensor
        )


    return fingerprint.cpu().numpy()[0]


# --------------------------------------------------
# Search database
# --------------------------------------------------

def search(fingerprint, top_k=5):

    query = fingerprint.reshape(1, -1)

    # Database fingerprints are already normalized
    similarities = np.dot(
        database,
        query.T
    ).flatten()

    indices = np.argsort(
        similarities
    )[::-1][:top_k]

    return [
        (metadata[i], similarities[i])
        for i in indices
    ]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m retrieval.identify <audio_file>"
        )

        sys.exit(1)


    input_file = sys.argv[1]


    if not os.path.exists(input_file):

        print(
            "Error: File not found:",
            input_file
        )

        sys.exit(1)


    print("=" * 60)
    print("AudioVision - Music Identification")
    print("=" * 60)

    print("Input:", input_file)


    # Load audio
    audio, sr = librosa.load(
        input_file,
        sr=SAMPLE_RATE,
        mono=True
    )


    duration = len(audio) / sr

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        "Generating visual fingerprint..."
    )


    fingerprint = generate_fingerprint(
        audio
    )


    print(
        "Fingerprint dimension:",
        fingerprint.shape[0]
    )


    results = search(
        fingerprint,
        top_k=5
    )


    print()
    print("=" * 60)
    print("TOP MATCHES")
    print("=" * 60)


    for rank, (name, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. {name:<40} "
            f"Similarity: {score:.4f}"
        )


    print("=" * 60)


if __name__ == "__main__":
    main()
