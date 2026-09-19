import os
import numpy as np
import librosa
import torch
from collections import defaultdict

from models.metric_cnn import MetricCNN

MODEL_PATH = "models/metric_cnn.pth"
FINGERPRINTS = "fingerprints/metric_fingerprints.npy"
METADATA = "fingerprints/metric_metadata.txt"

SR = 22050
SEGMENT_SECONDS = 5
N_SAMPLES = SR * SEGMENT_SECONDS

device = torch.device("cpu")

model = MetricCNN(embedding_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

database = np.load(FINGERPRINTS)

with open(METADATA) as f:
    metadata = [x.strip() for x in f]


def fingerprint(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel + 80) / 80
    mel = np.clip(mel, 0, 1)

    if mel.shape[1] < 216:
        mel = np.pad(
            mel,
            ((0, 0), (0, 216 - mel.shape[1]))
        )
    else:
        mel = mel[:, :216]

    x = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        emb = model(x).cpu().numpy()[0]

    return emb / (np.linalg.norm(emb) + 1e-8)


def identify(query_file):

    audio, _ = librosa.load(
        query_file,
        sr=SR,
        mono=True
    )

    segments = []

    for start in range(0, len(audio), N_SAMPLES):

        segment = audio[start:start + N_SAMPLES]

        if len(segment) < N_SAMPLES:
            segment = np.pad(
                segment,
                (0, N_SAMPLES - len(segment))
            )

        segments.append(segment)

    song_scores = defaultdict(list)

    for segment in segments:

        query_fp = fingerprint(segment)

        similarities = database @ query_fp

        # Group all database segments by song
        song_similarities = defaultdict(list)

        for i, score in enumerate(similarities):
            song = metadata[i].split("/")[-2]
            song_similarities[song].append(float(score))

        # For each song, use its strongest 3 matches
        for song, scores in song_similarities.items():

            scores.sort(reverse=True)

            best = scores[:3]

            song_scores[song].append(np.mean(best))

    # Average evidence across query segments
    final_scores = {}

    for song, scores in song_scores.items():
        final_scores[song] = np.mean(scores)

    ranking = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking, len(segments)


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m retrieval.identify_song <audio_file>")
        raise SystemExit

    query = sys.argv[1]

    ranking, count = identify(query)

    print("=" * 60)
    print("AudioVision - Song Identification")
    print("=" * 60)
    print("Query:", query)
    print("Query segments:", count)
    print()

    print("SONG RANKING")
    print("-" * 60)

    for i, (song, score) in enumerate(ranking, 1):
        print(f"{i}. {song:<12} {score:.4f}")

    print()
    print("Predicted song:", ranking[0][0])
    print("Similarity score:", f"{ranking[0][1]:.4f}")
    print("=" * 60)
