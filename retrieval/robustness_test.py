import numpy as np
import librosa
import soundfile as sf

from retrieval.identify_song import identify

SOURCE = "dataset/Sample1.mp3"
SR = 22050

audio, _ = librosa.load(
    SOURCE,
    sr=SR,
    mono=True,
    duration=15
)

np.random.seed(42)

noise = np.random.normal(
    0,
    0.05,
    len(audio)
)

noisy = audio + noise

noisy = noisy / max(
    1.0,
    np.max(np.abs(noisy))
)

sf.write(
    "queries/sample1_noisy.wav",
    noisy,
    SR
)

tests = {
    "Original": "queries/sample1_test.wav",
    "Noisy": "queries/sample1_noisy.wav"
}

print("=" * 60)
print("AudioVision Robustness Test")
print("=" * 60)

for name, file in tests.items():

    ranking, segments = identify(file)

    print()
    print(name)
    print("-" * 40)

    for i, (song, score) in enumerate(ranking[:3], 1):
        print(f"{i}. {song:<10} {score:.4f}")

    print("Prediction:", ranking[0][0])
