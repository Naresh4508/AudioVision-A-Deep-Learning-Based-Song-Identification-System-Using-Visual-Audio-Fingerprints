import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "dataset/sample.wav"
OUTPUT_DIR = "spectrograms"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading:", INPUT_FILE)

audio, sr = librosa.load(
    INPUT_FILE,
    sr=22050,
    mono=True
)

print("Audio loaded successfully")
print("Sample rate:", sr)
print("Duration:", round(len(audio) / sr, 2), "seconds")

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

print("Spectrogram shape:", mel_db.shape)

output_array = os.path.join(
    OUTPUT_DIR,
    "sample.npy"
)

output_image = os.path.join(
    OUTPUT_DIR,
    "sample.png"
)

np.save(output_array, mel_db)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("AudioVision - Sample1 Mel-Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    output_image,
    dpi=150
)

plt.close()

print("Saved NumPy array:", output_array)
print("Saved spectrogram:", output_image)
print("Processing complete!")
