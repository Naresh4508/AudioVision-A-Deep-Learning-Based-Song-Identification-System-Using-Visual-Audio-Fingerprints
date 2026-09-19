import numpy as np
import matplotlib.pyplot as plt

file = "spectrograms/Sample1/segment_0000.npy"

mel = np.load(file)

plt.figure(figsize=(10, 5))

plt.imshow(
    mel,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="dB")

plt.title("AudioVision - 5 Second Audio Fingerprint")
plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency Bins")

plt.tight_layout()

plt.savefig(
    "spectrograms/sample_segment.png",
    dpi=150
)

plt.show()
