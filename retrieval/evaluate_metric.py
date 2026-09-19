import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FINGERPRINTS = "fingerprints/metric_fingerprints.npy"
META = "fingerprints/metric_metadata.txt"

fingerprints = np.load(FINGERPRINTS)

with open(META) as f:
    metadata = [line.strip() for line in f]

similarities = cosine_similarity(fingerprints)

same = []
different = []

for i in range(len(metadata)):
    song_i = metadata[i].split("/")[-2]

    for j in range(i + 1, len(metadata)):
        song_j = metadata[j].split("/")[-2]

        if song_i == song_j:
            same.append(similarities[i, j])
        else:
            different.append(similarities[i, j])

same = np.array(same)
different = np.array(different)

print("=" * 55)
print("AudioVision Metric Learning Evaluation")
print("=" * 55)

print(f"Same-song pairs:      {len(same)}")
print(f"Different-song pairs: {len(different)}")

print()
print(f"Same-song mean similarity:      {np.mean(same):.4f}")
print(f"Different-song mean similarity: {np.mean(different):.4f}")

print()
print(f"Same-song median similarity:      {np.median(same):.4f}")
print(f"Different-song median similarity: {np.median(different):.4f}")

separation = np.mean(same) - np.mean(different)

print()
print(f"Separation: {separation:.4f}")
print("=" * 55)
