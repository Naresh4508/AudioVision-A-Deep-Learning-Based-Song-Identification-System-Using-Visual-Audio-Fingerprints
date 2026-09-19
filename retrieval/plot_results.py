import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

FINGERPRINTS = "fingerprints/metric_fingerprints.npy"
META = "fingerprints/metric_metadata.txt"

fp = np.load(FINGERPRINTS)

with open(META) as f:
    metadata = [x.strip() for x in f]

sim = cosine_similarity(fp)

same = []
different = []

for i in range(len(metadata)):
    song_i = metadata[i].split("/")[-2]

    for j in range(i + 1, len(metadata)):
        song_j = metadata[j].split("/")[-2]

        if song_i == song_j:
            same.append(sim[i, j])
        else:
            different.append(sim[i, j])

same = np.array(same)
different = np.array(different)

plt.figure(figsize=(9, 6))

plt.hist(
    same,
    bins=40,
    alpha=0.6,
    label="Same Song"
)

plt.hist(
    different,
    bins=40,
    alpha=0.6,
    label="Different Song"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("AudioVision Fingerprint Similarity Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    "fingerprints/similarity_distribution.png",
    dpi=300
)

print("Saved:")
print("fingerprints/similarity_distribution.png")
