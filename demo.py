import sys
from retrieval.identify_song import identify

if len(sys.argv) != 2:
    print("Usage:")
    print("python demo.py <audio_file>")
    raise SystemExit

query = sys.argv[1]

ranking, segments = identify(query)

print()
print("=" * 60)
print("          AudioVision")
print("   Song From Its Visual Fingerprint")
print("=" * 60)

print()
print("Input:", query)
print("Segments analyzed:", segments)

print()
print("Top Matches")
print("-" * 60)

for i, (song, score) in enumerate(ranking[:5], 1):
    print(f"{i}. {song:<15} {score:.4f}")

print()
print("IDENTIFIED SONG")
print("-" * 60)
print(ranking[0][0])
print()
print("=" * 60)
