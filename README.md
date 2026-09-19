# AudioVision — Song From Its Visual Fingerprint

> A deep learning mini-project that identifies a song from its learned **visual audio fingerprint** using Mel-spectrograms, CNN-based metric learning, and cosine similarity.

## Overview

**AudioVision** converts audio into visual time-frequency representations called **Mel-spectrograms** and learns a compact **128-dimensional fingerprint** for each audio segment.

Instead of using only conventional classification, AudioVision uses **triplet metric learning** so that:

* Segments from the **same song** have similar fingerprints.
* Segments from **different songs** have dissimilar fingerprints.
* An unknown query can be matched against a fingerprint database using **cosine similarity**.

### Pipeline

```text
Audio
  ↓
5-Second Segmentation
  ↓
Mel-Spectrogram
  ↓
CNN Feature Extraction
  ↓
Triplet Metric Learning
  ↓
128-Dimensional Fingerprint
  ↓
Cosine Similarity
  ↓
Song Ranking
  ↓
Identified Song
```

---

## Features

* 🎵 Audio preprocessing and segmentation
* 📊 Mel-spectrogram generation
* 🧠 MLP classification baseline
* 🧠 CNN classification baseline
* 🔗 CNN-based metric learning
* 🔢 128-dimensional audio fingerprints
* 🔍 Similarity-based song retrieval
* 📈 Fingerprint similarity analysis
* 🔊 Basic noise robustness testing
* 💻 CPU-compatible implementation

---

## Dataset

The prototype uses **7 reference songs**:

```text
Sample1
Sample2
Sample3
Sample4
Sample5
Sample6
Sample7
```

The recordings are divided into five-second segments.

| Parameter        |     Value |
| ---------------- | --------: |
| Songs            |         7 |
| Segments         |       350 |
| Segment duration | 5 seconds |
| Sampling rate    | 22,050 Hz |
| Channels         |      Mono |
| Mel bands        |       128 |
| FFT size         |      2048 |
| Hop length       |       512 |
| Fingerprint size |     128-D |

> The dataset is intentionally small because this is a mini-project proof of concept.

---

## Model Architecture

The metric-learning CNN uses:

```text
Input Mel-Spectrogram
        │
        ▼
Conv2D: 1 → 16
ReLU + MaxPool
        │
        ▼
Conv2D: 16 → 32
ReLU + MaxPool
        │
        ▼
Conv2D: 32 → 64
ReLU
        │
        ▼
Adaptive Average Pooling
        │
        ▼
Linear: 64 → 128
        │
        ▼
L2 Normalization
        │
        ▼
128-D Fingerprint
```

### Metric Learning

Training uses triplets:

```text
Anchor
  │
  ├── Positive → same song
  │
  └── Negative → different song
```

The model is trained using:

```text
TripletMarginLoss
Margin = 0.3
```

---

## Results

### Baseline Models

| Model |                              Result |
| ----- | ----------------------------------: |
| MLP   | **75.71%** peak validation accuracy |
| CNN   | **55.71%** best validation accuracy |

These are segment-level baseline results using a random split and should not be interpreted as independent song-level generalization accuracy.

### Metric Learning

| Metric                         |     Result |
| ------------------------------ | ---------: |
| Initial triplet loss           |     0.2635 |
| Final triplet loss             | **0.0588** |
| Same-song pairs                |      9,075 |
| Different-song pairs           |     52,000 |
| Same-song mean similarity      | **0.9433** |
| Different-song mean similarity | **0.6963** |
| Similarity separation          | **0.2470** |

### Song Identification

A known query generated from `Sample1.mp3` was correctly identified:

```text
1. Sample1    0.9890
2. Sample5    0.9751
3. Sample6    0.9734
4. Sample4    0.8524
5. Sample2    0.8418
6. Sample3    0.7858
7. Sample7    0.7814
```

**Predicted song: Sample1**

The `0.9890` value is a cosine similarity score, not a probability.

### Noise Test

| Condition | Prediction |  Score |
| --------- | ---------- | -----: |
| Original  | Sample1    | 0.9890 |
| Noisy     | Sample6    | 0.9754 |

The correct song, Sample1, remained a strong candidate at `0.9475` under noise, but was no longer ranked first. This demonstrates a current robustness limitation.

---

## Project Structure

```text
AudioVision/
│
├── dataset/
│   ├── Sample1.mp3
│   ├── Sample2.mp3
│   ├── ...
│   └── Sample7.mp3
│
├── preprocessing/
│   ├── audio_to_spectrogram.py
│   └── build_dataset.py
│
├── spectrograms/
│
├── models/
│   ├── mlp.py
│   ├── cnn.py
│   └── metric_cnn.py
│
├── training/
│   ├── dataset.py
│   ├── triplet_dataset.py
│   ├── train_mlp.py
│   ├── train_cnn.py
│   └── train_metric.py
│
├── retrieval/
│   ├── build_metric_fingerprints.py
│   ├── evaluate_metric.py
│   ├── identify_song.py
│   ├── robustness_test.py
│   └── plot_results.py
│
├── fingerprints/
├── queries/
│
├── demo.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AudioVision
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Ubuntu / WSL:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

---

## Usage

### Step 1 — Generate a Mel-spectrogram

```bash
python preprocessing/audio_to_spectrogram.py
```

This produces:

```text
spectrograms/sample.npy
spectrograms/sample.png
```

---

### Step 2 — Build the dataset

```bash
python preprocessing/build_dataset.py
```

This converts the MP3 files and generates five-second Mel-spectrogram segments.

Expected:

```text
Found 7 MP3 files
...
Dataset generation complete!
```

Approximately **350 segments** are generated.

---

### Step 3 — Train the MLP baseline

```bash
python -m training.train_mlp
```

The trained model is saved as:

```text
models/mlp_baseline.pth
```

---

### Step 4 — Train the CNN baseline

```bash
python -m training.train_cnn
```

The best model is saved as:

```text
models/cnn_best.pth
```

---

### Step 5 — Train the metric-learning model

```bash
python -m training.train_metric
```

The metric-learning model is saved as:

```text
models/metric_cnn.pth
```

Expected training behavior:

```text
Epoch 01 Triplet Loss: 0.2635
...
Epoch 30 Triplet Loss: 0.0588
```

---

### Step 6 — Build the fingerprint database

```bash
python -m retrieval.build_metric_fingerprints
```

This generates the 128-D fingerprints for the reference dataset.

Expected fingerprint matrix:

```text
(350, 128)
```

---

### Step 7 — Evaluate fingerprint quality

```bash
python -m retrieval.evaluate_metric
```

This calculates:

* Same-song similarity
* Different-song similarity
* Median similarities
* Similarity separation

---

### Step 8 — Generate similarity graph

```bash
python -m retrieval.plot_results
```

The resulting graph shows the distribution of cosine similarities between:

* Same-song pairs
* Different-song pairs

---

### Step 9 — Identify a song

For example:

```bash
python -m retrieval.identify_song queries/sample1_test.wav
```

The system generates fingerprints for the query and ranks the reference songs.

Example:

```text
SONG RANKING

1. Sample1    0.9890
2. Sample5    0.9751
3. Sample6    0.9734
4. Sample4    0.8524
5. Sample2    0.8418
6. Sample3    0.7858
7. Sample7    0.7814

Predicted song: Sample1
```

---

## Technologies Used

* **Python**
* **PyTorch**
* **Librosa**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **SoundFile**
* **FFmpeg**
* **Ubuntu / WSL**

---

## Limitations

The current implementation is a proof of concept and has several limitations:

1. The dataset contains only seven songs.
2. The number of training segments is relatively small.
3. Baseline classification experiments use a random segment-level split.
4. The similarity distributions still overlap.
5. The model is sensitive to additive noise.
6. The system can identify songs only when they are represented in the reference database.
7. The current system is not intended to compete with production-scale systems such as commercial music-recognition services.

---

## Future Improvements

Possible extensions include:

* Larger and more diverse datasets
* Song-disjoint train/test evaluation
* Top-1, Top-3 and Top-5 retrieval accuracy
* Contrastive learning
* Supervised contrastive learning
* Data augmentation
* Noise augmentation during training
* Time stretching and pitch-shifting augmentation
* Real-time microphone input
* Web interface
* Mobile application
* Larger pretrained audio models
* Confidence/rejection threshold for unknown songs

---

## Project Results

The project demonstrates that audio can be transformed into a **visual representation**, encoded into a **learned 128-D fingerprint**, and used for **similarity-based song identification**.

The key experimental result is:

```text
Same-song similarity       : 0.9433
Different-song similarity  : 0.6963
Separation                 : 0.2470
```

And for the known test query:

```text
Predicted Song             : Sample1
Similarity                 : 0.9890
```

---

## Author

**Naresh S**

**Project:** AudioVision — Song From Its Visual Fingerprint

---

## License

This project is intended for **academic and educational purposes**.
