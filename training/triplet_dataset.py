import os
import random

import numpy as np
import torch

from torch.utils.data import Dataset


class TripletAudioDataset(Dataset):

    def __init__(self, root_dir, triplets_per_song=100):

        self.samples_by_song = {}

        self.song_names = []

        for song_name in sorted(os.listdir(root_dir)):

            song_dir = os.path.join(
                root_dir,
                song_name
            )

            if not os.path.isdir(song_dir):
                continue

            files = sorted(
                os.path.join(song_dir, f)
                for f in os.listdir(song_dir)
                if f.endswith(".npy")
            )

            if len(files) >= 2:

                self.samples_by_song[song_name] = files
                self.song_names.append(song_name)

        self.triplets = []

        for song_name in self.song_names:

            positive_pool = self.samples_by_song[song_name]

            negative_songs = [
                s for s in self.song_names
                if s != song_name
            ]

            for _ in range(triplets_per_song):

                anchor, positive = random.sample(
                    positive_pool,
                    2
                )

                negative_song = random.choice(
                    negative_songs
                )

                negative = random.choice(
                    self.samples_by_song[negative_song]
                )

                self.triplets.append(
                    (anchor, positive, negative)
                )

        print("Songs:", self.song_names)
        print("Triplets:", len(self.triplets))

    def __len__(self):

        return len(self.triplets)

    def load_spectrogram(self, path):

        spectrogram = np.load(path)

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

        return tensor.unsqueeze(0)

    def __getitem__(self, index):

        anchor, positive, negative = self.triplets[index]

        return (
            self.load_spectrogram(anchor),
            self.load_spectrogram(positive),
            self.load_spectrogram(negative)
        )
