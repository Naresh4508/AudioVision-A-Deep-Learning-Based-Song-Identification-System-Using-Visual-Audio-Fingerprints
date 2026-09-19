import os
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioVisionDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []
        self.class_names = []

        song_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        self.class_names = song_dirs

        class_to_index = {
            name: index
            for index, name in enumerate(self.class_names)
        }

        for song_name in self.class_names:

            song_dir = os.path.join(
                root_dir,
                song_name
            )

            for filename in sorted(
                os.listdir(song_dir)
            ):

                if filename.endswith(".npy"):

                    path = os.path.join(
                        song_dir,
                        filename
                    )

                    label = class_to_index[song_name]

                    self.samples.append(
                        (path, label)
                    )

        print("Classes:", self.class_names)
        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        spectrogram = np.load(path)

        # Normalize from approximately [-80, 0] to [0, 1]
        spectrogram = (
            spectrogram + 80.0
        ) / 80.0

        spectrogram = np.clip(
            spectrogram,
            0.0,
            1.0
        )

        # Downsample for the MLP baseline
        spectrogram = spectrogram[::4, ::4]

        tensor = torch.tensor(
            spectrogram,
            dtype=torch.float32
        )

        return tensor, label
