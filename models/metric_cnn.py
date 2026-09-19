import torch.nn as nn
import torch.nn.functional as F


class MetricCNN(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, embedding_dim),

            nn.ReLU()
        )

    def forward(self, x):

        x = self.features(x)

        x = self.embedding(x)

        # L2-normalized fingerprint
        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x
