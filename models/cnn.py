import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        x = self.features(x)

        embedding = self.embedding(x)

        output = self.classifier(
            embedding
        )

        return output


    def get_embedding(self, x):

        x = self.features(x)

        embedding = self.embedding(x)

        # Normalize fingerprint
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding
