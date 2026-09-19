import torch.nn as nn


class AudioMLP(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 54, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        return self.network(x)
