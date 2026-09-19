import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.metric_cnn import MetricCNN
from training.triplet_dataset import TripletAudioDataset


DATASET_PATH = "spectrograms"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

MARGIN = 0.3

DEVICE = torch.device("cpu")


print("Using device:", DEVICE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = TripletAudioDataset(
    DATASET_PATH,
    triplets_per_song=100
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = MetricCNN(
    embedding_dim=128
).to(DEVICE)


criterion = nn.TripletMarginLoss(
    margin=MARGIN,
    p=2
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for anchor, positive, negative in loader:

        anchor = anchor.to(DEVICE)
        positive = positive.to(DEVICE)
        negative = negative.to(DEVICE)

        optimizer.zero_grad()

        anchor_embedding = model(anchor)

        positive_embedding = model(positive)

        negative_embedding = model(negative)

        loss = criterion(
            anchor_embedding,
            positive_embedding,
            negative_embedding
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Triplet Loss: {average_loss:.4f}"
    )


torch.save(
    model.state_dict(),
    "models/metric_cnn.pth"
)


print()
print("Metric learning complete.")
print("Model saved to models/metric_cnn.pth")
