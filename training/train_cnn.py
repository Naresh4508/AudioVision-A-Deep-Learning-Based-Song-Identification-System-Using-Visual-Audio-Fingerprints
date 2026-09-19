import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from training.dataset import AudioVisionDataset
from models.cnn import AudioCNN


DATASET_PATH = "spectrograms"

BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = AudioVisionDataset(DATASET_PATH)

num_classes = len(dataset.class_names)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = AudioCNN(
    num_classes=num_classes
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        # Add channel dimension
        inputs = inputs.unsqueeze(1)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.unsqueeze(1)

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_accuracy = (
        100 * val_correct / val_total
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/cnn_best.pth"
        )


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )


print("\nCNN training complete.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved to models/cnn_best.pth"
)
