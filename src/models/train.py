import os
import torch
from torch.utils.data import DataLoader

from src.models.motion_net_v3 import MotionNetLite
from src.data.dataset import MotionDataset


# ---------------- CONFIG ----------------
EPOCHS = 10
BATCH_SIZE = 2        # 🔥 small for CPU
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = "checkpoints"
torch.set_num_threads(8)
# ----------------------------------------


def main():

    device = torch.device("cpu")
    print("Using device:", device)

    train_dataset = MotionDataset("outputs/clips/train")
    val_dataset = MotionDataset("outputs/clips/val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    num_classes = len(train_dataset.classes)

    model = MotionNetLite(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):

        print(f"\n===== Epoch {epoch} =====")

        model.train()
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

            x = x.to(device).float()
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device)

                pred = model(x)
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    print("Training complete.")


if __name__ == "__main__":
    main()