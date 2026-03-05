import os
import torch
from torch.utils.data import DataLoader
from src.models.motion_net_v4 import MotionNetV4
from src.data.dataset import MotionDataset

# ---------------- CONFIG ----------------
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = "checkpoints"
# ----------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = MotionDataset("outputs/clips/train")
    val_dataset = MotionDataset("outputs/clips/val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)
    model = MotionNetV4(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch} =====")

        # ---------------- TRAIN ----------------
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

            x = x.to(device).float()
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---------------- SAVE BEST MODEL ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": val_acc
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("✓ Saved new best model")

    # ---------------- SAVE FINAL MODEL ----------------
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": num_classes
    }, os.path.join(CHECKPOINT_DIR, "final_model.pth"))

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)

if __name__ == "__main__":
    main()