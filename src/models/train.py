import torch
from torch.utils.data import DataLoader

from src.models.motion_model import MotionNetV2
from src.data.dataset import MotionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MotionDataset("outputs/clips/train")
val_dataset = MotionDataset("outputs/clips/val")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_classes = len(train_dataset.classes)

model = MotionNetV2(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):

    # ---- TRAIN ----
    model.train()
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_correct += (pred.argmax(1) == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            val_correct += (pred.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")