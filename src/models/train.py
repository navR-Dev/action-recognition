import torch
from torch.utils.data import DataLoader

from src.models.baseline_model import MotionNet
from src.data.dataset import MotionDataset

dataset = MotionDataset("outputs/clips")

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MotionNet(num_classes=len(dataset.classes))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):

    total = 0
    correct = 0

    for x,y in loader:

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1)==y).sum().item()
        total += y.size(0)

    print("epoch", epoch, "acc", correct/total)