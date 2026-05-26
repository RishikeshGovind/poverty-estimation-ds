import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math

from training.dataset import PovertyDataset
from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

cfg = load_config()
tcfg = cfg["training"]

batch_size = tcfg["batch_size"]
epochs = tcfg["epochs"]
lr = tcfg["lr"]
checkpoint_path = tcfg["checkpoint"]
loss_plot_path = tcfg["loss_plot"]

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info("Using device: %s", device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

train_dataset = PovertyDataset(train=True, transform=train_transform)
val_dataset = PovertyDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = ResNetRegression(in_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
criterion = nn.MSELoss()

best_val_loss = float("inf")
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
    epoch_val_loss = val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)

    scheduler.step(epoch_val_loss)
    logger.info(
        "Epoch %d: train_loss=%.4f val_loss=%.4f rmse=%.4f",
        epoch + 1, epoch_train_loss, epoch_val_loss, math.sqrt(epoch_val_loss),
    )

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Saved new best model to %s", checkpoint_path)

plt.figure()
plt.plot(train_losses, label="Train MSE")
plt.plot(val_losses, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Poverty Estimation Training Progress")
plt.savefig(loss_plot_path)
logger.info("Training history saved to %s", loss_plot_path)
