import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from training.dataset import PovertyDataset
from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

cfg = load_config()
checkpoint_path = cfg["evaluation"]["checkpoint"]
scatter_plot_path = cfg["evaluation"]["scatter_plot"]
predictions_csv = cfg["data"]["predictions_csv"]
batch_size = cfg["training"]["batch_size"]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

full_dataset = PovertyDataset(train=True, split=1.0)   # all rows for GeoJSON
full_loader = DataLoader(full_dataset, batch_size=batch_size)

dropout_p = cfg.get("model", {}).get("dropout_p", 0.0)
model = ResNetRegression(in_channels=3, dropout_p=dropout_p).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
logger.info("Loaded model from %s", checkpoint_path)

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in full_loader:
        images = images.to(device)
        preds = model(images).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

df_results = full_dataset.data.copy()
df_results["prediction"] = all_preds
os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
df_results.to_csv(predictions_csv, index=False)
logger.info("Saved predictions to %s", predictions_csv)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
mae = mean_absolute_error(all_labels, all_preds)
r2 = r2_score(all_labels, all_preds)
logger.info("RMSE=%.4f  MAE=%.4f  R²=%.4f", rmse, mae, r2)

os.makedirs(os.path.dirname(scatter_plot_path), exist_ok=True)
plt.figure(figsize=(6, 6))
plt.scatter(all_labels, all_preds, alpha=0.5)
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], "r--")
plt.xlabel("True Poverty Score")
plt.ylabel("Predicted Poverty Score")
plt.title("True vs Predicted Poverty Scores")
plt.grid(True)
plt.tight_layout()
plt.savefig(scatter_plot_path)
logger.info("Scatter plot saved to %s", scatter_plot_path)
