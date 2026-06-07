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

split = cfg["training"]["split"]   # default 0.8

# Held-out val split — the only split used for reported metrics.
val_dataset = PovertyDataset(train=False, split=split)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

dropout_p = cfg.get("model", {}).get("dropout_p", 0.0)
model = ResNetRegression(in_channels=3, dropout_p=dropout_p).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
logger.info("Loaded model from %s", checkpoint_path)
logger.info("Evaluating on held-out val split (%d%% of data, %d samples)",
            int((1 - split) * 100), len(val_dataset))

val_preds, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        preds = model(images).cpu().numpy()
        val_preds.extend(preds)
        val_labels.extend(labels.numpy())

# Save val predictions for inspection
df_results = val_dataset.data.copy()
df_results["prediction"] = val_preds
os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
df_results.to_csv(predictions_csv, index=False)
logger.info("Saved val predictions to %s", predictions_csv)

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)

rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
mae = mean_absolute_error(val_labels, val_preds)
r2 = r2_score(val_labels, val_preds)
logger.info("Held-out val  RMSE=%.4f  MAE=%.4f  R²=%.4f  n=%d", rmse, mae, r2, len(val_labels))

os.makedirs(os.path.dirname(scatter_plot_path), exist_ok=True)
plt.figure(figsize=(6, 6))
plt.scatter(val_labels, val_preds, alpha=0.5)
plt.plot([val_labels.min(), val_labels.max()], [val_labels.min(), val_labels.max()], "r--")
plt.xlabel("True Wealth Index (held-out val)")
plt.ylabel("Predicted Wealth Index")
plt.title(f"Held-out val  R²={r2:.3f}  RMSE={rmse:.3f}  n={len(val_labels)}")
plt.grid(True)
plt.tight_layout()
plt.savefig(scatter_plot_path)
logger.info("Scatter plot saved to %s", scatter_plot_path)
