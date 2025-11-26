import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_model import HeartNN  # uses your existing model

# ---------------------------
# Reproducibility
# ---------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load processed dataset
# ---------------------------
df = pd.read_csv("processed_heart.csv")

# Feature columns
features = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

X = df[features].values.astype(np.float32)
y = df["HeartDisease"].values.astype(np.float32).reshape(-1, 1)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

import joblib
joblib.dump(scaler, "models/scaler.pkl")

# ---------------------------
# Tensors
# ---------------------------
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------
# Model
# ---------------------------
model = HeartNN(input_dim=len(features)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Training
# ---------------------------
epochs = 50
train_losses = []
val_losses = []
val_accuracy = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        v_loss = criterion(val_logits, y_val_t).item()
        val_losses.append(v_loss)

        preds = (torch.sigmoid(val_logits) >= 0.5).float()
        acc = (preds.eq(y_val_t).sum().item()) / y_val_t.size(0)
        val_accuracy.append(acc)

    print(f"Epoch {epoch}/{epochs} | Train Loss={avg_train_loss:.4f} | "
          f"Val Loss={v_loss:.4f} | Val Acc={acc:.4f}")

# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), "models/nn_model.pth")
print("Neural network saved to models/nn_model.pth")

# ---------------------------
# Plot & Save Loss Curve
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("plots/loss_curve.png")
plt.close()

# ---------------------------
# Plot & Save Accuracy Curve
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(val_accuracy, label="Validation Accuracy", color="green")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("plots/accuracy_curve.png")
plt.close()

print("Training graphs saved in 'plots' folder!")
