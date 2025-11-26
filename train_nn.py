import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from nn_model import HeartNN

df = pd.read_csv("processed_heart.csv")

X = df.drop("HeartDisease", axis=1).values
y = df["HeartDisease"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

model = HeartNN(input_dim=X.shape[1])
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/nn_model.pth")
print("Saved nn_model.pth")
