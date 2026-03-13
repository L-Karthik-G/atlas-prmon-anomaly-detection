"""
The total dataset is small due to WSL crashing constraints during data collection.
To mitigate overfitting, the data is split chronologically into training and test sets with a 70:30 split.
"""

import json
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
import matplotlib.pyplot as plt



# Used AI assistance to create data in WSL using prmon
normal_files  = sorted(glob.glob("prmon_data/normal_run*.json"))
anomaly_files = sorted(glob.glob("prmon_data/anomaly_*.json"))

normal_samples = []
for file in normal_files:
    with open(file) as f:
        # Used AI assistance to load JSON file
        data = json.load(f)
    for d in data:
        normal_samples.append([d["rss_mb"], d["vms_mb"], d["nprocs"]])

X_train_full = np.array(normal_samples)
print("Normal data shape:", X_train_full.shape)


anomaly_samples = []
for file in anomaly_files:
    with open(file) as f:
        data = json.load(f)
    for d in data:
        anomaly_samples.append([d["rss_mb"], d["vms_mb"], d["nprocs"]])

X_anomaly = np.array(anomaly_samples)

print(f"Total normal samples:  {len(X_train_full)}")
print(f"Total anomaly samples: {len(X_anomaly)}")


# Fit scaler on normal data only, then apply to anomalies
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_anomaly    = scaler.transform(X_anomaly)


# Split chronologically: 70% train, 30% held-out test base
split_idx  = int(len(X_train_full) * 0.7)
X_train    = X_train_full[:split_idx]
X_test_base = X_train_full[split_idx:]

X_train_tensor   = torch.tensor(X_train,    dtype=torch.float32)
X_anomaly_tensor = torch.tensor(X_anomaly,  dtype=torch.float32)

print(f"Train Tensor Shape:  {X_train_tensor.shape}")
print(f"Anomaly Pool Shape:  {X_anomaly.shape}")


class Autoencoder(nn.Module):
    def __init__(self, input_dim=3):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z      = self.encoder(x)
        x_recon = self.decoder(z)
        return

Model     = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(Model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 100
for epoch in range(num_epochs):
    Model.train()
    optimizer.zero_grad()

    outputs = Model(X_train_tensor)
    loss    = criterion(outputs, X_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

Model.eval()
with torch.no_grad():
    recon_train = Model(X_train_tensor)
    recon_val = Model(torch.tensor(X_test_base, dtype=torch.float32))
val_loss = torch.mean((torch.tensor(X_test_base, dtype=torch.float32) - recon_val) ** 2)
print(f"Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f}")

recon_error_train = torch.mean((X_train_tensor - recon_train) ** 2, dim=1)
threshold = np.mean(recon_error_train.numpy()) + 3.8 * np.std(recon_error_train.numpy())

# Randomly insert bursts of real anomaly data into the held-out test window
X_test = X_test_base.copy()
y_test = np.zeros(len(X_test))

num_bursts = np.random.randint(2, 15)
for _ in range(num_bursts):
    burst_size  = np.random.randint(2, 20)
    start_idx   = np.random.randint(0, len(X_test) - burst_size)
    anomaly_batch_indices = np.random.choice(len(X_anomaly), burst_size)

    X_test[start_idx : start_idx + burst_size] = X_anomaly[anomaly_batch_indices]
    y_test[start_idx : start_idx + burst_size] = 1

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

Model.eval()
with torch.no_grad():
    recon_test = Model(X_test_tensor)

recon_error_test   = torch.mean((X_test_tensor - recon_test) ** 2, dim=1)
predicted_anomalies = recon_error_test > threshold
pred = predicted_anomalies.numpy().astype(int)
true = y_test.astype(int)

total_samples  = len(true)
total_anomalies = np.sum(true)
total_normals  = total_samples - total_anomalies

an_correct  = np.sum((pred == 1) & (true == 1))
norm_correct = np.sum((pred == 0) & (true == 0))
false_pos   = np.sum((pred == 1) & (true == 0))
an_missed   = np.sum((pred == 0) & (true == 1))

false_positive_idx = np.where((pred == 1) & (true == 0))[0]

accuracy  = (an_correct + norm_correct) / len(true)

# Used AI assistance for precision and recall formulae
precision = an_correct / (an_correct + false_pos)  if (an_correct + false_pos)  > 0 else 0
recall    = an_correct / (an_correct + an_missed)   if (an_correct + an_missed)  > 0 else 0

print(f"\nTotal samples:          {total_samples}")
print(f"Total normal samples:   {total_normals}")
print(f"Total anomaly samples:  {total_anomalies}")
print(f"Detected anomalies:     {np.sum(pred)}")
print(f"Threshold:              {threshold:.6f}")
print(f"False Positives:        {false_pos}")
print(f"FP indices:             {false_positive_idx}")
print(f"Accuracy:               {accuracy:.6f}")
print(f"Precision:              {precision:.6f}")
print(f"Recall:                 {recall:.6f}")


time_axis       = np.arange(len(X_test))
rss_values      = X_test[:, 0]
vms_values      = X_test[:, 1]
nprocs_values   = X_test[:, 2]
anomaly_indices = np.where(pred == 1)[0]

#Used AI assistance to split into subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(time_axis, rss_values, label="RSS Memory")
axes[0].scatter(anomaly_indices, rss_values[anomaly_indices], color="red", label="Detected Anomalies")
axes[0].set_ylabel("Normalized RSS")
axes[0].set_title("Anomaly Detection on prmon Time Series (RSS)")
axes[0].legend()

axes[1].plot(time_axis, vms_values, label="VMS Memory")
axes[1].scatter(anomaly_indices, vms_values[anomaly_indices], color="red", label="Detected Anomalies")
axes[1].set_ylabel("Normalized VMS")
axes[1].set_title("Anomaly Detection on prmon Time Series (VMS)")
axes[1].legend()

axes[2].plot(time_axis, nprocs_values, label="Nprocs")
axes[2].scatter(anomaly_indices, nprocs_values[anomaly_indices], color="red", label="Detected Anomalies")
axes[2].set_ylabel("Nprocs")
axes[2].set_title("Anomaly Detection on prmon Time Series (Nprocs)")
axes[2].set_xlabel("Time Step")
axes[2].legend()

plt.tight_layout()
plt.show()
