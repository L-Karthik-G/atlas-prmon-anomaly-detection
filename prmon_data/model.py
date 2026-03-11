import json
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

normal_files = sorted(glob.glob("prmon_data/normal_run*.json"))

normal_samples = []

for file in normal_files:
    with open(file) as f:
        data = json.load(f)

    for d in data:
        normal_samples.append([d["rss_mb"], d["vms_mb"], d["nprocs"]])

X_train = np.array(normal_samples)

print("Normal data shape:", X_train.shape)

anomaly_files = [
    "prmon_data/anomaly_highmem.json",
    "prmon_data/anomaly_highprocs.json",
    "prmon_data/anomaly_combined.json"
]

anomaly_samples = []

for file in anomaly_files:
    with open(file) as f:
        data = json.load(f)

    for d in data:
        anomaly_samples.append([d["rss_mb"], d["vms_mb"], d["nprocs"]])

X_anomaly = np.array(anomaly_samples)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_anomaly = scaler.transform(X_anomaly)

X_train = X_train[3:]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_anomaly_tensor = torch.tensor(X_anomaly, dtype=torch.float32)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=3):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8,1)

        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16,input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


Model = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(Model.parameters(), lr=0.001 , weight_decay=1e-5)

num_epochs = 100

for epoch in range(num_epochs):
    Model.train()
    optimizer.zero_grad()

    outputs = Model(X_train_tensor)

    loss = criterion(outputs, X_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


Model.eval()

with torch.no_grad():
    recon_train = Model(X_train_tensor)
    recon_anomaly = Model(X_anomaly_tensor)

recon_error_train = torch.mean((X_train_tensor - recon_train) ** 2, dim=1)
recon_error_anomaly = torch.mean((X_anomaly_tensor - recon_anomaly) ** 2, dim=1)

threshold = torch.quantile(recon_error_train, 0.99)

X_test = X_train.copy()
y_test = np.zeros(len(X_test))
num_bursts = np.random.randint(2, 12)

for _ in range(num_bursts):
    burst_size = np.random.randint(2, 15)
    start_idx = np.random.randint(0, len(X_test) - burst_size)
    
    anomaly_batch_indices = np.random.choice(len(X_anomaly), burst_size)
    
    X_test[start_idx : start_idx + burst_size] = X_anomaly[anomaly_batch_indices]
    y_test[start_idx : start_idx + burst_size] = 1

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

Model.eval()

with torch.no_grad():
    recon_test = Model(X_test_tensor)

recon_error_test = torch.mean((X_test_tensor - recon_test) ** 2, dim=1)
predicted_anomalies = recon_error_test > threshold
predicted_np = predicted_anomalies.numpy()

true = y_test.astype(int)

total_samples = len(true)
total_anomalies = np.sum(true)
total_normals = total_samples - total_anomalies

false_positive_idx = np.where((predicted_np == 1) & (y_test == 0))[0]
pred = predicted_anomalies.numpy().astype(int)


an_correct = np.sum((pred == 1) & (true == 1))
norm_correct = np.sum((pred == 0) & (true == 0))
false_pos = np.sum((pred == 1) & (true == 0))
an_missed = np.sum((pred == 0) & (true == 1))

accuracy = (an_correct+norm_correct)/(len(true))
precision = (an_correct)/(an_correct+false_pos)
recal = (an_correct)/(an_missed+an_correct)

print("Total samples:", total_samples)
print("Total normal samples:", total_normals)
print("Total anomaly samples:", total_anomalies)
print("Detected anomalies:", np.sum(pred))
print(f"Threshold: {threshold:.6f}")
print("False Positives (FP) - normal samples wrongly flagged:", false_pos)
print("Indices of the False Positive valuse: ",false_positive_idx)
print(f"Accuracy of model: {accuracy:.6f}")
print(f"Prescision of model: {precision:.6f}")
print(f"Recal of the model: {recal:.6f}")

time_axis = np.arange(len(X_test))
rss_values = X_test[:,0]
pred = predicted_anomalies.numpy()
anomaly_indices = np.where(pred == 1)[0]
plt.figure(figsize=(10,5))
plt.plot(time_axis, rss_values, label="RSS memory")

plt.scatter(
    anomaly_indices,
    rss_values[anomaly_indices],
    color="red",
    label="Detected Anomalies"
)
plt.xlabel("Time Step")
plt.ylabel("Normalized RSS")
plt.title("Anomaly Detection on prmon Time Series")
plt.legend()
plt.show()

