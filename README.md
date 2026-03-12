# Automated Software Performance Monitoring Warm-Up
---

## Overview

exploring process resource monitoring and anomaly detection using **prmon**. The goal is to generate time-series process metrics, inject artificial anomalies, and detect them using a ML model.

The workflow includes:

* Environment setup with WSL2 on Windows
* Data generation with prmon's mem-burner test
* Anomaly injection into the dataset
* Autoencoder-based anomaly detection
* Visualization of results

---

## Setup

1. Install **WSL2** on Windows and set up a Linux distribution.
2. Clone and install prmon from source:

```bash
git clone https://github.com/HSF/prmon.git
cd prmon
mkdir build && cd build
cmake ..
make
```

3. Install Python 3.x and required libraries:

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

---

## Data Generation

1. Run prmon with mem-burner tests to collect baseline process metrics:

   * **RSS (Resident Set Size)**
   * **VMS (Virtual Memory Size)**
   * **Nprocs (Number of active processes)**
2. Generate multiple normal runs to capture baseline behavior.
3. Introduce anomalies by increasing memory usage and process counts in random bursts.
4. Combine normal and anomalous runs into a single dataset for training and testing.

Example command:

```bash
./prmon --pid $(./burner --mem 500M &) --json output.json --timeout 60
```

---

## Model

An **autoencoder** is used for anomaly detection for this scenario:

* Symmetric encoder–decoder architecture
* ReLU layers for nonlinearity
* Input projected from 3 → 16 dimensions to improve learning
* Trained only on normal data

Anomalies are flagged based on reconstruction error exceeding **mean + 3.8σ** of the training reconstruction errors.

---

## Anomaly Detection Workflow

1. Load and preprocess JSON data into PyTorch tensors.
2. Train the autoencoder on normal data.
3. Apply the model to the mixed dataset containing injected anomalies.
4. Compute reconstruction errors and flag anomalies.
5. Visualize results (plots show anomalies over time-series metrics).

---

## Outputs and Plots

* Reconstruction error per time step
* Indices of detected anomalies
* Time-series plots with anomalies highlighted

<img width="1917" height="1125" alt="image" src="https://github.com/user-attachments/assets/981d84d3-ff70-4a4a-9230-f1526feeb6af" />


---

## Notes

* Each run produces slightly different simulated data, allowing evaluation of the model's robustness.
* Threshold selection is based on Gaussian estimates to balance sensitivity and false positives.
* [My Portfolio](https://karthikgollapudi.vercel.app/projects)
* [Reference for Autoencoders](https://karthikgollapudi.vercel.app/blog/)


