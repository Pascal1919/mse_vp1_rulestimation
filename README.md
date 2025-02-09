# 🔧 Predictive Maintenance using Deep Learning on CMAPSS

## 📌 Project Overview
This project aims to predict the **Remaining Useful Life (RUL)** of aircraft engines using a **combined dataset** from all four **CMAPSS** datasets. By leveraging multiple **Deep Learning architectures**, we aim to enhance the accuracy and robustness of RUL predictions.

## 📂 Dataset
The dataset used in this project is a **combination** of the four CMAPSS sub-datasets:
- **FD001**
- **FD002**
- **FD003**
- **FD004**

By merging these datasets, we create a more diverse and challenging dataset that includes **multiple operational conditions and fault modes**, improving the generalization capability of the models.

## 🏗️ Models Implemented
We train and evaluate multiple **Deep Learning models** for RUL prediction:
- **LSTM (Long Short-Term Memory)**
- **DCNN (Deep Convolutional Neural Network)**
- **Attention-based Model**
- **PINN (Physics-Informed Neural Network)**

These models are designed to handle **time-series sensor data**, capture long-term dependencies, and incorporate domain knowledge into the learning process.

## ⚙️ Installation
To set up the environment, use **Miniconda** and install the required dependencies:

```bash
# Create and activate the environment
conda create --name rul_pred python=3.9 -y
conda activate rul_pred

# Install dependencies
pip install -r requirements.txt
