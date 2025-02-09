# Predictive Maintenance using Deep Learning on CMAPSS

## 📌 Project Overview
This project aims to predict the **Remaining Useful Life (RUL)** of aircraft engines using **CMAPSS** datasets. Various **Deep Learning models** are trained on all four CMAPSS datasets to evaluate their effectiveness in predictive maintenance.

## 📂 Datasets
The project uses **CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** datasets, which contain sensor readings from engines over time until failure. The four datasets used are:
- **FD001**
- **FD002**
- **FD003**
- **FD004**

Each dataset has different operating conditions and fault modes, making the problem more challenging.

## 🏗️ Models Implemented
The following **Deep Learning models** are trained and compared:
- **LSTM (Long Short-Term Memory)**
- **DCNN (Deep Convolutional Neural Network)**
- **Attention-based Model**
- **PINN (Physics-Informed Neural Network)**

Each model is evaluated based on its ability to accurately predict the **RUL** of engines.

## ⚙️ Installation
To set up the environment, use **Miniconda** and install the required dependencies from `requirements.txt`:

```bash
# Create and activate the environment
conda create --name rul_pred python=3.9 -y
conda activate rul_pred

# Install dependencies
pip install -r requirements.txt
 
