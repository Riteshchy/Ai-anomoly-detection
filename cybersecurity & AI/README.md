# 🛡️ AI-Based Network Intrusion Detection System (IDS)

An academic implementation of a Deep Learning Anomaly Detection System using Autoencoders. This tool detects network intrusions (including zero-day attacks) by learning normal traffic patterns and flagging deviations, rather than relying on fixed signatures.

## 🚀 Key Features

* **Deep Autoencoder Model:** Unsupervised learning engine built with TensorFlow/Keras.
* **Interactive Dashboard:** A user-friendly GUI (Tkinter) for training, monitoring, and analysis.
* **Real-time Visualization:** Dynamic graphs for Training Loss, Confusion Matrices, and Reconstruction Error.
* **Forensic Reporting:** Automated PDF generation with ROC curves and statistical evidence.
* **Explainable AI (XAI):** Provides specific reasons for every flagged anomaly (e.g., "Deviation in Source Bytes").

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* PIP (Python Package Manager)

### Step 1: Clone or Extract
Ensure all project files (`main.py`, `gui.py`, `model.py`, `utils.py`, `requirements.txt`) are in the same folder.

### Step 2: Install Dependencies
Open your terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt