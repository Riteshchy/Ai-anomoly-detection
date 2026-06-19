import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import datetime
import json
import os
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(5, 4))
    
    # Custom Labels with Counts
    group_names = ['True Neg (Normal)', 'False Pos (False Alarm)', 
                   'False Neg (Missed)', 'True Pos (Caught)']
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='YlGnBu', cbar=True) 
    
    plt.title('Detection Accuracy Matrix')
    plt.ylabel('Actual Label (0:Normal, 1:Attack)')
    plt.xlabel('AI Prediction (0:Normal, 1:Attack)')
    plt.tight_layout()
    return fig

# --- NEW FUNCTION: Plot ROC Curve ---
def plot_roc_curve(y_true, y_scores):
    """
    Generates the Receiver Operating Characteristic (ROC) curve.
    y_true: Actual labels (0 or 1)
    y_scores: The MSE values (Anomaly Scores)
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def generate_pdf_report(metrics, counts, mse_values, predictions, threshold, save_path, top_threats, y_true):
    pdf = FPDF()
    pdf.add_page()
    
    # --- 1. Header ---
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", 'B', 24); pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 30, "NETWORK SECURITY AUDIT", ln=True, align='C')
    pdf.ln(15)

    # --- 2. Executive Summary ---
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 8, f"Total Analyzed: {counts['total']}", ln=True)
    
    pdf.set_text_color(39, 174, 96); pdf.cell(200, 8, f"Normal Traffic: {counts['normal']}", ln=True)
    pdf.set_text_color(192, 57, 43); pdf.cell(200, 8, f"Detected Anomalies: {counts['anomalies']}", ln=True)
    
    # --- Detailed Breakdown ---
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Detailed Detection Performance:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    if 'cm' in metrics:
        tn, fp, fn, tp = metrics['cm'].ravel()
        pdf.cell(200, 6, f"[-] True Negatives (Normal Correct): {tn}", ln=True)
        pdf.cell(200, 6, f"[!] False Positives (False Alarms): {fp}", ln=True)
        pdf.cell(200, 6, f"[x] False Negatives (Missed Attacks): {fn}", ln=True)
        pdf.cell(200, 6, f"[+] True Positives (Attacks Stopped): {tp}", ln=True)
    
    pdf.ln(5)
    pdf.cell(200, 8, f"Model Accuracy: {metrics['accuracy']:.2%}", ln=True)
    pdf.cell(200, 8, f"Precision: {metrics['precision']:.2%}", ln=True)
    pdf.cell(200, 8, f"Recall: {metrics['recall']:.2%}", ln=True)

    # --- 3. Visual Evidence (Updated) ---
    pdf.add_page()
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "2. Visual Evidence & Validation", ln=True)
    
    # A. Confusion Matrix
    fig_cm = plot_confusion_matrix(metrics['cm'])
    fig_cm.savefig("temp_cm.png", bbox_inches='tight', dpi=100)
    plt.close(fig_cm)
    
    # B. Reconstruction Error
    fig_err, ax = plt.subplots(figsize=(5, 4))
    ax.hist(mse_values[predictions == 0], bins=50, color='seagreen', alpha=0.6, label='Normal')
    ax.hist(mse_values[predictions == 1], bins=50, color='crimson', alpha=0.6, label='Threats')
    ax.axvline(threshold, color='black', linestyle='--', label='Threshold')
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    fig_err.savefig("temp_err.png", bbox_inches='tight', dpi=100)
    plt.close(fig_err)

    # C. ROC Curve (NEW)
    fig_roc = plot_roc_curve(y_true, mse_values)
    fig_roc.savefig("temp_roc.png", bbox_inches='tight', dpi=100)
    plt.close(fig_roc)

    # Place Images: Layout adjusted for 3 graphs
    # Row 1: Confusion Matrix & Error Hist
    pdf.image("temp_cm.png", x=10, y=40, w=90)
    pdf.image("temp_err.png", x=110, y=40, w=90)
    
    pdf.set_xy(10, 115)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(90, 10, "Fig 1: Detection Matrix", align='C')
    pdf.set_xy(110, 115)
    pdf.cell(90, 10, "Fig 2: Error Distribution", align='C')

    # Row 2: ROC Curve (Centered below)
    pdf.image("temp_roc.png", x=60, y=130, w=90)
    pdf.set_xy(60, 205)
    pdf.cell(90, 10, "Fig 3: ROC-AUC Performance Curve", align='C')

    # --- 4. Top Critical Threats Table ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, "3. Top Critical Threats Identified", ln=True)
    pdf.ln(5); pdf.set_font("Arial", '', 10)
    
    pdf.set_fill_color(231, 76, 60); pdf.set_text_color(255, 255, 255)
    pdf.cell(30, 10, "Packet ID", 1, 0, 'C', True)
    pdf.cell(40, 10, "MSE Score", 1, 0, 'C', True)
    pdf.cell(120, 10, "Anomaly Reason", 1, 1, 'C', True)
    
    pdf.set_text_color(0, 0, 0)
    for tid, score, reason in top_threats:
        pdf.cell(30, 10, str(tid), 1, 0, 'C')
        pdf.cell(40, 10, f"{score:.6f}", 1, 0, 'C')
        pdf.cell(120, 10, reason, 1, 1, 'L')

    

    # Cleanup
    try:
        pdf.output(save_path)
    except PermissionError:
        save_path = save_path.replace(".pdf", "_new.pdf")
        pdf.output(save_path)
        
    for f in ["temp_err.png", "temp_cm.png", "temp_roc.png"]:
        if os.path.exists(f): os.remove(f)

HISTORY_FILE = "scan_history.json"

def save_to_history(counts, timestamp):
    """Saves detection run to a JSON file."""
    entry = {
        "timestamp": timestamp,
        "total": int(counts['total']),
        "normal": int(counts['normal']),
        "anomalies": int(counts['anomalies'])
    }
    
    data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
        except:
            data = []
    
    data.append(entry)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_history():
    """Loads all history entries."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def clear_history_data():
    """Deletes the history file."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)