import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import os
import datetime
import sys

from model import AnomalyDetector
import utils

class CyberSecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ AI-IDS: Integrated Detection Dashboard")
        self.root.geometry("1200x850")
        
        # --- CRITICAL FIX: Bind the Safe Exit Function ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize Core Systems
        self.detector = AnomalyDetector()
        self.latest_metrics = None
        self.latest_preds = None
        self.mse_values = None 
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=70)
        header.pack(fill=tk.X)
        tk.Label(header, text="🛡️ AI ANOMALY DETECTION SYSTEM", font=("Segoe UI", 20, "bold"), bg="#2c3e50", fg="white").pack(pady=15)
        
        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_monitor = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook) # <--- NEW
        
        self.notebook.add(self.tab_train, text="  ⚙️ MODEL TRAINING  ")
        self.notebook.add(self.tab_monitor, text="  📡 LIVE MONITOR  ")
        self.notebook.add(self.tab_analysis, text="  📊 ANALYSIS & EXPORT  ")
        self.notebook.add(self.tab_history, text="  📜 HISTORY  ") # <--- NEW
        
        self._build_train_tab()
        self._build_monitor_tab()
        self._build_analysis_tab()
        self._build_history_tab() # <--- NEW

    def _build_train_tab(self):
        panel = ttk.Frame(self.tab_train, padding=20)
        panel.pack(fill=tk.BOTH, expand=True)
        
        ctrl_frame = ttk.LabelFrame(panel, text="Training Workflow", padding=15)
        ctrl_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(ctrl_frame, text="📁 1. Load Dataset", command=self.load_data).pack(side=tk.LEFT, padx=10)
        self.btn_train = ttk.Button(ctrl_frame, text="🚀 2. Start Training", state=tk.DISABLED, command=self.train_model)
        self.btn_train.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(ctrl_frame, text="🔄 Reset System", command=self.reset_system).pack(side=tk.RIGHT, padx=10)
        
        self.lbl_status = tk.Label(ctrl_frame, text="Status: Waiting for data...", fg="#7f8c8d", font=("Segoe UI", 10, "italic"))
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        self.train_plot_container = ttk.LabelFrame(panel, text="Training Convergence (Loss)")
        self.train_plot_container.pack(fill=tk.BOTH, expand=True, pady=10)
        self.train_plot_frame = tk.Frame(self.train_plot_container, bg="white")
        self.train_plot_frame.pack(fill=tk.BOTH, expand=True)

    def _build_monitor_tab(self):
        main_frame = ttk.Frame(self.tab_monitor, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        sidebar = ttk.Frame(main_frame)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.btn_detect = ttk.Button(sidebar, text="🔍 Run Detection", command=self.run_detection)
        self.btn_detect.pack(fill=tk.X, pady=5)
        
        self.sub_notebook = ttk.Notebook(main_frame)
        self.sub_notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.tab_all_threats = ttk.Frame(self.sub_notebook)
        self.tab_all_normal = ttk.Frame(self.sub_notebook)
        
        self.sub_notebook.add(self.tab_all_threats, text=" 🚨 Flagged Threats ")
        self.sub_notebook.add(self.tab_all_normal, text=" ✅ Normal Traffic ")

        self.tree_threats = self._create_treeview(self.tab_all_threats)
        self.tree_normal = self._create_treeview(self.tab_all_normal)

    def _build_analysis_tab(self):
        panel = ttk.Frame(self.tab_analysis, padding=20)
        panel.pack(fill=tk.BOTH, expand=True)
        
        summary_frame = ttk.LabelFrame(panel, text="📊 Live Statistics", padding=10)
        summary_frame.pack(fill=tk.X, pady=5)
        self.lbl_counts = tk.Label(summary_frame, text="Total: 0 | Normal: 0 | Anomalies: 0", 
                                  font=("Segoe UI", 12, "bold"), fg="#2c3e50")
        self.lbl_counts.pack()

        self.plot_container = ttk.Frame(panel)
        self.plot_container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.cm_frame = ttk.LabelFrame(self.plot_container, text="Confusion Matrix")
        self.cm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.error_frame = ttk.LabelFrame(self.plot_container, text="Reconstruction Error Distribution")
        self.error_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Button(panel, text="📄 Download PDF Security Report (Academic)", command=self.download_report).pack(pady=10)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.lbl_status.config(text=f"Loading: {os.path.basename(path)}... (Please Wait)", fg="#e67e22")
            self.root.update()
            threading.Thread(target=self._load_data_worker, args=(path,), daemon=True).start()

    def _load_data_worker(self, path):
        try:
            df = self.detector.load_data(path)
            self.detector.preprocess(df)
            self.root.after(0, lambda: self._load_data_success(path))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load: {e}"))
            self.root.after(0, lambda: self.lbl_status.config(text="Error loading data", fg="red"))

    def _load_data_success(self, path):
        self.lbl_status.config(text=f"Loaded: {os.path.basename(path)}", fg="#27ae60")
        self.btn_train.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Dataset loaded successfully!")

    def train_model(self):
        self.lbl_status.config(text="Training Model... (This may take a moment)", fg="#e67e22")
        self.btn_train.config(state=tk.DISABLED)
        threading.Thread(target=self._train_model_worker, daemon=True).start()

    def _train_model_worker(self):
        try:
            history = self.detector.build_and_train()
            self.root.after(0, lambda: self._train_model_success(history))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.root.after(0, lambda: self.lbl_status.config(text="Training Failed", fg="red"))
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))

    def _train_model_success(self, history):
        self.lbl_status.config(text="Model Trained Successfully", fg="#2980b9")
        self._plot_loss(history)
        messagebox.showinfo("Training", "AI Model Training Complete!")

    def reset_system(self):
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to clear all data and reset the system?")
        if not confirm:
            return

        plt.close('all') 
        
        self.detector = AnomalyDetector()
        self.latest_metrics = None
        self.latest_preds = None
        self.mse_values = None 

        self.lbl_status.config(text="Status: Waiting for data...", fg="#7f8c8d")
        self.lbl_counts.config(text="Total: 0 | Normal: 0 | Anomalies: 0")
        self.btn_train.config(state=tk.DISABLED)
        
        for w in self.train_plot_frame.winfo_children(): w.destroy()
        for w in self.cm_frame.winfo_children(): w.destroy()
        for w in self.error_frame.winfo_children(): w.destroy()

        self.tree_threats.delete(*self.tree_threats.get_children())
        self.tree_normal.delete(*self.tree_normal.get_children())
        
        self.log("SYSTEM", "System Reset Complete.")

    def _create_treeview(self, parent):
        cols = ("ID", "Error", "Status", "Reason")
        tree = ttk.Treeview(parent, columns=cols, show='headings')
        for c in cols: tree.heading(c, text=c)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    def run_detection(self):
        if self.detector.model is None:
            messagebox.showwarning("Error", "Please train the model first!")
            return
        
        self.btn_detect.config(state=tk.DISABLED)
        self.lbl_status.config(text="Running Detection...", fg="#e67e22")
        thread = threading.Thread(target=self._detection_worker, daemon=True)
        thread.start()

    def _detection_worker(self):
        try:
            self.log("SYSTEM", "AI is analyzing traffic...")
            preds, mse = self.detector.detect()
            self.latest_preds = preds
            self.mse_values = mse
            self.latest_metrics = self.detector.get_metrics(preds)
            counts = self.detector.get_summary_counts(preds)
            self.root.after(0, lambda: self._update_ui_after_detection(counts, mse))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
            self.root.after(0, lambda: self.btn_detect.config(state=tk.NORMAL))

    def log(self, category, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{category}] {message}")

    def _update_ui_after_detection(self, counts, mse_values):
        self.lbl_counts.config(text=f"Total: {counts['total']} | ✅ Normal: {counts['normal']} | 🚨 Anomalies: {counts['anomalies']}")
        self.lbl_status.config(text="Detection Complete", fg="#2980b9")
        
        self.tree_threats.delete(*self.tree_threats.get_children())
        self.tree_normal.delete(*self.tree_normal.get_children())
        
        limit = 1000 
        for i, (pred, loss) in enumerate(zip(self.latest_preds, mse_values)):
            if i >= limit: break 
            
            reason = self.detector.get_anomaly_reason(i) if pred == 1 else "Pattern Verified"
            row = (i, f"{loss:.6f}", "CRITICAL" if pred == 1 else "Normal", reason)
            
            if pred == 1:
                self.tree_threats.insert("", tk.END, values=row)
            else:
                self.tree_normal.insert("", tk.END, values=row)

        self._draw_detailed_analysis(mse_values)
        self.btn_detect.config(state=tk.NORMAL)
        # --- NEW: SAVE HISTORY AUTOMATICALLY ---
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        utils.save_to_history(counts, now)
        self.refresh_history_view()
        # ---------------------------------------

        self.log("SYSTEM", "UI Refresh Complete.")
        self.log("SYSTEM", "UI Refresh Complete.")

    def download_report(self):
        if self.mse_values is None:
            messagebox.showwarning("Warning", "Run detection first!")
            return
            
        try:
            report_dir = "reports"
            if not os.path.exists(report_dir): os.makedirs(report_dir)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(report_dir, f"Security_Report_{timestamp}.pdf")

            anomaly_indices = np.where(self.latest_preds == 1)[0]
            top_threats = []
            if len(anomaly_indices) > 0:
                top_indices = anomaly_indices[np.argsort(self.mse_values[anomaly_indices])[-5:][::-1]]
                top_threats = [(idx, self.mse_values[idx], self.detector.get_anomaly_reason(idx)) for idx in top_indices]

            utils.generate_pdf_report(
                self.latest_metrics, 
                self.detector.get_summary_counts(self.latest_preds),
                self.mse_values, 
                self.latest_preds,
                self.detector.threshold,
                save_path,
                top_threats,
                self.detector.y_test 
            )

            messagebox.showinfo("Success", f"Report saved: {save_path}")
            os.startfile(report_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Report failed: {e}")
            print(e)

    def _plot_loss(self, history):
        for w in self.train_plot_frame.winfo_children(): w.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history['loss'], color='#2980b9', label='Training Loss', linewidth=2)
        
        if 'val_loss' in history:
            ax.plot(history['val_loss'], color='#e67e22', linestyle='--', label='Validation Loss', linewidth=2)
        
        final_loss = history['loss'][-1]
        ax.annotate(f'Final MSE: {final_loss:.5f}', 
                    xy=(len(history['loss'])-1, final_loss),
                    xytext=(-40, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='black'))

        ax.set_title("Model Convergence Analysis (Train vs Val)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Reconstruction Error (MSE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.train_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_detailed_analysis(self, mse_values):
        for w in self.cm_frame.winfo_children(): w.destroy()
        for w in self.error_frame.winfo_children(): w.destroy()

        if 'cm' in self.latest_metrics:
            fig_cm = utils.plot_confusion_matrix(self.latest_metrics['cm'])
            canvas_cm = FigureCanvasTkAgg(fig_cm, master=self.cm_frame)
            canvas_cm.draw()
            canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        fig_err, ax = plt.subplots(figsize=(5, 4))
        
        normal_mse = mse_values[self.latest_preds == 0]
        threat_mse = mse_values[self.latest_preds == 1]

        if len(normal_mse) > 0:
            ax.hist(normal_mse, bins=50, alpha=0.6, color='seagreen', label='Normal Traffic')
        if len(threat_mse) > 0:
            ax.hist(threat_mse, bins=50, alpha=0.6, color='crimson', label='Threats')
        
        ax.axvline(self.detector.threshold, color='black', linestyle='--', label=f'Threshold')
        ax.set_title("Reconstruction Error Distribution")
        ax.set_xlabel("Mean Squared Error (MSE)")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        
        canvas_err = FigureCanvasTkAgg(fig_err, master=self.error_frame)
        canvas_err.draw()
        canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- NEW HISTORY SECTIONS ---
    def _build_history_tab(self):
        panel = ttk.Frame(self.tab_history, padding=20)
        panel.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        ctrl_frame = ttk.Frame(panel)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ctrl_frame, text="Scan History Log", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="🗑️ Clear History", command=self.clear_history_action).pack(side=tk.RIGHT, padx=5)
        ttk.Button(ctrl_frame, text="🔄 Refresh", command=self.refresh_history_view).pack(side=tk.RIGHT, padx=5)
        
        # Treeview
        cols = ("Time", "Total Packets", "Normal", "Threats", "Threat %")
        self.history_tree = ttk.Treeview(panel, columns=cols, show='headings')
        
        for c in cols: self.history_tree.heading(c, text=c)
        
        self.history_tree.column("Time", width=150)
        self.history_tree.column("Total Packets", width=100)
        self.history_tree.column("Normal", width=100)
        self.history_tree.column("Threats", width=100)
        self.history_tree.column("Threat %", width=80)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        self.refresh_history_view()

    def refresh_history_view(self):
        # Clear current list
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Load from file
        data = utils.load_history()
        
        # Insert (Newest first)
        for entry in reversed(data):
            t_pct = (entry['anomalies'] / entry['total'] * 100) if entry['total'] > 0 else 0
            row = (
                entry['timestamp'],
                entry['total'],
                entry['normal'],
                entry['anomalies'],
                f"{t_pct:.2f}%"
            )
            self.history_tree.insert("", tk.END, values=row)

    def clear_history_action(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete all scan history?"):
            utils.clear_history_data()
            self.refresh_history_view()
            messagebox.showinfo("Success", "History Cleared.")

    # --- CRITICAL FIX: FORCED EXIT ---
    def on_closing(self):
        """Cleanly handle application shutdown to prevent thread/Tkinter errors"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            try:
                # 1. Close plots
                plt.close('all') 
                # 2. Stop Tkinter loop
                self.root.quit()
                self.root.destroy()
            except:
                pass
            finally:
                # 3. NUCLEAR OPTION: Force kill the process.
                # This stops all background threads instantly.
                os._exit(0)