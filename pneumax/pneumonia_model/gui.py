import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf
import numpy as np
import os
import datetime
from utils import make_gradcam_heatmap, save_and_display_gradcam

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'transfer_learning_model_best.keras')
TEMP_HEATMAP_PATH = os.path.join(BASE_DIR, 'temp_analysis.jpg')

# --- Load Model ---
print("Initializing Medical Diagnostic System...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    MODEL_PATH = os.path.join(BASE_DIR, 'transfer_learning_model_final.keras')
    model = tf.keras.models.load_model(MODEL_PATH)

class PneumaxGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumax™ Clinical Diagnostic Suite")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f2f5")
        
        self.current_score = 0
        self.current_file = None
        
        self.setup_styles()
        self.build_ui()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Medical Theme Colors
        self.primary_blue = "#004a99"
        self.bg_light = "#f0f2f5"
        self.card_bg = "#ffffff"
        self.text_dark = "#1c1e21"
        self.accent_blue = "#00a8cc"
        
        self.style.configure("Medical.TFrame", background=self.bg_light)
        self.style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        
    def build_ui(self):
        # --- Top Navigation Bar ---
        top_bar = tk.Frame(self.root, bg=self.primary_blue, height=60)
        top_bar.pack(side="top", fill="x")
        
        title_lbl = tk.Label(top_bar, text="PNEUMAX™ DIAGNOSTIC SUITE", font=("Helvetica", 16, "bold"), 
                             bg=self.primary_blue, fg="white", padx=20)
        title_lbl.pack(side="left", pady=15)
        
        self.status_lbl = tk.Label(top_bar, text="SYSTEM READY", font=("Helvetica", 9, "bold"), 
                                   bg="#003366", fg="#00ff00", padx=15, pady=5)
        self.status_lbl.pack(side="right", padx=20)

        # --- Main Content Area ---
        content = tk.Frame(self.root, bg=self.bg_light, padx=20, pady=20)
        content.pack(fill="both", expand=True)
        
        # Sidebar (Patient Info)
        sidebar = tk.Frame(content, bg=self.card_bg, width=280, highlightbackground="#d1d3d4", highlightthickness=1)
        sidebar.pack(side="left", fill="y", padx=(0, 20))
        sidebar.pack_propagate(False)
        
        tk.Label(sidebar, text="PATIENT INFORMATION", font=("Helvetica", 10, "bold"), bg=self.card_bg, fg=self.primary_blue).pack(pady=20)
        
        self.add_input_field(sidebar, "Patient Name:", "John Doe")
        self.add_input_field(sidebar, "Patient ID:", "PX-99283")
        self.add_input_field(sidebar, "Age / Sex:", "45 / M")
        self.add_input_field(sidebar, "Scan Date:", datetime.date.today().strftime("%Y-%m-%d"))
        
        tk.Label(sidebar, text="DIAGNOSTIC CONTROLS", font=("Helvetica", 10, "bold"), bg=self.card_bg, fg=self.primary_blue).pack(pady=(40, 10))
        
        # Sensitivity Control
        tk.Label(sidebar, text="System Sensitivity", font=("Helvetica", 9), bg=self.card_bg).pack()
        self.sens_slider = ttk.Scale(sidebar, from_=0.1, to=0.9, value=0.5, orient="horizontal", command=self.update_results)
        self.sens_slider.pack(fill="x", padx=20, pady=5)
        self.sens_lbl = tk.Label(sidebar, text="Threshold: 0.50", font=("Helvetica", 8), bg=self.card_bg, fg="#606770")
        self.sens_lbl.pack()
        
        btn_upload = tk.Button(sidebar, text="UPLOAD X-RAY", font=("Helvetica", 11, "bold"), bg=self.primary_blue, fg="white",
                               activebackground=self.accent_blue, cursor="hand2", borderwidth=0, pady=12, command=self.open_file)
        btn_upload.pack(side="bottom", fill="x", padx=20, pady=20)

        # Center Analysis Area
        analysis_area = tk.Frame(content, bg=self.bg_light)
        analysis_area.pack(side="left", fill="both", expand=True)
        
        # Image Comparison Grid
        grid = tk.Frame(analysis_area, bg=self.bg_light)
        grid.pack(fill="both", expand=True)
        
        # Original View
        orig_card = tk.Frame(grid, bg=self.card_bg, highlightbackground="#d1d3d4", highlightthickness=1)
        orig_card.place(relx=0, rely=0, relwidth=0.48, relheight=0.6)
        tk.Label(orig_card, text="PRIMARY RADIOGRAPH", font=("Helvetica", 9, "bold"), bg=self.card_bg, fg="#606770").pack(pady=10)
        self.orig_img_lbl = tk.Label(orig_card, text="Awaiting Scan...", bg="#f8f9fa", fg="#adb5bd")
        self.orig_img_lbl.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        
        # Analysis View
        analysis_card = tk.Frame(grid, bg=self.card_bg, highlightbackground="#d1d3d4", highlightthickness=1)
        analysis_card.place(relx=0.52, rely=0, relwidth=0.48, relheight=0.6)
        tk.Label(analysis_card, text="AI PATHOLOGICAL REASONING", font=("Helvetica", 9, "bold"), bg=self.card_bg, fg="#606770").pack(pady=10)
        self.heat_img_lbl = tk.Label(analysis_card, text="Heatmap View", bg="#f8f9fa", fg="#adb5bd")
        self.heat_img_lbl.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        
        # Bottom Report Card
        report_card = tk.Frame(grid, bg=self.card_bg, highlightbackground="#d1d3d4", highlightthickness=1)
        report_card.place(relx=0, rely=0.65, relwidth=1, relheight=0.35)
        
        tk.Label(report_card, text="CLINICAL DIAGNOSTIC REPORT", font=("Helvetica", 10, "bold"), bg=self.card_bg, fg=self.primary_blue).pack(anchor="w", padx=20, pady=10)
        
        report_cols = tk.Frame(report_card, bg=self.card_bg)
        report_cols.pack(fill="both", expand=True, padx=20)
        
        # Left side of report
        self.diag_lbl = tk.Label(report_cols, text="DIAGNOSIS: N/A", font=("Helvetica", 22, "bold"), bg=self.card_bg, fg="#adb5bd")
        self.diag_lbl.pack(anchor="w")
        
        self.conf_lbl = tk.Label(report_cols, text="Confidence Level: --%", font=("Helvetica", 12), bg=self.card_bg, fg="#606770")
        self.conf_lbl.pack(anchor="w", pady=5)
        
        # Right side: AI Summary
        self.summary_text = tk.Text(report_cols, font=("Helvetica", 10), bg="#f8f9fa", relief="flat", height=4, padx=10, pady=10)
        self.summary_text.pack(fill="x", pady=10)
        self.summary_text.insert("1.0", "System idle. Please upload a chest X-ray (AP/PA view) to initiate AI-assisted pathological screening.")
        self.summary_text.configure(state="disabled")

    def add_input_field(self, parent, label, default):
        f = tk.Frame(parent, bg=self.card_bg, padx=20, pady=5)
        f.pack(fill="x")
        tk.Label(f, text=label, font=("Helvetica", 8), bg=self.card_bg, fg="#606770").pack(anchor="w")
        e = tk.Entry(f, bg="#f8f9fa", relief="flat", highlightthickness=1, highlightbackground="#d1d3d4")
        e.insert(0, default)
        e.pack(fill="x", pady=2)

    def update_results(self, val=None):
        if not self.current_file: return
        
        threshold = float(self.sens_slider.get())
        self.sens_lbl.configure(text=f"Threshold: {threshold:.2f}")
        
        if self.current_score > threshold:
            diag = "PNEUMONIA DETECTED"
            color = "#d93025" # Medical Red
            summary = "Pathological opacities detected. The AI visual reasoning highlights areas of concern in the lung parenchyma. Clinical correlation is advised."
        else:
            diag = "UNREMARKABLE"
            color = "#188038" # Medical Green
            summary = "No significant pathological opacities detected at the current sensitivity level. Lung fields appear clear."
            
        self.diag_lbl.configure(text=diag, fg=color)
        conf = self.current_score if self.current_score > threshold else (1 - self.current_score)
        self.conf_lbl.configure(text=f"Confidence Level: {conf*100:.1f}%")
        
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary)
        self.summary_text.configure(state="disabled")

    def open_file(self):
        file_path = filedialog.askopenfilename(title="Select Medical Imaging File", filetypes=[("DICOM/Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_file = file_path
            self.status_lbl.configure(text="ANALYZING...", fg="#ffcc00")
            self.root.update()
            
            # Load images
            orig_img = Image.open(file_path).resize((400, 400))
            self.orig_tk = ImageTk.PhotoImage(orig_img)
            self.orig_img_lbl.configure(image=self.orig_tk, text="")
            
            # Run AI
            img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array, verbose=0)
            self.current_score = float(predictions[0][0])
            
            heatmap = make_gradcam_heatmap(img_array, model)
            save_and_display_gradcam(file_path, heatmap, TEMP_HEATMAP_PATH)
            
            heat_img = Image.open(TEMP_HEATMAP_PATH).resize((400, 400))
            self.heat_tk = ImageTk.PhotoImage(heat_img)
            self.heat_img_lbl.configure(image=self.heat_tk, text="")
            
            self.update_results()
            self.status_lbl.configure(text="ANALYSIS COMPLETE", fg="#00ff00")

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumaxGUI(root)
    root.mainloop()
