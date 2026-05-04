import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# --- Setup Model ---
MODEL_PATH = 'checkpoints/transfer_learning_model_best.keras'
print("Loading AI Model. This might take a few seconds...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def analyze_image(file_path):
    # Preprocess
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0][0]
    
    # Increase the threshold to 0.85 to make it stricter and avoid False Positives
    if score > 0.85:
        confidence = score * 100
        return "PNEUMONIA", confidence, "#ff4c4c" # Red
    else:
        confidence = (1 - score) * 100
        return "NORMAL", confidence, "#4caf50" # Green

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select an X-Ray Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        # 1. Update Image Display
        img = Image.open(file_path)
        img = img.resize((300, 300)) # Resize for GUI display
        img_tk = ImageTk.PhotoImage(img)
        
        image_label.configure(image=img_tk, text="")
        image_label.image = img_tk # Keep a reference
        
        # 2. Analyze and Update Result
        result_label.configure(text="Analyzing...", fg="#ffffff")
        root.update()
        
        diagnosis, confidence, color = analyze_image(file_path)
        
        result_label.configure(
            text=f"Diagnosis: {diagnosis}\nConfidence: {confidence:.1f}%",
            fg=color
        )

# --- Build GUI ---
root = tk.Tk()
root.title("Pneumonia AI Diagnoser")
root.geometry("450x550")
root.configure(bg="#1e1e2e")

# Title
title_label = tk.Label(root, text="AI X-Ray Analysis", font=("Arial", 20, "bold"), bg="#1e1e2e", fg="#cdd6f4")
title_label.pack(pady=20)

# Image Frame (Placeholder)
image_label = tk.Label(root, text="No Image Selected", bg="#313244", fg="#a6adc8", width=40, height=18)
image_label.pack(pady=10)

# Button
btn = tk.Button(root, text="Select X-Ray Image", font=("Arial", 14), bg="#89b4fa", fg="#11111b", 
                activebackground="#b4befe", cursor="hand2", borderwidth=0, padx=20, pady=10, command=open_file)
btn.pack(pady=20)

# Result Text
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#1e1e2e", fg="#ffffff")
result_label.pack(pady=10)

# Run App
root.mainloop()
