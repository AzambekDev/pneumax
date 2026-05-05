from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image

app = Flask(__name__)

# --- Setup Model ---
# Use the transfer learning model by default as it's typically more accurate
MODEL_PATH = 'checkpoints/transfer_learning_model_best.keras'

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading AI Model from {MODEL_PATH}. This might take a few seconds...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found. Please train the model first.")

# Load the model on startup
load_model()

def analyze_image(img_stream):
    try:
        # Open image with PIL
        img = Image.open(img_stream).convert('RGB')
        
        # Resize to match model input shape
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Add batch dimension
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = float(predictions[0][0])
        
        # Since we used a Sigmoid activation (0 to 1), 
        # a score close to 0 means Normal, close to 1 means Pneumonia.
        if score > 0.5:
            confidence = score * 100
            diagnosis = "PNEUMONIA"
        else:
            confidence = (1 - score) * 100
            diagnosis = "NORMAL"
            
        return {
            "success": True,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "score": score
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
         return jsonify({"success": False, "error": "Model not found. Please train the model first."})
         
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
        
    if file:
        img_bytes = file.read()
        result = analyze_image(io.BytesIO(img_bytes))
        return jsonify(result)

if __name__ == '__main__':
    # Add flask to requirements if not there
    print("Starting Web GUI...")
    app.run(debug=True, port=5000)
