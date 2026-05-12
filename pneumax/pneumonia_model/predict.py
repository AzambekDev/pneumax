import argparse
import tensorflow as tf
import numpy as np

from utils import make_gradcam_heatmap, save_and_display_gradcam
import os

def predict_image(model_path, image_path):
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess the image
    print(f"Loading image from {image_path}...")
    try:
        # Resize to match the training input shape
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        # Convert to numpy array and add a batch dimension
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Make the prediction
    print("\nAnalyzing X-Ray...")
    predictions = model.predict(img_array, verbose=0)
    
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    output_heatmap = "prediction_reasoning.jpg"
    save_and_display_gradcam(image_path, heatmap, output_heatmap)
    
    score = predictions[0][0]
    
    if score > 0.65:
        confidence = score * 100
        print(f"\n🩺 Diagnosis: PNEUMONIA ({confidence:.2f}% confidence)")
        print(f"🔍 Reasoning visualization saved to: {output_heatmap}")
    else:
        confidence = (1 - score) * 100
        print(f"\n🩺 Diagnosis: NORMAL ({confidence:.2f}% confidence)")
        print(f"🔍 Reasoning visualization saved to: {output_heatmap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a single X-Ray image')
    parser.add_argument('--model_path', type=str, default='checkpoints/transfer_learning_model_best.keras',
                        help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the single X-Ray image you want to test')
    
    args = parser.parse_args()
    
    # Turn off TF logging spam
    tf.get_logger().setLevel('ERROR')
    
    predict_image(args.model_path, args.image_path)
