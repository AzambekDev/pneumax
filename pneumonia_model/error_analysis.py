import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_data, make_gradcam_heatmap, save_and_display_gradcam

def run_error_analysis(model_path, data_dir, output_dir='error_analysis'):
    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data (shuffle=False is critical here)
    _, _, test_ds = load_data(data_dir, batch_size=32)
    
    # Get true labels and image paths
    # Note: image_dataset_from_directory doesn't easily give file paths
    # So we'll iterate manually or use the underlying file paths
    test_dir = os.path.join(data_dir, 'test')
    class_names = sorted(os.listdir(test_dir))
    
    file_paths = []
    true_labels = []
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_path):
            file_paths.append(os.path.join(class_path, img_name))
            true_labels.append(i)
            
    print(f"Total test images: {len(file_paths)}")
    
    # Predict
    print("Generating predictions...")
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Identify errors
    false_positives = [] # Normal (0) predicted as Pneumonia (1)
    false_negatives = [] # Pneumonia (1) predicted as Normal (0)
    
    for i in range(len(y_pred)):
        if true_labels[i] == 0 and y_pred[i] == 1:
            false_positives.append(i)
        elif true_labels[i] == 1 and y_pred[i] == 0:
            false_negatives.append(i)
            
    print(f"Found {len(false_positives)} False Positives and {len(false_negatives)} False Negatives.")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'false_positives'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'false_negatives'), exist_ok=True)
    
    # Process top 10 of each
    def process_errors(indices, error_type):
        print(f"Processing top {min(10, len(indices))} {error_type}...")
        for i in indices[:10]:
            img_path = file_paths[i]
            img_name = os.path.basename(img_path)
            
            # Load and preprocess
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Generate Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model)
            cam_path = os.path.join(output_dir, f"{error_type}", f"gradcam_{img_name}")
            save_and_display_gradcam(img_path, heatmap, cam_path)
            
            print(f"  - Saved analysis for {img_name} to {cam_path}")

    process_errors(false_positives, 'false_positives')
    process_errors(false_negatives, 'false_negatives')
    
    print(f"\nError Analysis complete! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    import argparse
    import kagglehub
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/transfer_learning_model_best.keras')
    parser.add_argument('--data_dir', type=str, default='../data/chest_xray')
    args = parser.parse_args()
    
    if args.data_dir == '../data/chest_xray':
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        if os.path.exists(os.path.join(path, 'chest_xray')):
            args.data_dir = os.path.join(path, 'chest_xray')
        else:
            args.data_dir = path
            
    run_error_analysis(args.model_path, args.data_dir)
