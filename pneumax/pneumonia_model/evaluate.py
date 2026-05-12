import argparse
import os
import tensorflow as tf
import kagglehub
from utils import load_data, evaluate_model

def evaluate(model_path, data_dir, batch_size=32):
    # Setup test data
    _, _, test_ds = load_data(data_dir, batch_size=batch_size)
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    evaluate_model(model, test_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained Pneumonia Detection Models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.keras or .h5)')
    parser.add_argument('--data_dir', type=str, default='../data/chest_xray',
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if args.data_dir == '../data/chest_xray':
        print("Using kagglehub to download/locate the dataset...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Dataset downloaded to:", path)
        if os.path.exists(os.path.join(path, 'chest_xray')):
            args.data_dir = os.path.join(path, 'chest_xray')
        else:
            args.data_dir = path
            
    evaluate(args.model_path, args.data_dir, args.batch_size)
