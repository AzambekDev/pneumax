import argparse
import os
import tensorflow as tf
import kagglehub
from utils import load_data, plot_history
from custom_cnn import build_custom_cnn
from transfer_learning import build_transfer_learning_model

def train(model_type, data_dir, epochs=10, batch_size=32):
    # Setup data
    train_ds, val_ds, _ = load_data(data_dir, batch_size=batch_size)
    
    # Initialize model
    if model_type == "custom":
        print("Initializing Custom CNN Model...")
        model = build_custom_cnn()
        model_name = "custom_cnn_model"
    elif model_type == "transfer":
        print("Initializing Transfer Learning Model (MobileNetV2)...")
        # Start by fine-tuning at layer 100 as an example
        model = build_transfer_learning_model(fine_tune_at=100)
        model_name = "transfer_learning_model"
    else:
        raise ValueError("Invalid model type. Choose 'custom' or 'transfer'.")

    model.summary()

    # Callbacks
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_filepath = f"checkpoints/{model_name}_best.keras"
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3, # Stop after 3 epochs of no improvement
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )

    # Mathematically correct class weights to balance the dataset
    # Total training images: ~5216 (1341 Normal, 3875 Pneumonia)
    # Weight formula: (1 / class_count) * (total / 2.0)
    weight_for_0 = (1 / 1341) * (5216 / 2.0) # ~1.94
    weight_for_1 = (1 / 3875) * (5216 / 2.0) # ~0.67
    class_weights = {0: weight_for_0, 1: weight_for_1}

    # Train
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )
    
    # Plot results
    plot_history(history, title=f"{model_type.capitalize()} Model Training")
    
    # Save final model
    final_model_path = f"{model_name}_final.keras"
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    print(f"Best model saved to {checkpoint_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Models')
    parser.add_argument('--model', type=str, choices=['custom', 'transfer'], required=True,
                        help='Type of model to train (custom or transfer)')
    parser.add_argument('--data_dir', type=str, default='../data/chest_xray',
                        help='Path to the dataset directory (default: ../data/chest_xray)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.data_dir == '../data/chest_xray':
        print("Using kagglehub to download/locate the dataset...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Dataset downloaded to:", path)
        if os.path.exists(os.path.join(path, 'chest_xray')):
            args.data_dir = os.path.join(path, 'chest_xray')
        else:
            args.data_dir = path
            
    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    train(args.model, args.data_dir, args.epochs, args.batch_size)
