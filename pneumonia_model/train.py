import argparse
import os
import tensorflow as tf
import kagglehub
from utils import load_data, plot_history
from custom_cnn import build_custom_cnn
from transfer_learning import build_transfer_learning_model

def train(model_type, data_dir, epochs=20, batch_size=32):
    # Setup data
    train_ds, val_ds, _ = load_data(data_dir, batch_size=batch_size)
    
    # Common Setup
    os.makedirs('checkpoints', exist_ok=True)
    
    # Dynamically calculate class weights based on actual file counts
    train_normal_dir = os.path.join(data_dir, 'train', 'NORMAL')
    train_pneumonia_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')
    
    if os.path.exists(train_normal_dir) and os.path.exists(train_pneumonia_dir):
        num_normal = len(os.listdir(train_normal_dir))
        num_pneumonia = len(os.listdir(train_pneumonia_dir))
    else:
        num_normal = 1341
        num_pneumonia = 3875
        
    total_images = num_normal + num_pneumonia
    
    weight_for_0 = (1 / max(1, num_normal)) * (total_images / 2.0)
    weight_for_1 = (1 / max(1, num_pneumonia)) * (total_images / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    
    print(f"\nDataset Split: {num_normal} Normal | {num_pneumonia} Pneumonia")
    print(f"Calculated Class Weights: Normal={weight_for_0:.2f}, Pneumonia={weight_for_1:.2f}\n")
    
    if model_type == "custom":
        print("Initializing Custom CNN Model...")
        model = build_custom_cnn()
        model_name = "custom_cnn_model"
        
        checkpoint_filepath = f"checkpoints/{model_name}_best.keras"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True)
            
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            
        print(f"Starting training for {epochs} epochs...")
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=[model_checkpoint, early_stopping, reduce_lr], class_weight=class_weights
        )
        
    elif model_type == "transfer":
        print("Initializing Transfer Learning Model (Phase 1: Top Layers)...")
        model = build_transfer_learning_model()
        model_name = "transfer_learning_model"
        
        checkpoint_filepath = f"checkpoints/{model_name}_best.keras"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True)
            
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            
        phase1_epochs = 5
        print(f"\n--- PHASE 1: Training top layers for {phase1_epochs} epochs ---")
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=phase1_epochs,
            callbacks=[model_checkpoint, early_stopping, reduce_lr], class_weight=class_weights
        )
        
        phase2_epochs = epochs - phase1_epochs
        if phase2_epochs > 0:
            print(f"\n--- PHASE 2: Deep Fine-Tuning base model for {phase2_epochs} epochs ---")
            
            # Unfreeze the base model
            for layer in model.layers:
                if layer.name == 'efficientnetv2-b0':
                    layer.trainable = True
                    # Keep BatchNormalization frozen to prevent training instability
                    for sub_layer in layer.layers:
                        if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                            sub_layer.trainable = False
                            
            # Recompile with a much lower learning rate for fine-tuning
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                          loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
                          metrics=['accuracy'])
                          
            history2 = model.fit(
                train_ds, validation_data=val_ds, epochs=phase2_epochs,
                callbacks=[model_checkpoint, early_stopping, reduce_lr], class_weight=class_weights
            )
            
            # Merge histories for plotting
            for key in history.history:
                history.history[key].extend(history2.history[key])
                
    else:
        raise ValueError("Invalid model type. Choose 'custom' or 'transfer'.")

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
