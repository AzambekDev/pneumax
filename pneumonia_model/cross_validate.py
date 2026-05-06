import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from utils import load_data
from transfer_learning import build_transfer_learning_model

def run_cross_validation(data_dir, n_splits=5, epochs=5, batch_size=32):
    print(f"Starting {n_splits}-fold Cross-Validation...")
    
    # Load all training data
    train_dir = os.path.join(data_dir, 'train')
    
    # We need to get the file list and labels to do custom splitting
    file_paths = []
    labels = []
    for i, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_path = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_path):
            file_paths.append(os.path.join(class_path, img_name))
            labels.append(i)
            
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Create datasets for this fold
        def create_ds(paths, lbls):
            def load_and_preprocess(path, label):
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [224, 224])
                return img, label
            
            ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
            ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_ds = create_ds(file_paths[train_idx], labels[train_idx])
        val_ds = create_ds(file_paths[val_idx], labels[val_idx])
        
        # Build and compile model
        model = build_transfer_learning_model()
        
        # Train (Short training for CV)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
        
        # Evaluate on validation fold
        val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
        print(f"Fold {fold+1} Result -> Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        fold_results.append((val_acc, val_auc))
        
    # Summary
    accs = [r[0] for r in fold_results]
    aucs = [r[1] for r in fold_results]
    
    print("\n" + "="*30)
    print("CROSS-VALIDATION SUMMARY")
    print("="*30)
    print(f"Mean Accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"Mean AUC:      {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print("="*30)

if __name__ == "__main__":
    import argparse
    import kagglehub
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/chest_xray')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    if args.data_dir == '../data/chest_xray':
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        if os.path.exists(os.path.join(path, 'chest_xray')):
            args.data_dir = os.path.join(path, 'chest_xray')
        else:
            args.data_dir = path
            
    run_cross_validation(args.data_dir, n_splits=args.folds, epochs=args.epochs)
