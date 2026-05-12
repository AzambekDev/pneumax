import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_data(data_dir, batch_size=32, img_height=224, img_width=224):
    """Loads and preprocesses image data from directories."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    print("Loading Training Data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary'
    )

    print("Loading Validation Data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary'
    )
    
    print("Loading Testing Data...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def plot_history(history, title="Model Training History"):
    """Plots training/validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def evaluate_model(model, test_ds, class_names=['Normal', 'Pneumonia']):
    """Evaluates the model and prints classification report & confusion matrix."""
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = np.concatenate([y for x, y in test_ds], axis=0).flatten()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

def get_gradcam_model(model):
    """Creates a sub-model that outputs both predictions and the last conv layer activations."""
    base_model_layer = None
    # Check for common pre-trained architectures
    for layer in model.layers:
        lower_name = layer.name.lower()
        if 'efficientnet' in lower_name or 'densenet' in lower_name or 'resnet' in lower_name:
            base_model_layer = layer
            break
    
    if not base_model_layer:
        # Fallback for custom models: find the last Conv2D layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if not last_conv_layer:
             return None, None
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
        return grad_model, None
    else:
        # Dynamically find the last Conv2D layer within the base model
        last_conv_layer = None
        for layer in reversed(base_model_layer.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if not last_conv_layer:
            return None, None
            
        grad_model = tf.keras.models.Model(
            [base_model_layer.inputs], [last_conv_layer.output, base_model_layer.output]
        )
        return grad_model, base_model_layer

def make_gradcam_heatmap(img_array, model):
    """Generates a Grad-CAM heatmap for a given image array and model."""
    grad_model, base_model = get_gradcam_model(model)
    if grad_model is None:
        return np.zeros((224, 224))

    with tf.GradientTape() as tape:
        if base_model:
            x = img_array
            for layer in model.layers:
                if layer == base_model:
                    break
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    x = layer(x, training=False)
                except:
                    x = layer(x)
            
            conv_outputs, base_output = grad_model(x)
            
            # Reconstruct the path to the final prediction
            x = base_output
            found_base = False
            for layer in model.layers:
                if not found_base:
                    if layer == base_model:
                        found_base = True
                    continue
                try:
                    x = layer(x, training=False)
                except:
                    x = layer(x)
            predictions = x
        else:
            conv_outputs, predictions = grad_model(img_array)
            
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="heatmap.jpg", alpha=0.4):
    """Superimposes the heatmap onto the original image and saves it."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray(np.uint8(255 * jet_heatmap))
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    jet_heatmap_array = np.array(jet_heatmap)
    superimposed_img = jet_heatmap_array * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path
