import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(input_shape=(224, 224, 3)):
    """Builds an improved custom CNN model from scratch."""
    
    # Enhanced Data Augmentation layer built into the model
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ], name="data_augmentation")

    model = models.Sequential([
        # Input and Data Augmentation
        tf.keras.Input(shape=input_shape),
        data_augmentation,
        
        # Rescaling pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'), # Increased filters
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling replaces Flatten to reduce parameters and overfitting
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4), # Regularization
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output Layer (Binary Classification: Normal or Pneumonia)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use Focal Loss to heavily penalize hard-to-classify examples
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
