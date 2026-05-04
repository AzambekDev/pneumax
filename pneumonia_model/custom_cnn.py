import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(input_shape=(224, 224, 3)):
    """Builds a custom CNN model from scratch."""
    
    # Data Augmentation layer built into the model
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = models.Sequential([
        # Input and Data Augmentation
        tf.keras.Input(shape=input_shape),
        data_augmentation,
        
        # Rescaling pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dropout(0.5), # Regularization
        layers.Dense(512, activation='relu'),
        
        # Output Layer (Binary Classification: Normal or Pneumonia)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
