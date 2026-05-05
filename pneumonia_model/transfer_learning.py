import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """Builds a transfer learning model using EfficientNetV2B0 as the base (fully frozen initially)."""
    
    # Enhanced Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ], name="data_augmentation")

    # Load the pre-trained EfficientNetV2B0 base model
    base_model = EfficientNetV2B0(input_shape=input_shape,
                                  include_top=False,
                                  weights='imagenet')
    
    # Freeze the base_model entirely for Phase 1
    base_model.trainable = False

    # Create the top layers for our specific binary classification task
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(1, activation='sigmoid')
    
    # Build the full model
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False) # Important: keep BatchNormalization layers in inference mode
    x = global_average_layer(x)
    x = layers.Dropout(0.3)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model with Focal Loss to heavily penalize false negatives/positives
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_transfer_learning_model()
    model.summary()
