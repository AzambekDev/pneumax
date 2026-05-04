import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_transfer_learning_model(input_shape=(224, 224, 3), fine_tune_at=100):
    """Builds a transfer learning model using MobileNetV2 as the base."""
    
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Load the pre-trained MobileNetV2 base model
    # MobileNetV2 expects input pixels in [-1, 1], so we use preprocess_input
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')
    
    # Freeze the base_model
    base_model.trainable = False
    
    # Unfreeze top layers for fine-tuning if specified
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Create the top layers for our specific binary classification task
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(1, activation='sigmoid')
    
    # Build the full model
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False) # Important: keep BatchNormalization layers in inference mode
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    # We use a lower learning rate for transfer learning/fine-tuning
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_transfer_learning_model(fine_tune_at=None) # Start fully frozen
    model.summary()
