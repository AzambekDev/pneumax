import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model using DenseNet121.
    DenseNet is particularly effective for medical imaging due to its dense connectivity.
    """
    
    # Enhanced Data Augmentation for medical precision
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="data_augmentation")

    # Load the pre-trained DenseNet121 base model
    base_model = DenseNet121(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')
    
    # Freeze the base_model for initial training
    base_model.trainable = False

    # Build the full model
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # DenseNet expects specific preprocessing
    x = preprocess_input(x)
    x = base_model(x, training=False)
    
    # Global Pooling + Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="Pneumax_DenseNet_Diagnostic")
    
    # Compile with Focal Loss to handle class imbalance and prioritize high recall
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return model

if __name__ == "__main__":
    model = build_transfer_learning_model()
    model.summary()
