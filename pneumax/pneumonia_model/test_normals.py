import tensorflow as tf
import os
import kagglehub
import numpy as np

# Load the model
model = tf.keras.models.load_model('checkpoints/transfer_learning_model_best.keras')

# Get the path to the dataset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
normal_dir = os.path.join(path, 'chest_xray', 'test', 'NORMAL')

# List some normal images
normal_images = os.listdir(normal_dir)[:10]

print("Testing 10 healthy (NORMAL) X-Rays from the test dataset...")
correct = 0

for img_name in normal_images:
    img_path = os.path.join(normal_dir, img_name)
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    score = model.predict(img_array, verbose=0)[0][0]
    
    if score > 0.65:
        print(f"[WRONG] {img_name} -> Diagnosed as PNEUMONIA ({score*100:.1f}%)")
    else:
        print(f"[CORRECT] {img_name} -> Diagnosed as NORMAL ({(1-score)*100:.1f}%)")
        correct += 1

print(f"\nFinal Result: {correct}/10 correct on healthy images.")
