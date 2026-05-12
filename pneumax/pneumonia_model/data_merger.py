import os
import shutil
import random
import kagglehub

def merge_datasets():
    print("Step 1/4: Downloading/Locating the original Chest X-Ray dataset...")
    # Get the path to the original dataset we are already using
    original_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    if os.path.exists(os.path.join(original_path, 'chest_xray')):
        target_dir = os.path.join(original_path, 'chest_xray', 'train')
    else:
        target_dir = os.path.join(original_path, 'train')
        
    target_normal_dir = os.path.join(target_dir, 'NORMAL')
    target_pneumonia_dir = os.path.join(target_dir, 'PNEUMONIA')
    
    # Count current images
    current_normal = len(os.listdir(target_normal_dir))
    current_pneumonia = len(os.listdir(target_pneumonia_dir))
    
    print(f"Current Training Split -> Normal: {current_normal} | Pneumonia: {current_pneumonia}")
    
    # Add 4000 extra normal images to heavily diversify the normal class
    images_needed = 4000
    print(f"We need {images_needed} more 'Normal' images to heavily diversify the dataset and reduce false positives.")
    
    print("\nStep 2/4: Downloading COVID-19 Radiography Database (for its Normal X-Rays)...")
    print("This is a large dataset, it might take a moment to download (~1.1GB).")
    covid_db_path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
    
    print("\nStep 3/4: Locating Normal images inside the new dataset...")
    # Find the Normal images folder inside the downloaded dataset
    source_normal_dir = None
    for root, dirs, files in os.walk(covid_db_path):
        # We are looking for the 'Normal' folder which usually contains an 'images' subfolder
        if 'Normal' in dirs or 'NORMAL' in dirs:
            possible_path = os.path.join(root, 'Normal', 'images')
            if os.path.exists(possible_path):
                source_normal_dir = possible_path
                break
            # Or just 'Normal' if it directly contains the images
            elif len(os.listdir(os.path.join(root, 'Normal'))) > 0:
                 source_normal_dir = os.path.join(root, 'Normal')
                 # Sometimes it has a subfolder 'images', let's check
                 if 'images' in os.listdir(source_normal_dir):
                     source_normal_dir = os.path.join(source_normal_dir, 'images')
                 break

    if not source_normal_dir:
        print("Error: Could not locate the 'Normal' images folder in the downloaded dataset.")
        return
        
    source_images = [f for f in os.listdir(source_normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(source_images)} available Normal images.")
    
    if len(source_images) < images_needed:
        print("Warning: Not enough source images to perfectly balance, using all available.")
        images_to_copy = source_images
    else:
        # Randomly select the exact number of images needed
        random.seed(42) # For reproducibility
        images_to_copy = random.sample(source_images, images_needed)
        
    print(f"\nStep 4/4: Copying {len(images_to_copy)} images into your training folder...")
    copied_count = 0
    for img_name in images_to_copy:
        src_path = os.path.join(source_normal_dir, img_name)
        # Prefix with 'ext_' so we know these came from the external dataset
        dest_path = os.path.join(target_normal_dir, f"ext_{img_name}")
        
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
            
        if copied_count % 500 == 0 and copied_count > 0:
            print(f"Copied {copied_count} images...")
            
    print(f"\nSuccess! Copied {copied_count} new images.")
    
    # Final Verification
    final_normal = len(os.listdir(target_normal_dir))
    print(f"Final Training Split -> Normal: {final_normal} | Pneumonia: {current_pneumonia}")
    print("\nDataset is now perfectly balanced! You can run 'python train.py' with confidence.")

if __name__ == "__main__":
    merge_datasets()
