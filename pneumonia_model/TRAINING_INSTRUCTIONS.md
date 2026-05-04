# Pneumonia Model Training Instructions

Yes, you will need to source the X-ray images to train the model, as the code we wrote requires a dataset to learn from!

I designed the code to work seamlessly with the standard dataset for this task: the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney.

I've updated the code to use **kagglehub**! You no longer need to download the dataset manually.

### Installation

Make sure your terminal is in the `pneumonia_model` directory and run:

```bash
& "c:\Users\Azambek Sattarov\OneDrive\Desktop\project\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```
*(This command uses your virtual environment and handles spaces in your folder path correctly).*

### Start Training

The dataset will **automatically download** the first time you run a training command. 

**To train the Custom CNN model:**
```bash
& "c:\Users\Azambek Sattarov\OneDrive\Desktop\project\.venv\Scripts\python.exe" train.py --model custom --epochs 20
```

**To train the Transfer Learning model (MobileNetV2):**
```bash
& "c:\Users\Azambek Sattarov\OneDrive\Desktop\project\.venv\Scripts\python.exe" train.py --model transfer --epochs 20
```
