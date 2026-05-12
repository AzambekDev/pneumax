🫁 Pneumax: Deep Learning for Pediatric Pneumonia Detection
Pneumax is a specialized deep learning pipeline engineered to detect pneumonia in pediatric chest X-rays. By leveraging Keras and TensorFlow, the project compares the efficacy of a bespoke Convolutional Neural Network (CNN) against a fine-tuned MobileNetV2 architecture.

🚀 Key Features
Dual-Architecture Strategy: Compare a custom-built CNN against a high-performance Transfer Learning model.

Automated Data Management: Integrated hooks to handle the ingestion of the 5.8GB Paul Mooney dataset automatically.

Production-Ready Evaluation: Built-in generation of Confusion Matrices and Precision-Recall curves.

Interactive Interface: A dedicated GUI for users to upload images and receive instant classifications.

🛠 Project Architecture
The repository is structured to maintain a strict separation between model definitions and the execution pipeline:

Core Modules
custom_cnn.py: Defines a multi-layer CNN with Dropout and Batch Normalization.

transfer_learning.py: Implements MobileNetV2 with frozen base layers for feature extraction.

train.py: The orchestration engine for training, supporting dynamic hyperparameter tuning.

gui.py: A graphical interface for real-time image inference.

utils.py: Handles data augmentation, image resizing, and normalization.

📈 Technical Workflow
Data Ingestion: The pipeline fetches the dataset and splits it into Training, Validation, and Testing sets.

Preprocessing: Images are normalized and augmented (rotation, zoom, horizontal flip) to prevent overfitting.

Training:

Custom CNN: Trains from random initialization to learn specific spatial features.

MobileNetV2: Uses weights pre-trained on ImageNet, fine-tuning the final dense layers for binary classification.

Analysis: Performance is measured using Binary Cross-Entropy loss and monitored via Accuracy and F1-Score.

💻 Getting Started
Prerequisites
Python 3.9 or higher

TensorFlow 2.x

A Kaggle account for dataset access

Installation
Clone the repository to your local machine.

Set up a virtual environment to isolate your dependencies.

Install the required packages (TensorFlow, OpenCV, and Kagglehub).

Execution
The pipeline is managed via Python scripts. You can trigger the training process by specifying the model type (custom or transfer) and the number of iterations. For a more visual experience, the included GUI script allows for direct image uploads and immediate classification results.

📊 Dataset Reference
This project utilizes the Chest X-Ray Images (Pneumonia) dataset, which contains 5,863 JPEG images. The images were collected from pediatric patients at the Guangzhou Women and Children’s Medical Center.

Disclaimer: This tool is for educational and research purposes and is not a substitute for professional medical diagnosis
