🌿 Plant Disease Classification using Deep Learning

This project implements an end-to-end deep learning pipeline for plant disease classification from leaf images, covering data validation, baseline modeling, transfer learning, fine-tuning, evaluation, and deployment via a Streamlit web application.

The primary goal is to build a robust, reproducible, and deployable image classification system that can accurately distinguish between healthy and diseased plant leaves across multiple crop types.

📌 Project Motivation

Directly starting with deep pretrained models can sometimes hide issues related to:
dataset structure
label mapping
preprocessing mismatches
training pipeline bugs
To avoid this, the project follows a progressive modeling strategy:
Train a lightweight CNN from scratch to validate the data pipeline
Apply transfer learning with EfficientNet-B3
Fine-tune the pretrained backbone for disease-specific features
Deploy the final model for real-time inference

📂 Dataset

Source: New Plant Diseases Dataset (Augmented)
Classes: 38 (multiple crops × diseases + healthy)
Structure:

dataset/
└── New Plant Diseases Dataset(Augmented)/
    ├── train/
    ├── valid/
    └── test/

🧪 Experiments Overview
1️⃣ Baseline CNN (Sanity Check)

Before applying transfer learning, a lightweight custom CNN was trained from scratch.

Purpose
Validate dataset integrity
Confirm correct label handling
Verify loss function and training loop behavior
Setup
Architecture: Custom CNN
Epochs: 10
Optimizer: Adam
Loss: Categorical Cross-Entropy

Results

Training Accuracy: ~95%
Validation Accuracy: ~93%
Minor overfitting observed after epoch 6
This confirmed that the data pipeline and training setup were correct.

2️⃣ EfficientNet-B3 (Transfer Learning)

EfficientNet-B3 was chosen due to its compound scaling, balancing accuracy and computational efficiency.
Key Characteristics
MBConv blocks with Squeeze-and-Excitation
Depthwise separable convolutions
Swish activation
~26 MBConv blocks (~210 internal layers)

🔁 Training Strategy
Stage 1 — Feature Extraction (Head Training)

Backbone: Frozen EfficientNet-B3 (ImageNet-pretrained)
Trainable layers: Classification head only
Learning Rate: ~1e-3
Epochs: 3
Result
Validation Accuracy: ~96.9%
Demonstrates strong transferability of ImageNet features

Stage 2 — Fine-Tuning

Unfrozen: Top ~40% of EfficientNet-B3 backbone
Learning Rate: 1e-4 → 1e-5
Epochs: 15
Regularization:
Label smoothing (0.1)
Dropout (0.3–0.5)
Weight decay (1e-4)
Result
Validation Accuracy: ~98.9%
Lower validation loss
Improved class separation and recall

📊 Evaluation & Monitoring
To handle class imbalance and disease-specific performance, evaluation went beyond accuracy.

Metrics Tracked
Macro-F1 (primary metric)
Per-class precision & recall
Confusion matrix
Inference latency
Model size & memory footprint
Visualizations
Training vs validation accuracy curves
Training vs validation loss curves
Confusion matrices
Per-class recall and precision plots
All plots are saved experiment-wise for reproducibility and comparison.

📁 Project Structure
Classification_Project/
│
├── experiments/
│   ├── baseline_cnn/
│   │   ├── history.json
│   │   └── visualizations/
│   │       ├── accuracy_curve.png
│   │       ├── loss_curve.png
│   │       ├── confusion_matrix.png
│   │       └── per_class_recall.png
│   │
│   ├── efficientnet_b3/
│   │   ├── stage1_head_training/
│   │   └── stage2_finetuning/
│
├── models/
│   ├── baseline_cnn.h5
│   ├── efficientnet_b3_stage1.h5
│   ├── efficientnet_b3_stage2.keras
│   └── class_names.json
│
├── src/
│   ├── data/
│   ├── models/
│   ├── visualizations/
│   └── training scripts
│
├── streamlit_app/
│   ├── app.py
│   ├── utils.py
│   └── config.py
│
└── README.md

🚀 Deployment (Streamlit App)
A Streamlit-based web application was developed for real-time plant disease classification.

Features
Upload any leaf image
Model predicts disease / healthy class
Confidence score displayed
Runs fully offline
CPU-compatible
Uses correct EfficientNet preprocessing

Run the App
streamlit run streamlit_app/app.py

🧠 Key Engineering Learnings
Training a baseline model first prevents silent data bugs
Class-index persistence (class_names.json) is critical for deployment
Preprocessing mismatches can completely break inference
EfficientNet preprocessing must match training
Validation accuracy alone is insufficient for imbalanced datasets

📌 Final Notes
This project demonstrates a complete ML lifecycle:

data validation
model experimentation
transfer learning
fine-tuning
evaluation
deployment

The final EfficientNet-B3 model achieves high accuracy, strong generalization, and real-time usability, making it suitable for practical agricultural decision support systems.