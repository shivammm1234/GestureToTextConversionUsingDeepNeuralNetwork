"""
=============================================================================
Google Colab Training Script — Dual-Hand Gesture Recognition
=============================================================================
Full pipeline: Preprocessing → Augmentation → Training → Evaluation

HOW TO USE IN GOOGLE COLAB:
    1. Upload your project files to Google Drive
    2. Open a new Colab notebook
    3. Copy and paste each cell (marked with # %%) into separate Colab cells
    4. Run cells sequentially
    
Alternatively, upload this .py file and run:
    !python colab_training.py
=============================================================================
"""

# %% [markdown]
# # 🤟 Dual-Hand Gesture Recognition — Training Pipeline
# ## CNN-LSTM Model with MediaPipe Dual-Hand Landmarks
# 
# This notebook runs the complete training pipeline:
# 1. **Setup** — Install dependencies, mount Drive
# 2. **Preprocessing** — Extract dual-hand MediaPipe landmarks from videos
# 3. **Augmentation** — Generate augmented training data
# 4. **Training** — Train CNN-LSTM model
# 5. **Evaluation** — Generate metrics and plots

# %% Cell 1: Setup & Install Dependencies
# ============================================================================
# CELL 1: SETUP — Mount Drive & Install Dependencies
# ============================================================================

import os
import sys

# Mount Google Drive (for Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Google Drive mounted!")
except ImportError:
    IN_COLAB = False
    print("⚠️ Not running in Colab — skipping Drive mount")

# Install required packages
if IN_COLAB:
    os.system("pip install -q tensorflow==2.15.0 mediapipe==0.10.9 "
              "opencv-python==4.9.0.80 scikit-learn==1.4.2 "
              "matplotlib==3.8.4 seaborn==0.13.2 protobuf==3.20.3")
    print("✅ Dependencies installed!")

# %% Cell 2: Set Project Path
# ============================================================================
# CELL 2: SET PROJECT PATH
# ============================================================================

# ⚠️ MODIFY THIS PATH to match your Google Drive project location
if IN_COLAB:
    PROJECT_PATH = "/content/drive/MyDrive/NewMajorProject 23MArch"
else:
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Change to project directory
os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

print(f"📂 Project path: {PROJECT_PATH}")
print(f"📁 Files in project:")
for f in sorted(os.listdir(PROJECT_PATH)):
    if not f.startswith('.') and not f.startswith('__'):
        filepath = os.path.join(PROJECT_PATH, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"  📄 {f} ({size:.1f} KB)")
        else:
            items = len(os.listdir(filepath))
            print(f"  📁 {f}/ ({items} items)")

# %% Cell 3: Verify Configuration
# ============================================================================
# CELL 3: VERIFY DUAL-HAND CONFIGURATION
# ============================================================================

from config import (
    DATASET_PATH, PREPROCESSED_DATA_PATH, MODEL_SAVE_PATH,
    AUGMENTED_DATA_PATH, EVALUATION_PATH,
    NUM_HANDS, LANDMARKS_PER_HAND, NUM_LANDMARKS,
    NUM_COORDS, NUM_FEATURES, FEATURES_PER_HAND,
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS,
    AUGMENTATION_FACTOR, GESTURE_CLASSES, NUM_CLASSES,
    create_directories
)

print("=" * 60)
print("CONFIGURATION — Dual-Hand Gesture Recognition")
print("=" * 60)
print(f"  Hands:              {NUM_HANDS}")
print(f"  Landmarks per hand: {LANDMARKS_PER_HAND}")
print(f"  Total landmarks:    {NUM_LANDMARKS}")
print(f"  Features per frame: {NUM_FEATURES} ({NUM_HANDS} × {LANDMARKS_PER_HAND} × {NUM_COORDS})")
print(f"  Sequence length:    {SEQUENCE_LENGTH} frames")
print(f"  Gesture classes:    {NUM_CLASSES}")
print(f"  Batch size:         {BATCH_SIZE}")
print(f"  Epochs:             {EPOCHS}")
print(f"  Augmentation:       {AUGMENTATION_FACTOR}x")
print(f"  Dataset path:       {DATASET_PATH}")
print("=" * 60)

# Verify dataset exists
if os.path.exists(DATASET_PATH):
    gesture_dirs = [d for d in os.listdir(DATASET_PATH) 
                    if os.path.isdir(os.path.join(DATASET_PATH, d))]
    total_videos = sum(
        len([f for f in os.listdir(os.path.join(DATASET_PATH, d)) 
             if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        for d in gesture_dirs
    )
    print(f"\n✅ Dataset found: {len(gesture_dirs)} classes, ~{total_videos} videos")
else:
    print(f"\n❌ Dataset not found at: {DATASET_PATH}")
    print("   Please upload your dataset folder!")

# Create output directories
create_directories()
print("✅ Output directories created/verified")

# %% Cell 4: Check GPU
# ============================================================================
# CELL 4: CHECK GPU AVAILABILITY
# ============================================================================

import tensorflow as tf

print("=" * 60)
print("HARDWARE CHECK")
print("=" * 60)
print(f"  TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  ✅ GPU available: {len(gpus)}")
    for gpu in gpus:
        print(f"     → {gpu}")
    # Enable memory growth to prevent OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("  ⚠️ No GPU — training on CPU (will be slower)")
    print("     💡 Go to Runtime → Change runtime type → GPU")

print("=" * 60)

# %% Cell 5: Data Preprocessing
# ============================================================================
# CELL 5: DATA PREPROCESSING — Extract Dual-Hand Landmarks
# ============================================================================
# This extracts MediaPipe landmarks from ALL gesture videos using BOTH hands.
# Output: preprocessed_data/X_train.npy, X_test.npy, y_train.npy, y_test.npy

print("\n" + "=" * 60)
print("STEP 1: DATA PREPROCESSING (Dual-Hand)")
print("=" * 60)

from data_preprocessing import process_dataset, save_preprocessed_data

# Process all videos — extract dual-hand landmarks
X, y, label_encoder = process_dataset()

# Split and save
X_train, X_test, y_train, y_test = save_preprocessed_data(X, y, label_encoder)

print(f"\n✅ Preprocessing complete!")
print(f"   Training: {X_train.shape} → {X_train.shape[0]} samples × {X_train.shape[1]} frames × {X_train.shape[2]} features")
print(f"   Test:     {X_test.shape}")

# %% Cell 6: Data Augmentation
# ============================================================================
# CELL 6: DATA AUGMENTATION — Generate Augmented Samples
# ============================================================================
# Augments training data with noise, rotation, scaling, flipping, etc.
# Critical for small datasets — creates AUGMENTATION_FACTOR copies per sample.

print("\n" + "=" * 60)
print("STEP 2: DATA AUGMENTATION (Dual-Hand)")
print("=" * 60)

import numpy as np
from augment_data import augment_dataset, save_augmented_data

# Load preprocessed training data
X_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "y_train.npy"))

print(f"[LOADED] Training data: X={X_train.shape}, y={y_train.shape}")

# Augment
X_aug, y_aug = augment_dataset(X_train, y_train)

# Save
save_augmented_data(X_aug, y_aug)

print(f"\n✅ Augmentation complete!")
print(f"   Original:  {X_train.shape[0]} samples")
print(f"   Augmented: {X_aug.shape[0]} samples ({X_aug.shape[0]/X_train.shape[0]:.1f}x increase)")

# %% Cell 7: Model Training
# ============================================================================
# CELL 7: MODEL TRAINING — Train CNN-LSTM
# ============================================================================
# Trains the dual-hand gesture recognition model.
# Uses augmented data with class weights and learning rate scheduling.

print("\n" + "=" * 60)
print("STEP 3: MODEL TRAINING")
print("=" * 60)

from train import train_model

model, history = train_model()

print(f"\n✅ Training complete!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"   Model saved to: {MODEL_SAVE_PATH}")

# %% Cell 8: Model Evaluation
# ============================================================================
# CELL 8: MODEL EVALUATION — Generate Metrics & Plots
# ============================================================================
# Generates confusion matrix, per-class accuracy, training curves, etc.

print("\n" + "=" * 60)
print("STEP 4: MODEL EVALUATION")
print("=" * 60)

from evaluate import run_evaluation

run_evaluation()

print(f"\n✅ Evaluation complete!")
print(f"   Results saved to: {EVALUATION_PATH}")

# %% Cell 9: Display Results
# ============================================================================
# CELL 9: DISPLAY EVALUATION RESULTS
# ============================================================================

import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

eval_images = [
    "training_curves.png",
    "confusion_matrix.png",
    "per_class_accuracy.png",
    "per_class_f1_scores.png",
    "prediction_confidence.png",
]

for img_name in eval_images:
    img_path = os.path.join(EVALUATION_PATH, img_name)
    if os.path.exists(img_path):
        print(f"\n📊 {img_name}")
        try:
            display(IPImage(filename=img_path))
        except Exception:
            img = plt.imread(img_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_name)
            plt.show()

# Print classification report
report_path = os.path.join(EVALUATION_PATH, "classification_report.txt")
if os.path.exists(report_path):
    print("\n📋 CLASSIFICATION REPORT:")
    with open(report_path, "r") as f:
        print(f.read())

# %% Cell 10: Download Model
# ============================================================================
# CELL 10: DOWNLOAD TRAINED MODEL
# ============================================================================
# Download the trained model and files for local inference.

print("=" * 60)
print("DOWNLOAD FILES FOR LOCAL APP")
print("=" * 60)

files_to_download = [
    os.path.join(MODEL_SAVE_PATH, "gesture_model.h5"),
    os.path.join(MODEL_SAVE_PATH, "gesture_model_final.h5"),
    os.path.join(MODEL_SAVE_PATH, "training_history.json"),
    os.path.join(MODEL_SAVE_PATH, "model_config.json"),
    os.path.join(PREPROCESSED_DATA_PATH, "label_encoder.pkl"),
    os.path.join(PREPROCESSED_DATA_PATH, "class_names.txt"),
]

print("\n📥 Files to download for local app:")
for fpath in files_to_download:
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  ✅ {fpath} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ {fpath} — not found")

# If in Colab, offer direct download
if IN_COLAB:
    from google.colab import files
    print("\n💡 To download files, uncomment and run:")
    print("   # files.download('models/gesture_model.h5')")
    print("   # files.download('preprocessed_data/label_encoder.pkl')")

print(f"\n{'='*60}")
print("🎉 PIPELINE COMPLETE!")
print(f"{'='*60}")
print("Next steps:")
print("  1. Download gesture_model.h5 and label_encoder.pkl")
print("  2. Place them in your local project folder")
print("  3. Run: streamlit run app.py")
print("  4. Use both hands to perform gestures! 🤲")
