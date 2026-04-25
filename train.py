"""
=============================================================================
Model Training Script — Run on Google Colab
Trains the CNN-LSTM gesture recognition model on augmented landmark data.
=============================================================================

Usage (Colab):
    1. Run data_preprocessing.py first
    2. Run augment_data.py second
    3. Run: python train.py
    4. Output: models/ folder with gesture_model.h5 + training history

ORDER OF EXECUTION ON COLAB:
    Step 1: python data_preprocessing.py
    Step 2: python augment_data.py
    Step 3: python train.py
    Step 4: python evaluate.py
"""

import os
import json
import numpy as np
import pickle
from collections import Counter

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

from config import (
    PREPROCESSED_DATA_PATH, AUGMENTED_DATA_PATH, MODEL_SAVE_PATH,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, RANDOM_SEED,
    NUM_CLASSES, create_directories
)
from model_architecture import build_gesture_model, get_model_config


def load_training_data(use_augmented=True):
    """
    Load training data (augmented or original) and test data.
    
    Args:
        use_augmented: Whether to use augmented training data
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    if use_augmented and os.path.exists(os.path.join(AUGMENTED_DATA_PATH, "X_train_augmented.npy")):
        print("[INFO] Loading AUGMENTED training data...")
        X_train = np.load(os.path.join(AUGMENTED_DATA_PATH, "X_train_augmented.npy"))
        y_train = np.load(os.path.join(AUGMENTED_DATA_PATH, "y_train_augmented.npy"))
    else:
        print("[INFO] Loading ORIGINAL training data (no augmentation)...")
        X_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "X_train.npy"))
        y_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "y_train.npy"))
    
    X_test = np.load(os.path.join(PREPROCESSED_DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PREPROCESSED_DATA_PATH, "y_test.npy"))
    
    print(f"[INFO] Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test data:     X={X_test.shape}, y={y_test.shape}")
    print(f"[INFO] Unique classes in train: {len(np.unique(y_train))}")
    print(f"[INFO] Unique classes in test:  {len(np.unique(y_test))}")
    
    return X_train, y_train, X_test, y_test


def compute_class_weights(y_train):
    """
    Compute class weights to handle imbalanced dataset.
    Classes with fewer samples get higher weights.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"\n[INFO] Class weights computed for {len(classes)} classes")
    print(f"[INFO] Weight range: {min(weights):.3f} - {max(weights):.3f}")
    
    return class_weight_dict


def get_callbacks(model_save_path):
    """
    Define training callbacks for optimal training.
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, "gesture_model.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Log training history to CSV
        CSVLogger(
            os.path.join(model_save_path, "training_log.csv"),
            separator=',',
            append=False
        )
    ]
    
    return callbacks


def train_model():
    """Main training pipeline."""
    
    print("=" * 60)
    print("GESTURE RECOGNITION MODEL TRAINING")
    print("=" * 60)
    
    # Set random seed for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create directories
    create_directories()
    
    # Load data
    X_train, y_train, X_test, y_test = load_training_data(use_augmented=True)
    
    # Determine number of classes from data
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"\n[INFO] Number of classes: {num_classes}")
    
    # Build model
    print("\n[BUILDING] Model architecture...")
    model = build_gesture_model(num_classes=num_classes, summary=True)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Get callbacks
    callbacks = get_callbacks(MODEL_SAVE_PATH)
    
    # Print training configuration
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Validation split:  {VALIDATION_SPLIT}")
    print(f"  Learning rate:     {model.optimizer.learning_rate.numpy():.6f}")
    print(f"  Training samples:  {len(X_train)}")
    print(f"  Test samples:      {len(X_test)}")
    print(f"{'='*60}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n[INFO] GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"  → {gpu}")
    else:
        print("  → Training on CPU (will be slower)")
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING...")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Save final model (backup — best model already saved by checkpoint)
    final_model_path = os.path.join(MODEL_SAVE_PATH, "gesture_model_final.h5")
    model.save(final_model_path)
    print(f"\n[SAVED] Final model: {final_model_path}")
    
    # Save training history as JSON
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    history_path = os.path.join(MODEL_SAVE_PATH, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[SAVED] Training history: {history_path}")
    
    # Save model config
    config = get_model_config()
    config["num_classes"] = num_classes
    config["total_training_samples"] = len(X_train)
    config["total_test_samples"] = len(X_test)
    config["total_epochs_trained"] = len(history.history['loss'])
    config["final_train_accuracy"] = float(history.history['accuracy'][-1])
    config["final_val_accuracy"] = float(history.history['val_accuracy'][-1])
    config["best_val_accuracy"] = float(max(history.history['val_accuracy']))
    
    config_path = os.path.join(MODEL_SAVE_PATH, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Model config: {config_path}")
    
    # =========================================================================
    # TRAINING SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total epochs trained:   {config['total_epochs_trained']}")
    print(f"  Final train accuracy:   {config['final_train_accuracy']:.4f}")
    print(f"  Final val accuracy:     {config['final_val_accuracy']:.4f}")
    print(f"  Best val accuracy:      {config['best_val_accuracy']:.4f}")
    print(f"  Final train loss:       {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss:         {history.history['val_loss'][-1]:.4f}")
    print(f"{'='*60}")
    print(f"\n[NEXT] Run evaluate.py to generate evaluation metrics and plots")
    
    return model, history


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    model, history = train_model()
