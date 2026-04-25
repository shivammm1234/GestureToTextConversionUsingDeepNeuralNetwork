"""
=============================================================================
Model Architecture — CNN-LSTM for Dual-Hand Gesture Recognition
Uses TimeDistributed Conv1D for spatial features + Plain LSTM for temporal.
Input: (batch, 30, 126) — 42 landmarks × 3 coords (both hands)
=============================================================================

Usage (Colab):
    from model_architecture import build_gesture_model
    model = build_gesture_model(num_classes=56)
    model.summary()
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, LSTM,
    BatchNormalization, GlobalAveragePooling1D,
    TimeDistributed, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from config import (
    SEQUENCE_LENGTH, NUM_LANDMARKS, LANDMARKS_PER_HAND, NUM_COORDS,
    NUM_FEATURES, NUM_HANDS, FEATURES_PER_HAND,
    CNN_FILTERS, LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE
)


def build_gesture_model(num_classes, summary=True):
    """
    Build CNN-LSTM model for dual-hand gesture recognition from landmarks.
    
    Architecture:
        Input (30, 126) → Reshape per frame to (42, 3)
        → TimeDistributed Conv1D layers (spatial features from both hands)
        → LSTM layers (temporal modeling)
        → Dense + Softmax (classification)
    
    Args:
        num_classes: Number of gesture classes
        summary: Whether to print model summary
    
    Returns:
        Compiled Keras Model
    """
    
    # =========================================================================
    # INPUT LAYER
    # =========================================================================
    inputs = Input(
        shape=(SEQUENCE_LENGTH, NUM_FEATURES), 
        name="landmark_sequence_input"
    )
    
    # =========================================================================
    # SPATIAL FEATURE EXTRACTION (CNN)
    # Each frame's 126 features → reshape to (42, 3) → Conv1D
    # =========================================================================
    
    # Reshape each timestep: (batch, 30, 126) → (batch, 30, 42, 3)
    x = TimeDistributed(
        Reshape((NUM_LANDMARKS, NUM_COORDS)), 
        name="reshape_to_landmarks"
    )(inputs)
    
    # Conv1D Block 1: Extract local finger/joint patterns
    x = TimeDistributed(
        Conv1D(CNN_FILTERS[0], kernel_size=3, activation='relu', padding='same'),
        name="conv1d_block1"
    )(x)
    x = TimeDistributed(
        BatchNormalization(), 
        name="bn_block1"
    )(x)
    
    # Conv1D Block 2: Extract higher-level hand shape features
    x = TimeDistributed(
        Conv1D(CNN_FILTERS[1], kernel_size=3, activation='relu', padding='same'),
        name="conv1d_block2"
    )(x)
    x = TimeDistributed(
        BatchNormalization(), 
        name="bn_block2"
    )(x)
    
    # Global Average Pooling per frame
    x = TimeDistributed(
        GlobalAveragePooling1D(), 
        name="spatial_pooling"
    )(x)
    
    # =========================================================================
    # TEMPORAL MODELING (LSTM) — Plain LSTM (not bidirectional)
    # =========================================================================
    
    # LSTM Layer 1: Final temporal encoding
    x = LSTM(
        LSTM_UNITS[0], 
        return_sequences=False, 
        name="lstm_layer_1"
    )(x)
    x = Dropout(0.4, name="lstm_dropout_1")(x)
    
    # =========================================================================
    # CLASSIFICATION HEAD
    # =========================================================================
    x = Dense(DENSE_UNITS, activation='relu', name="dense_1")(x)
    x = Dropout(DROPOUT_RATE, name="dropout_head")(x)
    
    outputs = Dense(num_classes, activation='softmax', name="gesture_output")(x)
    
    # =========================================================================
    # BUILD & COMPILE
    # =========================================================================
    model = Model(inputs=inputs, outputs=outputs, name="GestureCNN_LSTM")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if summary:
        print("\n" + "=" * 60)
        print("GESTURE CNN-LSTM MODEL ARCHITECTURE")
        print("=" * 60)
        model.summary()
        print(f"\nInput shape:  (batch, {SEQUENCE_LENGTH}, {NUM_FEATURES})")
        print(f"Output shape: (batch, {num_classes})")
        print(f"CNN filters:  {CNN_FILTERS}")
        print(f"LSTM units:   {LSTM_UNITS}")
        print(f"Total params: {model.count_params():,}")
    
    return model


def get_model_config():
    """Return model configuration as a dictionary (for logging)."""
    return {
        "sequence_length": SEQUENCE_LENGTH,
        "num_landmarks": NUM_LANDMARKS,
        "num_coords": NUM_COORDS,
        "num_features": NUM_FEATURES,
        "cnn_filters": CNN_FILTERS,
        "lstm_units": LSTM_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "architecture": "TimeDistributed_Conv1D + Plain_LSTM"
    }


# =============================================================================
# MAIN — Test model creation
# =============================================================================
if __name__ == "__main__":
    from config import NUM_CLASSES
    
    model = build_gesture_model(num_classes=NUM_CLASSES, summary=True)
    
    # Test with dummy data
    import numpy as np
    dummy_input = np.random.randn(2, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"\n[TEST] Dummy input shape:  {dummy_input.shape}")
    print(f"[TEST] Dummy output shape: {dummy_output.shape}")
    print(f"[TEST] Output sums to 1:   {np.allclose(dummy_output.sum(axis=1), 1.0)}")
    print(f"\n[OK] Model builds and runs successfully!")
