"""
=============================================================================
Configuration File for Gesture-to-Text Conversion System
CNN-LSTM + NLP Pipeline
=============================================================================
"""

import os

# =============================================================================
# PATHS
# =============================================================================
DATASET_PATH = "dataset/"
PREPROCESSED_DATA_PATH = "preprocessed_data/"
MODEL_SAVE_PATH = "models/"
EVALUATION_PATH = "evaluation_results/"
AUGMENTED_DATA_PATH = "augmented_data/"

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================
NUM_HANDS = 2               # Use both hands for gesture recognition
LANDMARKS_PER_HAND = 21     # MediaPipe hand landmarks per hand
NUM_LANDMARKS = LANDMARKS_PER_HAND * NUM_HANDS  # 42 total landmarks (both hands)
NUM_COORDS = 3              # x, y, z coordinates per landmark
FEATURES_PER_HAND = LANDMARKS_PER_HAND * NUM_COORDS  # 63 features per hand
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 126 features per frame (both hands)

# =============================================================================
# SEQUENCE / TEMPORAL SETTINGS
# =============================================================================
SEQUENCE_LENGTH = 30        # Number of frames sampled per video
MAX_HAND_DETECTION_FAILURES = 0.5  # Skip video if >50% frames have no hand

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
CNN_FILTERS = [32, 64]              # Conv1D filter sizes (Reduced for preventing overfitting)
LSTM_UNITS = [64]                   # LSTM layer units (Single lightweight layer)
DENSE_UNITS = 128                   # Dense layer before output (Reduced)
DROPOUT_RATE = 0.5                   # Dropout rate
LEARNING_RATE = 0.001                # Initial learning rate

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 16
EPOCHS = 150
VALIDATION_SPLIT = 0.2     # 20% of training data for validation
TEST_SPLIT = 0.20           # 20% of total data for testing (increased for small dataset)
RANDOM_SEED = 42

# =============================================================================
# DATA AUGMENTATION SETTINGS
# =============================================================================
AUGMENTATION_FACTOR = 20    # How many augmented copies per original sample (Massive increase)
NOISE_STD = 0.02            # Gaussian noise standard deviation
SCALE_RANGE = (0.9, 1.1)   # Random scaling range
SHIFT_RANGE = 0.05          # Max coordinate shift
TEMPORAL_SHIFT = 3          # Max frames to shift temporally
ROTATION_ANGLE_RANGE = (-15, 15)  # Rotation angle in degrees

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.6  # Minimum prediction confidence
BUFFER_SIZE = 30             # Frame buffer for real-time inference
PREDICTION_COOLDOWN = 15     # Frames between predictions

# =============================================================================
# GESTURE CLASSES (auto-populated from dataset)
# =============================================================================
GESTURE_CLASSES = [
    "all", "book", "can", "computer", "cool", "deaf", "dog", "drink",
    "family", "fine", "finish", "go", "hearing", "help", "language",
    "later", "like", "many", "mother", "no", "now", "saw", "scream",
    "sea", "shout", "singer", "skip", "sofa", "solve", "something",
    "talent", "telescope", "tempt", "tend", "text", "than", "therefore",
    "thrill", "towel", "truth", "turn", "tv", "unique", "upstairs",
    "vacant", "very", "walk", "water", "waterfall", "weelchair",
    "weigh", "what", "who", "woman", "yes"
]

NUM_CLASSES = len(GESTURE_CLASSES)

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
def create_directories():
    """Create all required output directories."""
    for path in [PREPROCESSED_DATA_PATH, MODEL_SAVE_PATH, 
                 EVALUATION_PATH, AUGMENTED_DATA_PATH]:
        os.makedirs(path, exist_ok=True)
    print(f"[INFO] Directories created/verified.")
    print(f"[INFO] Number of gesture classes: {NUM_CLASSES}")
    print(f"[INFO] Sequence length: {SEQUENCE_LENGTH}")
    print(f"[INFO] Features per frame: {NUM_FEATURES}")
