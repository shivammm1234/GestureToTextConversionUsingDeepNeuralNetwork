"""
=============================================================================
Data Augmentation Script — Run on Google Colab
Augments preprocessed DUAL-HAND landmark data to increase training samples.
=============================================================================

Usage (Colab):
    1. Run data_preprocessing.py first
    2. Run: python augment_data.py
    3. Output: augmented_data/ folder with augmented .npy files
    
This is critical for a small dataset (~357 videos across 56 classes).
Augmentation creates AUGMENTATION_FACTOR copies of each sample with variations.
"""

import os
import numpy as np
import pickle
from collections import Counter
from config import (
    PREPROCESSED_DATA_PATH, AUGMENTED_DATA_PATH,
    NUM_HANDS, LANDMARKS_PER_HAND, NUM_LANDMARKS, NUM_COORDS,
    FEATURES_PER_HAND, NUM_FEATURES, SEQUENCE_LENGTH,
    AUGMENTATION_FACTOR, NOISE_STD, SCALE_RANGE, SHIFT_RANGE,
    TEMPORAL_SHIFT, ROTATION_ANGLE_RANGE, RANDOM_SEED,
    create_directories
)


np.random.seed(RANDOM_SEED)


# =============================================================================
# AUGMENTATION FUNCTIONS (DUAL-HAND AWARE)
# =============================================================================

def add_gaussian_noise(landmarks, std=NOISE_STD):
    """
    Add Gaussian noise to landmark coordinates.
    Simulates natural hand tremor and detection jitter.
    Only adds noise to non-zero (detected) hand features.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)  # 126 features
        std: standard deviation of noise
    Returns:
        Augmented landmarks
    """
    noise = np.random.normal(0, std, landmarks.shape).astype(np.float32)
    augmented = landmarks + noise
    # Don't add noise to zero-padded frames
    zero_mask = np.all(landmarks == 0, axis=1)
    augmented[zero_mask] = 0
    # Also don't add noise to zero-padded hands within frames
    for i in range(len(augmented)):
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            if np.all(landmarks[i, start:end] == 0):
                augmented[i, start:end] = 0
    return augmented


def random_scaling(landmarks, scale_range=SCALE_RANGE):
    """
    Apply random uniform scaling to all coordinates.
    Simulates varying hand distances from camera.
    Scales each detected hand independently.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        scale_range: (min_scale, max_scale)
    Returns:
        Augmented landmarks
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    augmented = landmarks.copy()
    
    for i in range(len(augmented)):
        if np.all(augmented[i] == 0):
            continue
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_data = augmented[i, start:end]
            if np.all(hand_data == 0):
                continue
            frame = hand_data.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            # Scale x, y coordinates (keep z relative)
            center_x = frame[:, 0].mean()
            center_y = frame[:, 1].mean()
            frame[:, 0] = (frame[:, 0] - center_x) * scale + center_x
            frame[:, 1] = (frame[:, 1] - center_y) * scale + center_y
            augmented[i, start:end] = frame.flatten()
    
    return augmented


def random_shift(landmarks, shift_range=SHIFT_RANGE):
    """
    Apply random translation to all landmarks.
    Simulates hand position variation in frame.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        shift_range: maximum shift magnitude
    Returns:
        Augmented landmarks
    """
    shift_x = np.random.uniform(-shift_range, shift_range)
    shift_y = np.random.uniform(-shift_range, shift_range)
    
    augmented = landmarks.copy()
    for i in range(len(augmented)):
        if np.all(augmented[i] == 0):
            continue
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_data = augmented[i, start:end]
            if np.all(hand_data == 0):
                continue
            frame = hand_data.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            frame[:, 0] += shift_x
            frame[:, 1] += shift_y
            augmented[i, start:end] = frame.flatten()
    
    return augmented


def random_rotation_2d(landmarks, angle_range=ROTATION_ANGLE_RANGE):
    """
    Apply random 2D rotation (in x-y plane) to landmarks.
    Simulates hand tilt/rotation. Rotates each hand around its own centroid.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        angle_range: (min_deg, max_deg)
    Returns:
        Augmented landmarks
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    augmented = landmarks.copy()
    for i in range(len(augmented)):
        if np.all(augmented[i] == 0):
            continue
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_data = augmented[i, start:end]
            if np.all(hand_data == 0):
                continue
            frame = hand_data.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            
            # Rotate around centroid
            cx, cy = frame[:, 0].mean(), frame[:, 1].mean()
            x_centered = frame[:, 0] - cx
            y_centered = frame[:, 1] - cy
            
            frame[:, 0] = x_centered * cos_a - y_centered * sin_a + cx
            frame[:, 1] = x_centered * sin_a + y_centered * cos_a + cy
            
            augmented[i, start:end] = frame.flatten()
    
    return augmented


def temporal_jitter(landmarks, max_shift=TEMPORAL_SHIFT):
    """
    Apply random temporal shift to the frame sequence.
    Simulates gesture starting at slightly different times.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        max_shift: maximum number of frames to shift
    Returns:
        Augmented landmarks
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    augmented = np.roll(landmarks, shift, axis=0)
    
    # Zero out the wrapped-around frames
    if shift > 0:
        augmented[:shift] = 0
    elif shift < 0:
        augmented[shift:] = 0
    
    return augmented


def speed_variation(landmarks, speed_factor=None):
    """
    Simulate speed variation by resampling the temporal dimension.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        speed_factor: if None, randomly chosen between 0.8 and 1.2
    Returns:
        Augmented landmarks of same shape
    """
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)
    
    seq_len = len(landmarks)
    new_len = int(seq_len * speed_factor)
    
    if new_len < 2:
        return landmarks.copy()
    
    # Create new indices
    original_indices = np.arange(seq_len)
    new_indices = np.linspace(0, seq_len - 1, new_len)
    
    # Interpolate each feature
    augmented = np.zeros((new_len, landmarks.shape[1]), dtype=np.float32)
    for j in range(landmarks.shape[1]):
        augmented[:, j] = np.interp(new_indices, original_indices, landmarks[:, j])
    
    # Resize back to original sequence length
    if new_len >= SEQUENCE_LENGTH:
        indices = np.linspace(0, new_len - 1, SEQUENCE_LENGTH, dtype=int)
        augmented = augmented[indices]
    else:
        padding = np.zeros((SEQUENCE_LENGTH - new_len, landmarks.shape[1]), dtype=np.float32)
        augmented = np.vstack([augmented, padding])
    
    return augmented


def horizontal_flip(landmarks):
    """
    Mirror landmarks horizontally (flip x-coordinates) AND swap left/right hands.
    Simulates viewing the gesture from the opposite side.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
    Returns:
        Augmented landmarks
    """
    augmented = landmarks.copy()
    for i in range(len(augmented)):
        if np.all(augmented[i] == 0):
            continue
        
        # Swap left and right hand data blocks
        left_hand = augmented[i, :FEATURES_PER_HAND].copy()
        right_hand = augmented[i, FEATURES_PER_HAND:].copy()
        augmented[i, :FEATURES_PER_HAND] = right_hand
        augmented[i, FEATURES_PER_HAND:] = left_hand
        
        # Flip x coordinates for both hands
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_data = augmented[i, start:end]
            if np.all(hand_data == 0):
                continue
            frame = hand_data.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            frame[:, 0] = 1.0 - frame[:, 0]
            augmented[i, start:end] = frame.flatten()
    
    return augmented


def random_finger_dropout(landmarks, dropout_rate=0.1):
    """
    Randomly zero out some landmarks to simulate partial occlusion.
    Keeps wrist landmarks (index 0) for both hands.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
        dropout_rate: probability of dropping each landmark
    Returns:
        Augmented landmarks
    """
    augmented = landmarks.copy()
    for i in range(len(augmented)):
        if np.all(augmented[i] == 0):
            continue
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_data = augmented[i, start:end]
            if np.all(hand_data == 0):
                continue
            frame = hand_data.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            # Randomly drop landmarks (but never drop the wrist — index 0)
            drop_mask = np.random.random(LANDMARKS_PER_HAND) < dropout_rate
            drop_mask[0] = False  # Keep wrist
            frame[drop_mask] = 0
            augmented[i, start:end] = frame.flatten()
    
    return augmented


def augment_single_sample(landmarks):
    """
    Apply a random combination of augmentations to a single sample.
    
    Args:
        landmarks: shape (SEQUENCE_LENGTH, NUM_FEATURES)
    Returns:
        Augmented landmarks
    """
    augmented = landmarks.copy()
    
    # Randomly select which augmentations to apply
    augmentations = [
        (add_gaussian_noise, 0.8),        # 80% chance
        (random_scaling, 0.7),            # 70% chance
        (random_shift, 0.6),              # 60% chance
        (random_rotation_2d, 0.6),        # 60% chance
        (temporal_jitter, 0.4),           # 40% chance
        (speed_variation, 0.4),           # 40% chance
        (horizontal_flip, 0.3),           # 30% chance
        (random_finger_dropout, 0.2),     # 20% chance
    ]
    
    for aug_func, probability in augmentations:
        if np.random.random() < probability:
            augmented = aug_func(augmented)
    
    return augmented


# =============================================================================
# MAIN AUGMENTATION PIPELINE
# =============================================================================

def augment_dataset(X_train, y_train, augmentation_factor=AUGMENTATION_FACTOR):
    """
    Augment the entire training dataset with class-balanced augmentation.
    Classes with fewer samples get more augmentations.
    
    Args:
        X_train: shape (num_samples, SEQUENCE_LENGTH, NUM_FEATURES)
        y_train: shape (num_samples,)
        augmentation_factor: base number of augmented copies per sample
    
    Returns:
        X_augmented, y_augmented (includes originals + augmented)
    """
    print(f"\n{'='*60}")
    print(f"DATA AUGMENTATION (Dual-Hand)")
    print(f"{'='*60}")
    
    # Analyze class distribution
    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    
    print(f"[INFO] Original training samples: {len(X_train)}")
    print(f"[INFO] Features per frame: {NUM_FEATURES} (dual-hand)")
    print(f"[INFO] Class distribution range: {min(class_counts.values())} - {max_count}")
    print(f"[INFO] Base augmentation factor: {augmentation_factor}")
    
    X_augmented = list(X_train.copy())
    y_augmented = list(y_train.copy())
    
    for class_label in sorted(class_counts.keys()):
        class_mask = y_train == class_label
        class_samples = X_train[class_mask]
        class_count = class_counts[class_label]
        
        # More augmentation for underrepresented classes
        balance_factor = max(1, int(max_count / class_count))
        effective_factor = augmentation_factor * balance_factor
        
        # Cap to avoid excessive augmentation
        effective_factor = min(effective_factor, augmentation_factor * 3)
        
        augmented_count = 0
        for sample in class_samples:
            for _ in range(effective_factor):
                aug_sample = augment_single_sample(sample)
                X_augmented.append(aug_sample)
                y_augmented.append(class_label)
                augmented_count += 1
        
        print(f"  Class {class_label:>3d}: {class_count} originals → "
              f"+{augmented_count} augmented (factor={effective_factor})")
    
    X_augmented = np.array(X_augmented, dtype=np.float32)
    y_augmented = np.array(y_augmented)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]
    
    print(f"\n[RESULT] Augmented training set: {X_augmented.shape}")
    print(f"[RESULT] Augmentation ratio: {len(X_augmented)/len(X_train):.1f}x")
    
    # New class distribution
    new_counts = Counter(y_augmented)
    print(f"[RESULT] New distribution range: {min(new_counts.values())} - {max(new_counts.values())}")
    
    return X_augmented, y_augmented


def save_augmented_data(X_aug, y_aug):
    """Save augmented data to disk."""
    create_directories()
    
    np.save(os.path.join(AUGMENTED_DATA_PATH, "X_train_augmented.npy"), X_aug)
    np.save(os.path.join(AUGMENTED_DATA_PATH, "y_train_augmented.npy"), y_aug)
    
    print(f"\n[SAVED] Augmented data saved to: {AUGMENTED_DATA_PATH}")
    print(f"[SAVED] X_train_augmented.npy: {X_aug.shape}")
    print(f"[SAVED] y_train_augmented.npy: {y_aug.shape}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GESTURE DATA AUGMENTATION PIPELINE (Dual-Hand)")
    print("=" * 60)
    
    # Load preprocessed data
    print("\n[LOADING] Preprocessed training data...")
    X_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(PREPROCESSED_DATA_PATH, "y_train.npy"))
    print(f"[LOADED] X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Augment
    X_aug, y_aug = augment_dataset(X_train, y_train)
    
    # Save
    save_augmented_data(X_aug, y_aug)
    
    print("\n[DONE] Augmentation complete! Next step: run train.py")
