"""
=============================================================================
Data Preprocessing Script — Run on Google Colab
Extracts MediaPipe DUAL-HAND landmarks from gesture videos and saves as NumPy arrays.
=============================================================================

Usage (Colab):
    1. Upload dataset/ folder to Colab or mount Google Drive
    2. Run: python data_preprocessing.py
    3. Output: preprocessed_data/ folder with .npy files + label_encoder.pkl
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    DATASET_PATH, PREPROCESSED_DATA_PATH, SEQUENCE_LENGTH,
    NUM_HANDS, LANDMARKS_PER_HAND, NUM_LANDMARKS, NUM_COORDS,
    FEATURES_PER_HAND, NUM_FEATURES, MAX_HAND_DETECTION_FAILURES,
    TEST_SPLIT, RANDOM_SEED, GESTURE_CLASSES, create_directories
)


def initialize_mediapipe():
    """Initialize MediaPipe Hands solution for BOTH hands."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,           # Detect BOTH hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands


def extract_landmarks_from_frame(frame, hands):
    """
    Extract hand landmarks from a single frame using BOTH hands.
    
    Args:
        frame: BGR image from OpenCV
        hands: MediaPipe Hands object
    
    Returns:
        numpy array of shape (126,) with landmark coordinates for both hands,
        or None if no hands detected.
        Format: [left_hand_63_features | right_hand_63_features]
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Initialize both hands as zeros
    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    
    # Track which hands we found
    hands_found = []
    
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Determine handedness (Left/Right)
        if results.multi_handedness and idx < len(results.multi_handedness):
            handedness = results.multi_handedness[idx].classification[0].label
        else:
            # Fallback: sort by x-position (leftmost = Left hand in mirrored view)
            avg_x = np.mean([lm.x for lm in hand_landmarks.landmark])
            handedness = "Left" if avg_x > 0.5 else "Right"
        
        if handedness == "Left":
            left_hand = landmarks
        else:
            right_hand = landmarks
        
        hands_found.append(handedness)
    
    # If only one hand detected and it's ambiguous, use position-based sorting
    if len(results.multi_hand_landmarks) == 1 and len(hands_found) == 1:
        # Single hand goes to whichever slot was assigned
        pass
    
    # Concatenate: [left_hand (63) | right_hand (63)] = 126 features
    combined = np.concatenate([left_hand, right_hand])
    
    return combined


def normalize_landmarks(landmarks_sequence):
    """
    Normalize landmarks relative to each hand's bounding box.
    Centers landmarks and scales to [0, 1] range.
    Normalizes each hand independently.
    
    Args:
        landmarks_sequence: numpy array of shape (seq_len, 126)
    
    Returns:
        Normalized landmarks of same shape
    """
    normalized = landmarks_sequence.copy()
    
    for i in range(len(normalized)):
        if np.all(normalized[i] == 0):
            continue
        
        # Process each hand independently
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_features = normalized[i, start:end]
            
            # Skip if this hand has no data (all zeros)
            if np.all(hand_features == 0):
                continue
            
            # Reshape to (21, 3) for this hand
            frame_landmarks = hand_features.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            
            # Get x, y, z separately
            x_coords = frame_landmarks[:, 0]
            y_coords = frame_landmarks[:, 1]
            z_coords = frame_landmarks[:, 2]
            
            # Center and normalize by range
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            
            if x_range > 0:
                frame_landmarks[:, 0] = (x_coords - x_coords.min()) / x_range
            if y_range > 0:
                frame_landmarks[:, 1] = (y_coords - y_coords.min()) / y_range
            
            # Z-coordinate: normalize by max absolute value
            z_max = np.abs(z_coords).max()
            if z_max > 0:
                frame_landmarks[:, 2] = z_coords / z_max
            
            normalized[i, start:end] = frame_landmarks.flatten()
    
    return normalized


def extract_landmarks_from_video(video_path, hands):
    """
    Extract and sample dual-hand landmarks from a video file.
    
    Args:
        video_path: Path to the video file
        hands: MediaPipe Hands object
    
    Returns:
        numpy array of shape (SEQUENCE_LENGTH, 126) or None if too many failures
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  [WARNING] Cannot open video: {video_path}")
        return None
    
    all_landmarks = []
    total_frames = 0
    failed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        landmarks = extract_landmarks_from_frame(frame, hands)
        
        if landmarks is not None:
            all_landmarks.append(landmarks)
        else:
            failed_frames += 1
            all_landmarks.append(np.zeros(NUM_FEATURES, dtype=np.float32))
    
    cap.release()
    
    if total_frames == 0:
        print(f"  [WARNING] Empty video: {video_path}")
        return None
    
    # Check if too many frames failed hand detection
    failure_rate = failed_frames / total_frames
    if failure_rate > MAX_HAND_DETECTION_FAILURES:
        print(f"  [WARNING] High failure rate ({failure_rate:.1%}) for: {video_path}")
        # Still process it but warn — don't skip (small dataset)
    
    all_landmarks = np.array(all_landmarks)
    
    # Sample or pad to SEQUENCE_LENGTH frames
    if len(all_landmarks) >= SEQUENCE_LENGTH:
        # Uniform sampling
        indices = np.linspace(0, len(all_landmarks) - 1, SEQUENCE_LENGTH, dtype=int)
        sampled = all_landmarks[indices]
    else:
        # Pad with last frame (or zeros if empty)
        padding_needed = SEQUENCE_LENGTH - len(all_landmarks)
        if len(all_landmarks) > 0:
            last_frame = all_landmarks[-1:]
            padding = np.repeat(last_frame, padding_needed, axis=0)
        else:
            padding = np.zeros((padding_needed, NUM_FEATURES), dtype=np.float32)
        sampled = np.vstack([all_landmarks, padding])
    
    # Normalize landmarks
    sampled = normalize_landmarks(sampled)
    
    return sampled


def process_dataset():
    """
    Process entire dataset: extract dual-hand landmarks from all videos.
    
    Returns:
        X: numpy array of shape (num_samples, SEQUENCE_LENGTH, NUM_FEATURES)
        y: numpy array of labels
        label_encoder: Fitted LabelEncoder
    """
    X = []
    y = []
    
    hands = initialize_mediapipe()
    
    # Get valid gesture directories (skip empty ones)
    gesture_dirs = sorted([
        d for d in os.listdir(DATASET_PATH) 
        if os.path.isdir(os.path.join(DATASET_PATH, d))
        and len(os.listdir(os.path.join(DATASET_PATH, d))) > 0
    ])
    
    print(f"[INFO] Found {len(gesture_dirs)} gesture classes")
    print(f"[INFO] Classes: {gesture_dirs}")
    print(f"[INFO] Using DUAL-HAND detection (max_num_hands=2)")
    print(f"[INFO] Features per frame: {NUM_FEATURES} (2 hands × 21 landmarks × 3 coords)")
    
    for gesture_name in gesture_dirs:
        gesture_path = os.path.join(DATASET_PATH, gesture_name)
        video_files = sorted([
            f for f in os.listdir(gesture_path) 
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])
        
        print(f"\n[PROCESSING] '{gesture_name}' — {len(video_files)} videos")
        
        for video_file in video_files:
            video_path = os.path.join(gesture_path, video_file)
            landmarks = extract_landmarks_from_video(video_path, hands)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(gesture_name)
                print(f"  ✓ {video_file} — shape: {landmarks.shape}")
            else:
                print(f"  ✗ {video_file} — SKIPPED")
    
    hands.close()
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n{'='*60}")
    print(f"[RESULT] Total samples: {len(X)}")
    print(f"[RESULT] X shape: {X.shape}")
    print(f"[RESULT] Features per frame: {NUM_FEATURES} (dual-hand)")
    print(f"[RESULT] Classes: {len(label_encoder.classes_)}")
    print(f"[RESULT] Class distribution:")
    for cls in label_encoder.classes_:
        count = np.sum(y == cls)
        print(f"  {cls}: {count} samples")
    print(f"{'='*60}")
    
    return X, y_encoded, label_encoder


def save_preprocessed_data(X, y, label_encoder):
    """Split data into train/test and save as .npy files."""
    create_directories()
    
    # Train-test split (disabled stratify due to small dataset size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT, 
        random_state=RANDOM_SEED
    )
    
    # Save arrays
    np.save(os.path.join(PREPROCESSED_DATA_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(PREPROCESSED_DATA_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(PREPROCESSED_DATA_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(PREPROCESSED_DATA_PATH, "y_test.npy"), y_test)
    
    # Save label encoder
    with open(os.path.join(PREPROCESSED_DATA_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    # Save class names for easy reference
    with open(os.path.join(PREPROCESSED_DATA_PATH, "class_names.txt"), "w") as f:
        for cls in label_encoder.classes_:
            f.write(f"{cls}\n")
    
    print(f"\n[SAVED] Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"[SAVED] Test set:     X={X_test.shape}, y={y_test.shape}")
    print(f"[SAVED] Label encoder: {len(label_encoder.classes_)} classes")
    print(f"[SAVED] All files saved to: {PREPROCESSED_DATA_PATH}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GESTURE DATA PREPROCESSING — Dual-Hand MediaPipe Extraction")
    print("=" * 60)
    
    X, y, label_encoder = process_dataset()
    X_train, X_test, y_train, y_test = save_preprocessed_data(X, y, label_encoder)
    
    print("\n[DONE] Preprocessing complete! Next step: run augment_data.py")
