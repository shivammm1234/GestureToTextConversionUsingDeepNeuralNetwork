"""
=============================================================================
Streamlit App — Real-Time ASL Gesture Recognition
Webcam → MediaPipe Landmarks → CNN-LSTM → Prediction
=============================================================================

Usage:
    streamlit run app.py
"""

import os
import cv2
import time
import pickle
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from collections import deque
from nlp_module import GestureToSentence
from model_architecture import build_gesture_model

# Import config constants
from config import (
    SEQUENCE_LENGTH, NUM_FEATURES, NUM_HANDS, 
    LANDMARKS_PER_HAND, NUM_COORDS, FEATURES_PER_HAND
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ASL Gesture-to-Text | MediaPipe + CNN-LSTM",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — Dark Premium Theme
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding: 10px 0;
    }

    .sub-title {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0a1628 100%);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 210, 255, 0.15);
        margin: 10px 0;
    }

    .prediction-label {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d2ff;
        text-transform: uppercase;
        letter-spacing: 3px;
    }

    .confidence-text {
        font-size: 1.1rem;
        color: #8892b0;
        margin-top: 8px;
    }

    .sentence-card {
        background: linear-gradient(135deg, #1a3a1a 0%, #0a1628 100%);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(46, 204, 113, 0.15);
        margin: 10px 0;
    }

    .sentence-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2ecc71;
        font-style: italic;
    }

    .status-active {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #2ecc71;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
        50%       { opacity: 0.8; box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
    }

    .history-item {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 15px;
        margin: 5px 0;
        color: #ccd6f6;
        border-left: 3px solid #7b2ff7;
    }

    .word-chip {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .progress-bar-outer {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        height: 10px;
        margin: 8px auto;
        max-width: 320px;
        overflow: hidden;
    }
    /* Interactive Pill Buttons Styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #7b2ff7 0%, #00d2ff 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 5px 15px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.3);
    }

    .stButton > button[kind="secondary"]:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(123, 47, 247, 0.5);
        background: linear-gradient(135deg, #ff6b6b 0%, #7b2ff7 100%);
    }

    /* Word Buffer Container */
    .word-buffer-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MEDIAPIPE LOGIC
# =============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_landmarks_from_frame(frame, hands_model):
    """Extract hand landmarks from frame (dual-hand)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)
    
    # Initialize both hands as zeros
    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Determine handedness
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
            else:
                avg_x = np.mean([lm.x for lm in hand_landmarks.landmark])
                handedness = "Left" if avg_x > 0.5 else "Right"
            
            if handedness == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks
                
    return np.concatenate([left_hand, right_hand]), results

def normalize_landmarks(landmarks_sequence):
    """Normalize landmarks sequence (seq_len, 126)."""
    normalized = landmarks_sequence.copy()
    for i in range(len(normalized)):
        if np.all(normalized[i] == 0): continue
        for hand_idx in range(NUM_HANDS):
            start = hand_idx * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            hand_features = normalized[i, start:end]
            if np.all(hand_features == 0): continue
            
            frame_landmarks = hand_features.reshape(LANDMARKS_PER_HAND, NUM_COORDS)
            x_coords, y_coords = frame_landmarks[:, 0], frame_landmarks[:, 1]
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            
            if x_range > 0: frame_landmarks[:, 0] = (x_coords - x_coords.min()) / x_range
            if y_range > 0: frame_landmarks[:, 1] = (y_coords - y_coords.min()) / y_range
            
            z_max = np.abs(frame_landmarks[:, 2]).max()
            if z_max > 0: frame_landmarks[:, 2] = frame_landmarks[:, 2] / z_max
            
            normalized[i, start:end] = frame_landmarks.flatten()
    return normalized

# =============================================================================
# LOAD RESOURCES
# =============================================================================
MODEL_DIR = "slr_models-20260416T061651Z-3-001/slr_models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model_resumed_final.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

@st.cache_resource
def load_resources():
    """Load Keras model, LabelEncoder, and MediaPipe Hands."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None, f"Model not found at {MODEL_PATH}"
    if not os.path.exists(LABEL_ENCODER_PATH):
        return None, None, None, None, f"Label Encoder not found at {LABEL_ENCODER_PATH}"

    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load weights into a manually rebuilt architecture
        # This avoids Keras 3 (Colab) vs Keras 2 (Local) deserialization issues
        model = build_gesture_model(num_classes=len(label_encoder.classes_), summary=False)
        try:
            model.load_weights(MODEL_PATH)
        except Exception:
            # Fallback to .h5 if .keras weight loading fails in Keras 2
            H5_PATH = MODEL_PATH.replace(".keras", ".h5")
            if os.path.exists(H5_PATH):
                model.load_weights(H5_PATH)
            else:
                # If resumed final h5 doesn't exist, try gesture_model_final.h5
                ALT_H5 = os.path.join(MODEL_DIR, "gesture_model_final.h5")
                model.load_weights(ALT_H5)
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        nlp = GestureToSentence()
        
        return model, label_encoder, hands, nlp, None
    except Exception as e:
        return None, None, None, None, f"Error loading resources: {str(e)}"

# =============================================================================
# INFERENCE HELPER
# =============================================================================
def run_inference(model, landmark_buffer, label_encoder):
    """Run model on buffered landmarks."""
    seq = np.array(landmark_buffer)  # (SEQUENCE_LENGTH, 126)
    seq = normalize_landmarks(seq)
    seq = np.expand_dims(seq, axis=0)  # (1, 30, 126)
    
    prediction = model.predict(seq, verbose=0)[0]
    pred_idx = np.argmax(prediction)
    confidence = prediction[pred_idx]
    label = label_encoder.inverse_transform([pred_idx])[0]
    
    return label, confidence

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown('<h1 class="main-title">🤟 ASL Gesture-to-Text</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-title">MediaPipe Landmarks + CNN-LSTM • Real-Time Recognition</p>',
        unsafe_allow_html=True
    )

    model, label_encoder, hands, nlp, error = load_resources()

    if error:
        st.error(f"⚠️ {error}")
        st.info(f"Ensure your model files are in the `{MODEL_DIR}` folder.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.4, 0.95, 0.65, 0.05)
        cooldown_frames = st.slider("Prediction Cooldown (2-3s recommended)", 10, 150, 60, 5)
        show_landmarks = st.checkbox("Show Landmarks Overlay", value=True)
        
        st.markdown("---")
        st.markdown("## 📊 Model Info")
        st.markdown("**Architecture:** CNN-LSTM")
        st.markdown(f"**Sequence:** {SEQUENCE_LENGTH} frames")
        st.markdown(f"**Features:** {NUM_FEATURES} (Dual-Hand)")
        st.markdown(f"**Classes:** {len(label_encoder.classes_)}")
        
        st.markdown("---")
        st.markdown("## 🔤 Recognized Signs")
        st.write(", ".join(label_encoder.classes_))

    # Layout
    col_cam, col_output = st.columns([3, 2])

    with col_output:
        prediction_placeholder = st.empty()
        st.markdown("### ✏️ Gesture Words")
        if "gesture_text" not in st.session_state: st.session_state.gesture_text = ""
        gesture_text_input = st.text_area("Recognized gestures:", value=st.session_state.gesture_text, height=80, key="gt_area")
        if gesture_text_input != st.session_state.gesture_text: st.session_state.gesture_text = gesture_text_input

        sentence_placeholder = st.empty()
        
        if st.button("✨ Generate Sentence", use_container_width=True, type="secondary"):
            if st.session_state.word_buffer:
                st.session_state.current_sentence = nlp.construct_sentence(st.session_state.word_buffer) if nlp else " ".join(st.session_state.word_buffer)
                # Also update history
                if st.session_state.current_sentence not in st.session_state.sentence_history:
                    st.session_state.sentence_history.append(st.session_state.current_sentence)
                st.rerun()
            else:
                st.warning("Detection buffer is empty! Please sign some gestures first.")

        buffer_placeholder = st.empty()
        st.markdown("### 📝 Sentence History")
        history_placeholder = st.empty()

        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("🗑️ Clear Words", use_container_width=True):
            st.session_state.word_buffer = []
            st.session_state.gesture_text = ""
            st.session_state.current_sentence = ""
            st.rerun()

        if col_b2.button("⬅️ Undo Word", use_container_width=True):
            if st.session_state.word_buffer:
                st.session_state.word_buffer.pop()
                st.session_state.gesture_text = " ".join(st.session_state.word_buffer)
                st.rerun()

        if col_b3.button("🧹 Clear History", use_container_width=True):
            st.session_state.sentence_history = []
            st.rerun()



    with col_cam:
        st.markdown("### 📹 Live Webcam Feed")
        start_btn = st.button("▶️ Start Camera", type="primary", use_container_width=True)
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
        frame_placeholder = st.empty()
        buf_progress = st.empty()

    # Session State
    defaults = {
        "landmark_buffer": deque(maxlen=SEQUENCE_LENGTH),
        "word_buffer": [],
        "sentence_history": [],
        "cooldown_counter": 0,
        "camera_active": False,
        "current_sentence": "",
        "gesture_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    
    # Manual Text Area Sync: Update word_buffer if user types
    if st.session_state.gt_area != st.session_state.gesture_text:
        st.session_state.gesture_text = st.session_state.gt_area
        st.session_state.word_buffer = [w.strip() for w in st.session_state.gesture_text.split() if w.strip()]

    if start_btn: st.session_state.camera_active = True
    if stop_btn: st.session_state.camera_active = False

    # Webcam Loop
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Landmark Extraction
            landmarks, results = extract_landmarks_from_frame(frame, hands)
            st.session_state.landmark_buffer.append(landmarks)
            
            # Overlay Landmarks
            if show_landmarks and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Inference
            curr_label, curr_conf = None, 0.0
            buf_len = len(st.session_state.landmark_buffer)
            if buf_len == SEQUENCE_LENGTH and st.session_state.cooldown_counter <= 0:
                # Visibility check: Count frames where hands are actually detected
                # (A frame is non-empty if it contains any non-zero landmark data)
                visible_frames = sum([1 for f in st.session_state.landmark_buffer if not np.all(f == 0)])
                visibility_ratio = visible_frames / SEQUENCE_LENGTH

                # Only run inference if hands were visible in at least 50% of the buffer
                if visibility_ratio >= 0.5:
                    curr_label, curr_conf = run_inference(model, st.session_state.landmark_buffer, label_encoder)
                    if curr_conf >= confidence_threshold:
                        if not st.session_state.word_buffer or st.session_state.word_buffer[-1] != curr_label:
                            st.session_state.word_buffer.append(curr_label)
                            st.session_state.gesture_text = " ".join(st.session_state.word_buffer)
                        
                        # Reset buffer for the next gesture
                        st.session_state.landmark_buffer.clear()
                        st.session_state.cooldown_counter = 10 # Small internal debounce only (no UI blocking)

            if st.session_state.cooldown_counter > 0: st.session_state.cooldown_counter -= 1

            # Display
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
            # Show live word buffer
            if st.session_state.word_buffer:
                words_html = " ".join([f'<span class="word-chip">{w}</span>' for w in st.session_state.word_buffer])
                buffer_placeholder.markdown(f'<div class="word-buffer-container">{words_html}</div>', unsafe_allow_html=True)
            elif curr_label:
                prediction_placeholder.markdown(f'<div class="prediction-card"><div class="prediction-label">{curr_label.replace("_"," ")}</div><div class="confidence-text">Confidence: {curr_conf:.1%}</div></div>', unsafe_allow_html=True)
            else:
                prediction_placeholder.markdown('<div class="prediction-card"><div class="prediction-label" style="color:#8892b0;">Watching…</div></div>', unsafe_allow_html=True)

            if st.session_state.current_sentence:
                sentence_placeholder.markdown(f'<div class="sentence-card"><div class="sentence-text">"{st.session_state.current_sentence}"</div></div>', unsafe_allow_html=True)



            if st.session_state.sentence_history:
                html = "".join([f'<div class="history-item">{s}</div>' for s in reversed(st.session_state.sentence_history[-5:])])
                history_placeholder.markdown(html, unsafe_allow_html=True)

        cap.release()
    else:
        prediction_placeholder.markdown('<div class="prediction-card"><div class="prediction-label" style="color:#8892b0;">📸 Camera Off</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
