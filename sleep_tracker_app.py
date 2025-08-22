import streamlit as st
import numpy as np
import tensorflow_hub as hub
import librosa
import tensorflow as tf
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import joblib
import datetime

# ------------------------
# Load models
# ------------------------
@st.cache_resource
def load_models():
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    clf = joblib.load("sneeze_sniff_classifier.pkl")
    return yamnet, clf

yamnet_model, sneeze_sniff_clf = load_models()

# Load class map
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
with open(class_map_path, 'r') as f:
    class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]

sneeze_sniff_classes = ["sneeze", "sniff", "neither"]

# ------------------------
# Event mapping & thresholds
# ------------------------
mapping = {
    "snore": ["Snoring"],
    "cough": ["Cough"],
    "fart": ["Fart"],
    "sleep_talking": ["Speech", "Babbling", "Whimper"],
    "sneeze_sniff": [],  # custom classifier
    "laughter": ["Laughter"],
    "music": ["Music"]
}

# Tuneable thresholds for each class
class_thresholds = {
    "snore": 0.3,
    "cough": 0.3,
    "fart": 0.3,
    "sleep_talking": 0.5,
    "sneeze_sniff": 0.7,  # custom classifier probability
    "laughter": 0.1,
    "music": 0.3
}


# ------------------------
# Recorder
# ------------------------
def record_audio(duration=5, sr=16000):
    st.write(f"🎤 Recording {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten(), sr

# ------------------------
# Predict sneeze/sniff
# ------------------------
def predict_sneeze_sniff(chunk, threshold=0.7):
    _, embeddings, _ = yamnet_model(chunk)
    mean_embedding = np.mean(embeddings.numpy(), axis=0)
    probs = sneeze_sniff_clf.predict_proba([mean_embedding])[0]
    pred_idx = np.argmax(probs)
    pred_prob = probs[pred_idx]
    if pred_prob < threshold or sneeze_sniff_classes[pred_idx] == "neither":
        return None
    return "sneeze_sniff"

# ------------------------
# Event detector
# ------------------------
def detect_events(waveform, sr=16000, window_sec=1.0):
    window_samples = int(window_sec * sr)
    num_windows = len(waveform) // window_samples
    session_dir = tempfile.mkdtemp()
    events = {evt: [] for evt in mapping.keys()}
    current_event, current_start, current_chunk = None, 0, []

    for i in range(num_windows):
        start, end = i * window_samples, (i + 1) * window_samples
        chunk = waveform[start:end]
        if len(chunk) == 0: continue
        if sr != 16000:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)

        scores, embeddings, _ = yamnet_model(chunk)
        mean_scores = np.mean(scores.numpy(), axis=0)

        silence_idx = class_names.index("Silence")
        if mean_scores[silence_idx] >= 0.5:
            pred_event = None
        else:
            pred_event = predict_sneeze_sniff(chunk, threshold=class_thresholds["sneeze_sniff"])
            if not pred_event:
                candidates = {}
                for evt, yam_classes in mapping.items():
                    if evt == "sneeze_sniff":
                        continue
                    idxs = [class_names.index(c) for c in yam_classes if c in class_names]
                    if not idxs: continue
                    score = mean_scores[idxs].max()
                    if score >= class_thresholds[evt]:
                        candidates[evt] = score
                if candidates:
                    pred_event = max(candidates, key=candidates.get)

        if pred_event == current_event:
            current_chunk.append(chunk)
        else:
            if current_event is not None:
                clip_audio = np.concatenate(current_chunk)
                timestamp = current_start / sr
                events[current_event].append((timestamp, clip_audio))
            current_event, current_start, current_chunk = pred_event, start, [chunk]

    if current_event and current_chunk:
        clip_audio = np.concatenate(current_chunk)
        timestamp = current_start / sr
        events[current_event].append((timestamp, clip_audio))

    # Save clips to temp dir
    saved_clips = {}
    for evt, clips in events.items():
        saved_clips[evt] = []
        for ts, audio in clips:
            clip_path = os.path.join(session_dir, f"{evt}_{ts:.2f}.wav")
            sf.write(clip_path, audio, sr)
            saved_clips[evt].append((ts, clip_path))

    return saved_clips

# ------------------------
# Streamlit UI
# ------------------------
st.title("😴 Sleep Tracker Demo")
st.write("Record short audio, detect events, and listen to detected clips.")

duration = st.slider("Recording duration (seconds)", 3, 15, 5)

if st.button("🎤 Start Recording"):
    waveform, sr = record_audio(duration=duration)
    st.success("Recording finished!")
    
    events, session_dir = detect_events(waveform, sr=sr, window_sec=1.0)

    st.subheader("Summary of Detected Events")
    for evt, timestamps in events.items():
        st.write(f"**{evt}**: {len(timestamps)} events")
        
        # Playback each clip
        for i, ts in enumerate(timestamps):
            clip_path = os.path.join(session_dir, f"{evt}_{ts:.2f}.wav")
            st.audio(clip_path, format="audio/wav")  # specify format for reliability
            st.write(f"- At ~{ts:.2f} sec")

