# sleep_tracker_app.py
import streamlit as st
import numpy as np
import tensorflow_hub as hub
import librosa
import tensorflow as tf
import os
import datetime
import soundfile as sf
import joblib
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# ------------------------
# Load YAMNet and classifier (cache)
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
# Mapping & thresholds (adjusted per your last request)
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

class_thresholds = {
    "snore": 0.3,
    "cough": 0.3,
    "fart": 0.3,
    "sleep_talking": 0.5,
    "sneeze_sniff": 0.7,
    "laughter": 0.3,
    "music": 0.3
}

# ------------------------
# Custom classifier predictor (unchanged)
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
# Event detector (slightly adapted to accept waveform array directly)
# ------------------------
def detect_events_from_waveform(waveform, sr=16000, window_sec=1.0, save_clips=True):
    window_samples = int(window_sec * sr)
    num_windows = len(waveform) // window_samples
    session_dir = f"sleep_logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if save_clips:
        os.makedirs(session_dir, exist_ok=True)
    events = {evt: [] for evt in mapping.keys()}
    current_event = None
    current_start = 0
    current_chunk = []
    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = waveform[start:end]
        if len(chunk) == 0:
            continue
        if sr != 16000:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
        scores, embeddings, spectrogram = yamnet_model(chunk)
        mean_scores = np.mean(scores.numpy(), axis=0)
        # Silence check (if available)
        if "Silence" in class_names:
            silence_idx = class_names.index("Silence")
        else:
            silence_idx = None
        if silence_idx is not None and mean_scores[silence_idx] >= 0.5:
            pred_event = None
        else:
            pred_event = predict_sneeze_sniff(chunk, threshold=class_thresholds["sneeze_sniff"])
            if not pred_event:
                candidates = {}
                for evt, yam_classes in mapping.items():
                    if evt == "sneeze_sniff":
                        continue
                    idxs = [class_names.index(c) for c in yam_classes if c in class_names]
                    if not idxs:
                        continue
                    score = mean_scores[idxs].max()
                    if score >= class_thresholds.get(evt, 0.3):
                        candidates[evt] = score
                if candidates:
                    pred_event = max(candidates, key=candidates.get)
        # merge logic
        if pred_event == current_event:
            current_chunk.append(chunk)
        else:
            if current_event is not None:
                clip_audio = np.concatenate(current_chunk)
                timestamp = current_start / sr
                events[current_event].append(timestamp)
                if save_clips:
                    clip_path = os.path.join(session_dir, f"{current_event}_{timestamp:.2f}.wav")
                    sf.write(clip_path, clip_audio, sr)
            current_event = pred_event
            current_start = start
            current_chunk = [chunk]
    if current_event is not None and current_chunk:
        clip_audio = np.concatenate(current_chunk)
        timestamp = current_start / sr
        events[current_event].append(timestamp)
        if save_clips:
            clip_path = os.path.join(session_dir, f"{current_event}_{timestamp:.2f}.wav")
            sf.write(clip_path, clip_audio, sr)
    return events, session_dir

# ------------------------
# Helper to convert buffer (list of frames arrays) -> mono waveform @16000
# ------------------------
def frames_to_waveform(frames, sample_rate):
    """
    frames: list of numpy arrays with shape (channels, samples) or (samples,) depending on source
    sample_rate: sample rate of frames
    returns: 1-D float32 numpy array at 16000 Hz
    """
    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    # concatenate along time axis
    # frames may be (channels, samples) -> convert to mono
    mono_buffers = []
    for f in frames:
        arr = np.asarray(f)
        if arr.ndim == 2:
            # channels x samples -> average to mono
            mono = arr.mean(axis=0)
        else:
            mono = arr
        mono_buffers.append(mono)
    combined = np.concatenate(mono_buffers).astype(np.float32)
    # if incoming is int16-like, normalize if needed; webrtc frames from av are float32 in [-1,1], but we check range
    if combined.dtype != np.float32:
        combined = combined.astype(np.float32)
    # ensure samplerate 16000
    if sample_rate != 16000:
        combined = librosa.resample(combined, orig_sr=sample_rate, target_sr=16000)
    return combined

# ------------------------
# AudioProcessor to collect frames using streamlit-webrtc
# ------------------------
class Recorder(AudioProcessorBase):
    def __init__(self):
        # buffer list of numpy arrays (samples)
        self.buffers = []
        # detect sample_rate from first frame
        self.sample_rate = None

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to numpy array: shape (channels, samples)
        arr = frame.to_ndarray()
        # arr dtype depends on format; convert to float32 in range [-1,1] if ints
        if arr.dtype.kind in ('i', 'u'):
            # typical int16 -> scale
            arr = arr.astype('float32') / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype('float32')
        self.buffers.append(arr)
        # set sample_rate if not set
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
        # return frame unchanged
        return frame

    def get_recording(self):
        return self.buffers, (self.sample_rate or 48000)

# ------------------------
# Streamlit UI
# ------------------------
st.title("😴 Sleep Tracker (WebRTC demo)")
st.write("Use the widget below to record audio in the browser. When ready, click **Capture (Stop & Process)**.")

# Start the WebRTC widget
webrtc_ctx = webrtc_streamer(
    key="speech-webrtc",
    mode=WebRtcMode.SENDRECV,  # we only need audio from browser
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=Recorder,
    async_processing=True,
)

# Buttons to capture and process
col1, col2 = st.columns(2)
with col1:
    if st.button("Capture (Stop & Process)"):
        # Attempt to access Recorder instance and its buffers
        if webrtc_ctx and webrtc_ctx.audio_processor:
            recorder = webrtc_ctx.audio_processor
            frames, sr = recorder.get_recording()
            waveform = frames_to_waveform(frames, sr)
            if waveform.size == 0:
                st.warning("No audio recorded. Make sure the WebRTC widget is running and you gave microphone permission.")
            else:
                st.success("Captured audio — analyzing...")
                # run detection
                events, session_dir = detect_events_from_waveform(waveform, sr=16000, window_sec=1.0, save_clips=True)
                # show summary and provide playback
                st.subheader("Summary of Detected Events")
                any_event = False
                for evt, timestamps in events.items():
                    if len(timestamps) > 0:
                        any_event = True
                    st.write(f"**{evt}**: {len(timestamps)}")
                    for t in timestamps:
                        clip_path = os.path.join(session_dir, f"{evt}_{t:.2f}.wav")
                        if os.path.exists(clip_path):
                            st.audio(clip_path, format="audio/wav")
                            st.write(f"- At ~{t:.2f} sec")
                if not any_event:
                    st.info("No events detected.")
        else:
            st.warning("WebRTC widget not ready yet. Wait until it shows 'Connected' in the widget.")

with col2:
    if st.button("Clear Recording Buffer"):
        if webrtc_ctx and webrtc_ctx.audio_processor:
            webrtc_ctx.audio_processor.buffers = []
            st.info("Cleared local recording buffer.")
        else:
            st.warning("Widget not ready.")

st.write("Tip: Click the small play/stop icon in the WebRTC widget to start/stop the microphone stream. Then press **Capture (Stop & Process)** to process what was buffered.")
