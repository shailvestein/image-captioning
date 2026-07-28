import os
import math
import pickle
import threading
from PIL import Image

import torch
import torch.nn as nn
import streamlit as st
from Model import TransformerImageCaptioning
from FeatureExtractor import FeatureExtractor, image_transform

# THREAD LOCK FOR QUEUE MANAGEMENT
# Locking mechanism to process one image at a time

@st.cache_resource
def get_processing_lock():
    return threading.Lock()

processing_lock = get_processing_lock()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# VOCABULARY & UNPICKLER CLASSES
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text):
        tokenized_text = text.split()
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Vocabulary':
            return Vocabulary
        return super().find_class(module, name)


# LOAD ASSETS (CACHED)
@st.cache_resource
def load_all_assets():
    # 1. Load Vocabulary Object
    with open("./weights/vocab.pkl", "rb") as f:
        vocab = CustomUnpickler(f).load()

    vocab_size = len(vocab)

    # 2. Instantiate and Load Transformer Model
    model = TransformerImageCaptioning(
        vocab_size=vocab_size,
        max_length=40,
        feature_input_shape=1792
    ).to(device)

    model.load_state_dict(torch.load("./weights/transformer_image_captioning_effnet.pth", map_location=device))
    model.eval()

    # 3. Load EfficientNet Feature Extractor
    feature_extractor = FeatureExtractor(fine_tune=False).to(device)
    feature_extractor.eval()

    transform = image_transform()

    return vocab, model, feature_extractor, transform


# PAGE CONFIG & CUSTOM CSS (CARDS & UI)
st.set_page_config(
    page_title="Image Caption AI Generator",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Responsive CSS Styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding-top: 1rem;
    }
    
    /* Title Styling */
    .title-text {
        text-align: center;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }

    /* Cards Grid Layout */
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .card-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .card-title {
        font-weight: 600;
        color: #0F172A;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .card-desc {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.4;
    }

    /* Output Box */
    .output-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .output-caption {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A8A;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


#  UI HEADER & CARDS
st.markdown("<h1 class='title-text'>🖼️ AI Image Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Transform your visual content into meaningful descriptive sentences instantly using Deep Learning.</p>", unsafe_allow_html=True)

# Load heavy assets
with st.spinner("⚡ Initializing AI Models..."):
    vocab, model, feature_extractor, transform = load_all_assets()

# Project Info Cards in Simple English
st.markdown("""
<div class="card-container">
    <div class="info-card">
        <div class="card-icon">👁️</div>
        <div class="card-title">1. EfficientNet-B4</div>
        <div class="card-desc">Extracts deep visual features and patterns from your uploaded image.</div>
    </div>
    <div class="info-card">
        <div class="card-icon">🧠</div>
        <div class="card-title">2. Transformer Decoder</div>
        <div class="card-desc">Processes visual features and builds contextual word relationships.</div>
    </div>
    <div class="info-card">
        <div class="card-icon">🔍</div>
        <div class="card-title">3. Beam Search</div>
        <div class="card-desc">Evaluates multiple word paths to output the most accurate caption.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# FILE UPLOAD & QUEUE-LOCKED PROCESSING
uploaded_file = st.file_uploader("Choose an image to describe...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Generate Caption", use_container_width=True):
        
        # Check if another user is currently using CPU/GPU
        if processing_lock.locked():
            st.info("⌛ Another image is currently being processed. You have been placed in the queue...")

        # Acquire lock to ensure safe single-request execution
        with processing_lock:
            with st.spinner("🧠 Analyzing image and generating caption..."):
                # 1. Feature Extraction
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feature = feature_extractor(img_tensor)

                # 2. Beam Search Caption Generation
                caption = model.predict(img_feature, vocab, device, beam_width=3)
                
                res = ""
                for c in caption:
                    if c in ["<unk>", "<start>", "<end>"]:
                        continue
                    res += c + " "
                
                clean_caption = res.strip()

            # Display Result Card
            st.markdown(f"""
            <div class="output-box">
                <p style="font-size:0.85rem; color:#3B82F6; font-weight:700; margin-bottom:4px;">GENERATED CAPTION</p>
                <p class="output-caption">"{clean_caption}"</p>
            </div>
            """, unsafe_allow_html=True)
