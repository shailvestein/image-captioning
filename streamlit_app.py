import os
import math
import pickle
from PIL import Image

import torch
import torch.nn as nn
import streamlit as st
from Model import TransformerImageCaptioning
from FeatureExtractor import FeatureExtractor, image_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Vocabulary Class KO SABSE PEHLE DEFINE KAREIN
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


# Custom Unpickler (Pickle ke Class Mismatch Error se bachne ke liye)
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Vocabulary':
            return Vocabulary
        return super().find_class(module, name)


@st.cache_resource
def load_all_assets():
    # 1. Load Vocabulary Object
    with open("./weights/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

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


# ==========================================
# 4. STREAMLIT UI IMPLEMENTATION
# ==========================================

st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.title("🖼️ Image Caption Generator")
st.write("Upload an image to generate a caption using Transformer & EfficientNet-B4.")

# Load models and assets
with st.spinner("Loading models into memory..."):
    vocab, model, feature_extractor, transform = load_all_assets()

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("Extracting features and generating caption..."):
            # 1. Image Preprocessing & Feature Extraction
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feature = feature_extractor(img_tensor)

            # 2. Beam Search Caption Generation
            caption = model.predict(img_feature, vocab, device, beam_width=3)
            res = ""
            for c in caption:
                if c == "<unk>":
                    continue
                res += c + " "
        # Display Result
        st.success("### Generated Caption:")
        st.write(f"**\"{res[1:-1]\"**")
