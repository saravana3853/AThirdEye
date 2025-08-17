
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import ViTFeatureExtractor, TFAutoModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, AdditiveAttention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import h5py
from gtts import gTTS
import os

# Decoder model
class CaptionDecoder(Model):
    def __init__(self, vocab_size, embed_dim=256, lstm_units=512,**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        
        self.embedding = Embedding(vocab_size, embed_dim)
        self.attention = AdditiveAttention()
        self.lstm = LSTM(lstm_units, return_sequences=True)
        self.fc = Dense(vocab_size)
        self.attn_weights = None
        self.feature_proj = Dense(embed_dim)

    def call(self, features, captions):
        features = self.feature_proj(features)  # [B, 196, embed_dim]
        embedded = self.embedding(captions)     # [B, T, embed_dim]
        context = self.attention([embedded, features],return_attention_scores=True)  # [B, T, embed_dim]
        self.attn_weights = context[1]  # Save for viz
        x = tf.concat([context[0], embedded], axis=-1)
        x = self.lstm(x)
        return self.fc(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "lstm_units": self.lstm_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load HF ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

def verify_model_weights():
    with h5py.File("./caption_decoder.weights.h5", "r") as f:
        return list(f.keys())

def retrieve_tokenizer():        
    file_path = './tokenizer.pickle'
    with open(file_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Image preprocessing + feature extraction
def extract_features(img):
    inputs = feature_extractor(images=img, return_tensors="tf")
    outputs = vit_model(inputs["pixel_values"])
    patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove [CLS] token → shape: [1, 196, 768]
    return tf.squeeze(patch_embeddings, axis=0)  # shape: [196, 768]

def load_model(vocab_size):
    h5_model = CaptionDecoder(vocab_size=vocab_size)
    dummy_features = tf.random.uniform((1, 196, 768))
    dummy_captions = tf.random.uniform((1, 20), maxval=vocab_size, dtype=tf.int32)
    h5_model(dummy_features,dummy_captions)
    h5_model.load_weights('./caption_decoder.weights.h5')
    return h5_model

def generate_caption(features,model,tokenizer,max_len=30):
    input_seq = tokenizer.texts_to_sequences(["<start>"])[0]
    for _ in range(max_len):
        seq = pad_sequences([input_seq], maxlen=max_len, padding='post')
        preds = model(tf.expand_dims(features, 0), tf.convert_to_tensor(seq))
        pred_id = tf.argmax(preds[0, len(input_seq)-1]).numpy()
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        word = index_word.get(pred_id, '<unk>')
        if word == '<end>':
            break
        input_seq.append(pred_id)

    return ' '.join([index_word.get(i, '') for i in input_seq[1:]])

def play_audio(caption):
    caption_audio=gTTS(text=caption,lang='en',slow=False)
    caption_audio.save("output.mp3")
    with open("output.mp3", "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")

with st.container():
    st.title("Image Caption Reader")
    st.write("Upload an image and get predictions.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess (adjust as per your model's input shape)
        img_resized = image.resize((224, 224))

        features=extract_features(img_resized)
        tokenizer=retrieve_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        #st.write('Model Weights : '+str(verify_model_weights()))
        #st.write('vocab_size : '+str(vocab_size))
        model=load_model(vocab_size=vocab_size)
        caption=generate_caption(features,model,tokenizer)
        st.success(str(caption))
        play_audio(str(caption))
