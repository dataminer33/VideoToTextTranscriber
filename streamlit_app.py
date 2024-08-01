import streamlit as st
import os
import tempfile
import torch
import whisper
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import time
import plotly.graph_objects as go

# Styling
st.set_page_config(page_title="Video Transcriber Pro", page_icon="🎥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #4e54c8, #8f94fb);
    }
    .big-font {
        font-size:30px !important;
        color: #ffffff;
    }
    .stButton>button {
        color: #4e54c8;
        background-color: #ffffff;
        border-radius: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stSelectbox>div>div>input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    return whisper.load_model(model_name)

def extract_audio(file, file_type, chunk_duration_ms=600000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmpfile:
        tmpfile.write(file.getvalue())
        file_path = tmpfile.name

    audio = AudioSegment.from_file(file_path, format=file_type)
    audio = audio.set_channels(1).set_sample_width(2)
    
    for i, chunk in enumerate(tqdm(audio[::chunk_duration_ms])):
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        yield samples, audio.frame_rate

    os.unlink(file_path)

def transcribe_audio(model, audio_generator):
    full_transcript = ""
    for audio_chunk, sample_rate in audio_generator:
        with torch.no_grad():
            result = model.transcribe(audio_chunk)
        full_transcript += result["text"] + " "
        torch.cuda.empty_cache()
    return full_transcript.strip()

def main():
    st.markdown("<h1 class='big-font'>Video Transcriber Pro 🎥✨</h1>", unsafe_allow_html=True)
    
    model_name = st.selectbox("Select Whisper Model", ["tiny", "base", "small", "medium", "large"])
    model = load_model(model_name)
    
    file_type = st.selectbox("Select File Type", ["Video", "Audio"])
    if file_type == "Video":
        file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    else:
        file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])

    if file:
        audio_generator = extract_audio(file, file.type.split('/')[-1])
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(model, audio_generator)
        st.markdown("### Transcription")
        st.write(transcript)
    
if __name__ == "__main__":
    main()
