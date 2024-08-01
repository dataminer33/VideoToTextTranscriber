import streamlit as st
import os
import tempfile
import torch
import whisper
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

# Styling
st.set_page_config(page_title="Video Transcriber Pro", page_icon="🎥", layout="wide")

# Custom CSS for professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
        padding: 0;
    }
    .main {
        padding: 2rem 1rem;
    }
    .sidebar .sidebar-content {
        background: #002f4b;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: #ffffff;
    }
    .big-font {
        font-size:30px !important;
        color: #002f4b;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0073e6;
        border-radius: 20px;
    }
    .stSelectbox>div>div>input {
        border-radius: 20px;
    }
    .stFileUploader>div>button {
        color: #ffffff;
        background-color: #0073e6;
        border-radius: 20px;
    }
    .stSpinner {
        color: #0073e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    return whisper.load_model(model_name)

def extract_audio(file, file_type, chunk_duration_ms=5000):
    if file_type == "video":
        suffix = '.mp4'
        format_type = "mp4"
    else:
        suffix = '.' + file.name.split('.')[-1]
        format_type = suffix[1:]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
        tmpfile.write(file.getvalue())
        file_path = tmpfile.name

    audio = AudioSegment.from_file(file_path, format=format_type)
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
    st.sidebar.markdown("<h2 class='big-font'>Video Transcriber Pro 🎥✨</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("Upload your video or audio file to get an accurate transcription using the Whisper model.")
    
    model_name = st.sidebar.selectbox("Select Whisper Model", ["tiny", "base", "small", "medium", "large"])
    model = load_model(model_name)
    
    file_type = st.sidebar.selectbox("Select File Type", ["Video", "Audio"])
    if file_type == "Video":
        file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        file_format = "video"
    else:
        file = st.sidebar.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])
        file_format = "audio"

    if file:
        audio_generator = extract_audio(file, file_format)
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(model, audio_generator)
        st.markdown("<h3 class='big-font'>Transcription</h3>", unsafe_allow_html=True)
        st.write(transcript)

        # Add a download button for the transcript
        st.markdown("<h3 class='big-font'>Download Transcription</h3>", unsafe_allow_html=True)
        st.download_button(
            label="Download as Text File",
            data=transcript,
            file_name="transcription.txt",
            mime="text/plain"
        )

    st.sidebar.markdown("""
        <hr style="border-top: 3px solid #ffffff;">
        <div style="text-align:center;">
            <p style="color: #ffffff;">Developed by [Your Name]</p>
            <a href="https://github.com/[YourGitHubUsername]/[YourRepoName]" target="_blank" style="color: #ffffff;">GitHub Repository</a>
        </div>
    """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
