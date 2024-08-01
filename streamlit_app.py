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

def extract_audio(video_file, chunk_duration_ms=5000):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.getvalue())
        video_path = tmpfile.name

    audio = AudioSegment.from_file(video_path, format="mp4")
    audio = audio.set_channels(1).set_sample_width(2)
    
    for i, chunk in enumerate(tqdm(audio[::chunk_duration_ms])):
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        yield samples, audio.frame_rate

    os.unlink(video_path)

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
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if uploaded_file is not None:
        model = load_model(model_name)
        
        if st.button("Transcribe Video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            audio_generator = extract_audio(uploaded_file)
            transcript = transcribe_audio(model, audio_generator)
            
            end_time = time.time()
            process_time = end_time - start_time
            
            progress_bar.progress(100)
            status_text.text("Transcription Complete!")
            
            st.text_area("Transcript", transcript, height=300)
            
            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
            # Special Feature 1: Transcription Stats
            st.subheader("Transcription Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Processing Time", f"{process_time:.2f} seconds")
            col2.metric("Word Count", len(transcript.split()))
            col3.metric("Character Count", len(transcript))
            
            # Special Feature 2: Word Frequency Chart
            st.subheader("Word Frequency")
            word_freq = {}
            for word in transcript.lower().split():
                if len(word) > 3:  # Ignore short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig = go.Figure(data=[go.Bar(x=[word for word, _ in top_words], y=[freq for _, freq in top_words])])
            fig.update_layout(title="Top 10 Words", xaxis_title="Word", yaxis_title="Frequency")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()