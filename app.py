import streamlit as st
import tempfile
import os
import ffmpeg
import whisper
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

# Load Hugging Face token from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Missing Hugging Face token. Add HF_TOKEN to .env file.")
    st.stop()

login(hf_token)

st.set_page_config(page_title="Video Insight Extractor", layout="wide")
st.set_option('server.maxUploadSize', 1024)

st.title("🎥 MP4 Video Insight Extractor")

video_file = st.file_uploader("Upload MP4 file", type=["mp4"])

if video_file is not None:
    with st.spinner("Saving uploaded video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name

    audio_path = video_path.replace(".mp4", ".wav")

    with st.spinner("Extracting audio with ffmpeg..."):
        try:
            ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
        except ffmpeg.Error as e:
            st.error("FFmpeg error: " + str(e))
            st.stop()

    with st.spinner("Transcribing with Whisper..."):
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript = result["text"]

    st.subheader("📜 Full Transcript")
    st.text_area("Transcript", transcript, height=200)

    with st.spinner("Summarizing..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
        summary = ""
        for chunk in chunks:
            summary += summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "

    st.subheader("🧠 Key Insights")
    st.write(summary.strip())

    st.subheader("❓ Questions & Answers")
    lines = transcript.split('.')
    qa_pairs = []
    for i, line in enumerate(lines):
        if '?' in line:
            question = line.strip()
            for j in range(i+1, min(i+4, len(lines))):
                answer = lines[j].strip()
                if answer:
                    qa_pairs.append((question, answer))
                    break

    if qa_pairs:
        for idx, (q, a) in enumerate(qa_pairs, 1):
            st.markdown(f"**Q{idx}:** {q}?")
            st.markdown(f"**A{idx}:** {a}")
    else:
        st.info("No Q&A found.")

    os.remove(video_path)
    os.remove(audio_path)
