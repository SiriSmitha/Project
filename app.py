import streamlit as st
from gtts import gTTS
import os
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to summarize text
def summarize_text(text):
    cleaned_text = clean_text(text)
    summary = summ_obj(cleaned_text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Function to generate audio from text using gTTS
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    return "output.mp3"

# Function to transcribe audio using Wav2Vec2 model
def transcribe_audio(audio_input):
    input_values = tokenizer(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Function to process microphone input
def process_microphone_input(duration=5, sample_rate=16000):
    audio_frames = []
    try:
        while True:
            audio_input = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            audio_frames.append(audio_input)
            key = input("")  # Wait for user to press a key
            if key.lower() == "q":
                break
    except KeyboardInterrupt:
        pass
    audio_input = np.concatenate(audio_frames)
    audio_input = np.squeeze(audio_input)
    return audio_input

def main():
    st.title("Text Summarization, TTS, and STT")

    # Text Summarization
    st.header("Text Summarization")
    text_input = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        if text_input:
            summary = summarize_text(text_input)
            st.subheader("Summary:")
            st.write(summary)

    # Text-to-Speech (TTS)
    st.header("Text-to-Speech (TTS)")
    tts_text = st.text_input("Enter the text for TTS:")
    if st.button("Generate Audio"):
        if tts_text:
            audio_file = generate_audio(tts_text)
            st.audio(audio_file, format='audio/mp3')

    # Speech-to-Text (STT)
    st.header("Speech-to-Text (STT)")
    st.info("Press 'q' to stop recording.")
    if st.button("Start Recording"):
        audio_input = process_microphone_input()
        transcription = transcribe_audio(audio_input)
        st.subheader("Transcription:")
        st.write(transcription)

if __name__ == "__main__":
    main()

