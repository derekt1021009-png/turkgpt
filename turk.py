CONVERSATION_LIMIT = 120  # seconds

from dotenv import load_dotenv
import os
import time
import uuid
import re
import numpy as np
from collections import Counter

from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

load_dotenv()
client = OpenAI()

USE_MIC = False  # Set to True on a real machine with a microphone

conversation = [
    {
        "role": "system",
        "content": "You are Mustafa Kumal Ataturk, you are irratable and downright mean, you tell fun facts about turkey the food, turkey the nation, and Sean Turkington, high school AP Pre Calc teacher and creator of turksmathstuff yt channel and website, you sometimes randomly give AP PreCalc questions"
    }
]

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper loaded.")

try:
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

audio_files = []


def record_audio(fs=16000, silence_threshold=0.002, silence_duration=2.0, max_duration=20):
    recording = []
    silence_start = None
    recording_started = False
    start_time = None
    filename = f"user_{uuid.uuid4().hex[:8]}.wav"

    with sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(1024)
            volume = np.linalg.norm(data) / len(data)

            if not recording_started:
                if volume > silence_threshold:
                    recording_started = True
                    start_time = time.time()
                    recording.append(data)
                continue

            recording.append(data)

            if volume < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    break
            else:
                silence_start = None

            if time.time() - start_time > max_duration:
                break

    if not recording_started:
        return None

    audio = np.concatenate(recording, axis=0).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    write(filename, fs, audio_int16)
    print("Saved:", filename)
    return filename


def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def speak(text):
    print(f"\nAtaturk: {text}\n")
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass
    time.sleep(0.6)


def get_ai_response(user_text):
    conversation.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.6
    )
    ai_text = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": ai_text})
    return ai_text


def get_user_input():
    if USE_MIC:
        print("Listening...")
        filename = record_audio()
        if filename:
            audio_files.append(filename)
            text = transcribe(filename)
            print(f"You said: {text}")