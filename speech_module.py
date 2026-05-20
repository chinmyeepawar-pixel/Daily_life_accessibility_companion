import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Load model once
model = whisper.load_model("base")


def speech_to_text():
    try:
        fs = 16000
        duration = 5  # 🔥 increased duration for better capture

        print("🎤 Recording...")

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        wav.write("temp.wav", fs, recording)

        # 🔥 IMPORTANT FIX (CPU compatibility + stability)
        result = model.transcribe("temp.wav", fp16=False)

        text = result["text"].strip()

        print("Recognized:", text)

        return text

    except Exception as e:
        print("Speech Error:", e)
        return ""