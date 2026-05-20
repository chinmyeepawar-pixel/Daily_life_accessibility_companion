import sounddevice as sd
import numpy as np
import threading
from tts_module import speak

running = False


def detect_sound():
    global running

    def callback(indata, frames, time, status):
        volume = np.linalg.norm(indata)

        if volume > 30:
            speak("Loud sound detected")

    with sd.InputStream(callback=callback):
        while running:
            sd.sleep(500)


def start_sound_detection():
    global running
    running = True
    threading.Thread(target=detect_sound, daemon=True).start()


def stop_sound_detection():
    global running
    running = False


def emergency_alert():
    speak("Emergency activated")