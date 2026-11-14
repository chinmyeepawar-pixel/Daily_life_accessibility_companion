"""
Daily Life Accessibility Companion
Speech Recognition Module

Improved continuous-conversation recording:
- Lazy Whisper Base loading (once)
- Configurable recording duration (7 seconds by default)
- Microphone input diagnostics
- Silence detection without rejecting quiet speech too aggressively
- CPU-compatible transcription
- Prevents simultaneous recordings
"""

import threading
import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16000
RECORDING_DURATION = 7
CHANNELS = 1
WHISPER_MODEL_NAME = "base"

# Low enough to retain quiet speech; near-silent recordings are skipped.
SILENCE_THRESHOLD = 0.0012

_model = None
_model_lock = threading.Lock()
_recording_lock = threading.Lock()


def _load_whisper_model():
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            print(f"Loading Whisper {WHISPER_MODEL_NAME} model...")
            _model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
            print("Whisper model loaded successfully.")

    return _model


def _record_audio(duration):
    print(f"\nRecording for up to {duration} seconds. Speak clearly now...")

    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocking=True
        )
    except Exception as error:
        raise RuntimeError(
            f"Microphone recording failed. Check Windows microphone permissions "
            f"and the default input device. Details: {error}"
        ) from error

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        raise RuntimeError("The microphone returned an empty recording.")

    if not np.all(np.isfinite(audio)):
        raise RuntimeError("The microphone returned invalid audio values.")

    return audio


def speech_to_text(duration=RECORDING_DURATION, language="en"):
    """
    Records one utterance and returns its transcription.
    Called repeatedly by the continuous SpeechWorker.
    """

    if not _recording_lock.acquire(blocking=False):
        print("Speech recognition is already recording.")
        return ""

    try:
        if duration <= 0:
            print("Recording duration must be greater than zero.")
            return ""

        model = _load_whisper_model()
        audio = _record_audio(duration)

        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        print(f"Microphone RMS: {rms:.5f} | peak: {peak:.5f}")

        if rms < SILENCE_THRESHOLD or peak < 0.004:
            print("Recording was too quiet. Listening again...")
            return ""

        # Whisper expects mono float audio. Gentle normalization can help
        # low-volume microphones without changing the speech content.
        if peak > 0:
            audio = audio / max(1.0, peak / 0.95)

        print("Transcribing...")
        result = model.transcribe(
            audio,
            language=language,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0
        )

        text = result.get("text", "").strip()
        print(f"Recognized speech: {text if text else '[nothing recognized]'}")
        return text

    except Exception as error:
        print(f"Speech recognition error: {error}")
        return ""

    finally:
        _recording_lock.release()


def is_model_loaded():
    return _model is not None
