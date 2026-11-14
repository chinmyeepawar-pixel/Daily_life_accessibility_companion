"""
Daily Life Accessibility Companion
Windows SAPI5 Direct Text-to-Speech Module

- Persistent SAPI5 voice on a dedicated COM worker thread
- FIFO queue; each announcement is spoken completely in sequence
- Queue wait includes the currently playing utterance
- Safe readiness/error state and graceful shutdown
"""

import atexit
import queue
import threading

import pythoncom
import win32com.client

DEFAULT_VOICE_INDEX = 0
SPEECH_RATE = 2
SPEECH_VOLUME = 100

_speech_queue = queue.Queue()
_worker_thread = None
_worker_lock = threading.Lock()
_shutdown_requested = False
_engine_ready = threading.Event()
_engine_error = None


def _speech_worker():
    global _engine_error

    voice = None
    com_initialized = False

    try:
        pythoncom.CoInitialize()
        com_initialized = True

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = voice.GetVoices()

        if voices.Count == 0:
            raise RuntimeError("No Windows SAPI voices were found.")

        print("\nAvailable Windows voices:")
        for index in range(voices.Count):
            print(f"{index}: {voices.Item(index).GetDescription()}")

        selected_index = DEFAULT_VOICE_INDEX
        if selected_index < 0 or selected_index >= voices.Count:
            selected_index = 0

        voice.Voice = voices.Item(selected_index)
        voice.Rate = SPEECH_RATE
        voice.Volume = SPEECH_VOLUME

        _engine_error = None
        _engine_ready.set()
        print("Windows SAPI5 initialised successfully.")

        while True:
            message = _speech_queue.get()

            if message is None:
                _speech_queue.task_done()
                break

            try:
                print(f"TTS speaking: {message}")
                # Synchronous playback: do not take the next item until done.
                voice.Speak(message, 0)
                print("TTS playback completed.")
            except Exception as error:
                _engine_error = error
                print(f"TTS playback error: {error}")
            finally:
                _speech_queue.task_done()

    except Exception as error:
        _engine_error = error
        _engine_ready.set()
        print(f"TTS initialisation error: {error}")

    finally:
        voice = None

        if com_initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception as error:
                print(f"COM cleanup warning: {error}")

        # Mark any abandoned queue items as completed if the worker failed.
        while True:
            try:
                _speech_queue.get_nowait()
            except queue.Empty:
                break
            else:
                _speech_queue.task_done()

        print("TTS worker stopped.")


def _ensure_worker():
    global _worker_thread, _engine_error

    with _worker_lock:
        if _shutdown_requested:
            return False

        if _worker_thread is None or not _worker_thread.is_alive():
            _engine_ready.clear()
            _engine_error = None
            _worker_thread = threading.Thread(
                target=_speech_worker,
                name="AccessibilityTTSWorker",
                daemon=True
            )
            _worker_thread.start()

        return True


def speak(text):
    """Queue text for sequential speech. Returns True when accepted."""
    if text is None:
        return False

    message = str(text).strip()
    if not message:
        return False

    if not _ensure_worker():
        print("TTS unavailable or shutting down.")
        return False

    # Wait briefly for engine initialisation before accepting the message.
    if not _engine_ready.wait(timeout=15):
        print("TTS engine did not become ready in time.")
        return False

    with _worker_lock:
        if _shutdown_requested or _engine_error is not None:
            if _engine_error is not None:
                print(f"TTS unavailable: {_engine_error}")
            return False
        _speech_queue.put(message)

    return True


def wait_until_ready(timeout=15):
    if not _ensure_worker():
        return False
    return _engine_ready.wait(timeout=timeout) and _engine_error is None


def wait_for_queue(timeout=None):
    """Wait until queued and currently playing speech has completed."""
    if timeout is None:
        _speech_queue.join()
        return True

    completed = threading.Event()

    def waiter():
        _speech_queue.join()
        completed.set()

    threading.Thread(target=waiter, daemon=True).start()
    return completed.wait(timeout=timeout)


def shutdown_tts(wait=True):
    global _shutdown_requested

    with _worker_lock:
        if _shutdown_requested:
            worker = _worker_thread
        else:
            _shutdown_requested = True
            worker = _worker_thread
            if worker is not None and worker.is_alive():
                _speech_queue.put(None)

    if wait and worker is not None:
        worker.join(timeout=30)


atexit.register(shutdown_tts)
