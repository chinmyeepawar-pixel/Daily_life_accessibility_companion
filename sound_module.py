
"""
Daily Life Accessibility Companion
Environmental Sound Monitoring Module

Features:
- Real-time microphone monitoring
- RMS loudness measurement
- Microphone sensitivity diagnostics
- Loud sound detection
- Announcement cooldown
- Prevention of duplicate monitoring threads
- Safe microphone start and stop
- Integration with central TTS module
"""

import threading
import time

import numpy as np
import sounddevice as sd

from tts_module import speak


# ==================================================
# CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600

# Keep this threshold for the calibration test.
# We will adjust it using your actual microphone readings.
LOUD_SOUND_THRESHOLD = 0.12

# Minimum time between loud sound announcements.
ANNOUNCEMENT_COOLDOWN = 5.0

# Print microphone readings every half second.
LEVEL_DISPLAY_INTERVAL = 0.5


# ==================================================
# INTERNAL STATE
# ==================================================

running = False

_stop_event = threading.Event()
_monitor_thread = None

_state_lock = threading.Lock()

_last_announcement_time = 0.0
_last_level_print_time = 0.0


# ==================================================
# AUDIO CALLBACK
# ==================================================

def _audio_callback(indata, frames, time_info, status):
    """
    Receives microphone audio and measures RMS volume.

    Also prints microphone readings for sensitivity
    calibration.
    """

    global _last_announcement_time
    global _last_level_print_time

    if status:
        print(f"Audio input status: {status}")

    try:

        # Convert audio input to float32.
        audio = np.asarray(indata, dtype=np.float32)

        # Calculate RMS volume.
        rms_volume = float(
            np.sqrt(np.mean(np.square(audio)))
        )

        current_time = time.monotonic()

        # ------------------------------------------
        # MICROPHONE LEVEL DIAGNOSTICS
        # ------------------------------------------

        if (
            current_time - _last_level_print_time
            >= LEVEL_DISPLAY_INTERVAL
        ):

            print(
                f"Microphone RMS: {rms_volume:.4f}"
            )

            _last_level_print_time = current_time

        # ------------------------------------------
        # LOUD SOUND DETECTION
        # ------------------------------------------

        if rms_volume >= LOUD_SOUND_THRESHOLD:

            if (
                current_time - _last_announcement_time
                >= ANNOUNCEMENT_COOLDOWN
            ):

                _last_announcement_time = current_time

                print(
                    f"\nLOUD SOUND DETECTED! "
                    f"RMS: {rms_volume:.4f}"
                )

                speak("Loud sound detected.")

    except Exception as error:

        print(f"Sound processing error: {error}")


# ==================================================
# SOUND MONITORING WORKER
# ==================================================

def _sound_worker():
    """
    Runs microphone monitoring in a background thread.
    """

    global running

    try:

        print("\nSound monitoring started.")
        print("Microphone sensitivity calibration active.")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=_audio_callback
        ):

            while not _stop_event.wait(0.1):
                pass

    except Exception as error:

        print(f"Sound monitoring error: {error}")

    finally:

        with _state_lock:
            running = False

        print("\nSound monitoring stopped.")


# ==================================================
# START SOUND DETECTION
# ==================================================

def start_sound_detection():
    """
    Starts microphone monitoring.

    Prevents duplicate monitoring threads.
    """

    global running
    global _monitor_thread

    with _state_lock:

        if running:
            print("Sound monitoring is already active.")
            return

        if (
            _monitor_thread is not None
            and _monitor_thread.is_alive()
        ):

            print("Previous sound monitor is still stopping.")
            return

        _stop_event.clear()

        running = True

        _monitor_thread = threading.Thread(
            target=_sound_worker,
            name="AccessibilitySoundWorker",
            daemon=True
        )

        _monitor_thread.start()

        print("Sound detection requested.")


# ==================================================
# STOP SOUND DETECTION
# ==================================================

def stop_sound_detection():
    """
    Stops microphone monitoring safely.
    """

    global running
    global _monitor_thread

    with _state_lock:

        worker = _monitor_thread

        if worker is None or not worker.is_alive():
            running = False
            print("Sound monitoring is already stopped.")
            return

        _stop_event.set()

    # Join outside the lock.
    if worker is not threading.current_thread():
        worker.join(timeout=5)

    with _state_lock:

        if not worker.is_alive():
            _monitor_thread = None
            running = False
        else:
            print("Warning: sound monitor is still stopping.")


# ==================================================
# EMERGENCY ALERT
# ==================================================

def emergency_alert():
    """
    Sends an emergency announcement through central TTS.

    This is a local audio announcement.
    It does not contact emergency services.
    """

    print("Emergency alert activated.")

    speak(
        "Emergency activated. Please seek assistance."
    )


# ==================================================
# CLEANUP
# ==================================================

def cleanup_sound_detection():
    """
    Stops monitoring during application shutdown.
    """

    stop_sound_detection()
