
from tts_module import (
    speak,
    wait_until_ready,
    wait_for_queue,
    shutdown_tts
)

print("\nTesting batch-based TTS...")

if wait_until_ready(timeout=15):

    print("\nSubmitting four separate announcements...")

    speak("First announcement. Welcome to the accessibility companion.")

    speak("Second announcement. The vision system is ready.")

    speak("Third announcement. Speech processing is working.")

    speak("Fourth announcement. TTS testing is complete.")

    print("\nWaiting for announcements to finish...")

    wait_for_queue()

    print("\nAll announcements have been processed.")

else:
    print("TTS engine failed to initialise.")

shutdown_tts(wait=True)

print("\nTTS test finished.")
