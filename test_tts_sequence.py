
from tts_module import (
    speak,
    wait_until_ready,
    wait_for_queue,
    shutdown_tts
)

print("\nDaily Life Accessibility Companion")
print("Direct Windows SAPI5 Test")

if not wait_until_ready(timeout=15):

    print("Windows SAPI5 initialisation failed.")
    raise SystemExit(1)

announcements = [
    "First announcement. The camera is starting.",
    "Second announcement. A person is in front of you.",
    "Third announcement. A book is detected.",
    "Fourth announcement. Vision testing is complete."
]

try:

    for index, message in enumerate(announcements, start=1):

        print(f"\nSubmitting announcement {index}.")

        speak(message)

    print("\nWaiting for all announcements...")

    completed = wait_for_queue(timeout=60)

    if completed:
        print("All announcements processed.")
    else:
        print("Speech queue timed out.")

finally:

    shutdown_tts(wait=True)

print("\nDirect SAPI5 test completed.")
