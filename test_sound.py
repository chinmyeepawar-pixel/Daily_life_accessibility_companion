
import time

from sound_module import (
    start_sound_detection,
    stop_sound_detection,
    emergency_alert
)

print("\nTesting environmental sound monitoring...")

try:

    start_sound_detection()

    print("\nSound monitoring is active.")
    print("Try clapping near your microphone.")
    print("Monitoring for 15 seconds...")

    time.sleep(15)

    print("\nTesting emergency announcement...")
    emergency_alert()

    print("\nWaiting for speech to finish...")
    time.sleep(5)

finally:

    stop_sound_detection()

print("\nSound module test completed.")
