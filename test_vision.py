
"""
Daily Life Accessibility Companion
Vision and Voice Integration Test

Tests:
- Camera input
- YOLOv8m detection
- Stable object detection
- Natural scene descriptions
- TTS queue integration
"""

import cv2

from vision_module import process_frame

from tts_module import (
    speak,
    wait_until_ready,
    wait_for_queue,
    shutdown_tts
)


print("\nDaily Life Accessibility Companion")
print("Natural Vision and Voice Test")

# ==================================================
# INITIALISE TTS
# ==================================================

print("\nInitialising speech engine...")

if not wait_until_ready(timeout=15):

    print("TTS initialisation failed.")

    raise SystemExit(1)

print("Speech engine ready.")

# ==================================================
# INITIALISE CAMERA
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Camera could not be opened.")

    shutdown_tts()

    raise SystemExit(1)

print("\nCamera started.")
print("Press Q to close the camera window.")

# ==================================================
# MAIN LOOP
# ==================================================

last_scene = ""

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("Unable to capture camera frame.")

            break

        (
            frame,
            description,
            navigation,
            speak_flag,
            message
        ) = process_frame(frame)

        # ------------------------------------------
        # PRINT SCENE ONLY WHEN IT CHANGES
        # ------------------------------------------

        if description and description != last_scene:

            print("\nScene:", description)

            last_scene = description

        # ------------------------------------------
        # SEND ANNOUNCEMENT TO TTS
        # ------------------------------------------

        if speak_flag and message:

            print("\nSpeaking:", message)

            speak(message)

        # ------------------------------------------
        # DISPLAY CAMERA
        # ------------------------------------------

        cv2.imshow(
            "Daily Life Accessibility Companion",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

finally:

    camera.release()

    cv2.destroyAllWindows()

    print("\nClosing vision test...")

    # Wait for outstanding voice announcements.
    completed = wait_for_queue(timeout=30)

    if completed:
        print("All voice announcements completed.")
    else:
        print("Some voice announcements may still be pending.")

    shutdown_tts()

print("\nVision and voice test completed.")
