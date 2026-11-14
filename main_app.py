"""
Daily Life Accessibility Companion
Desktop Application

Speech conversation stabilised:
- Continuous push-once listening loop
- Waits for TTS playback before recording again
- Clear Start/Stop Listening toggle
- Speech and AI run in a background QThread

Existing vision, OCR, sound monitoring and emergency controls retained.
Facial emotion recognition has been removed from this desktop interface
to reduce model load and simplify operation.
"""

import sys
import time
import threading

import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QTextEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy
)

from vision_module import process_frame
from tts_module import speak, wait_until_ready, wait_for_queue, shutdown_tts
from sound_module import start_sound_detection, stop_sound_detection, emergency_alert

APP_NAME = "Daily Life Accessibility Companion"
CAMERA_INDEX = 0
CAMERA_DELAY_MS = 80


class VisionWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    announcement_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        camera = None
        try:
            self.status_changed.emit("Opening camera...")
            camera = cv2.VideoCapture(self.camera_index)

            if not camera.isOpened():
                self.error_occurred.emit(
                    "Unable to open the camera. Check camera permissions and connections."
                )
                return

            self.status_changed.emit("Vision mode active")

            while not self._stop_event.is_set():
                success, frame = camera.read()
                if not success:
                    self.msleep(100)
                    continue

                try:
                    processed, description, navigation, speak_flag, message = process_frame(frame)
                    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    height, width, channels = rgb.shape
                    image = QImage(
                        rgb.data, width, height, channels * width,
                        QImage.Format_RGB888
                    ).copy()

                    self.frame_ready.emit(image, frame.copy())

                    if description:
                        self.status_changed.emit(description)

                    if speak_flag and message:
                        self.announcement_ready.emit(message)

                except Exception as error:
                    self.error_occurred.emit(f"Vision processing error: {error}")

                self.msleep(CAMERA_DELAY_MS)

        except Exception as error:
            self.error_occurred.emit(f"Camera worker error: {error}")

        finally:
            if camera is not None:
                camera.release()
            self.status_changed.emit("Vision mode stopped")


class SpeechWorker(QThread):
    result_ready = pyqtSignal(str, str, str, str)
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._reply_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self._reply_event.set()

    def release_after_reply(self):
        self._reply_event.set()

    def _wait_for_reply(self):
        while not self._stop_event.is_set():
            if self._reply_event.wait(0.1):
                self._reply_event.clear()
                break

        if not self._stop_event.is_set():
            # Queue join includes the utterance currently being spoken.
            if not wait_for_queue(timeout=120):
                print("Warning: TTS queue did not finish within 120 seconds.")

    def run(self):
        try:
            from speech_module import speech_to_text

            # The greeting is queued by the GUI before this worker starts.
            if not wait_for_queue(timeout=30):
                self.status_changed.emit("Waiting for voice output timed out")

            while not self._stop_event.is_set():
                self.status_changed.emit("Listening... speak now")

                text = speech_to_text(duration=7, language="en")

                if self._stop_event.is_set():
                    break

                if not text or not text.strip():
                    self.status_changed.emit("I didn't catch that. Listening again...")
                    continue

                text = text.strip()
                lower = text.lower()
                print(f"Conversation input: {text}")

                if "stop listening" in lower or "stop conversation" in lower:
                    self.result_ready.emit(text, "Okay, I'll stop listening now.", "", "stop_listening")
                    break

                if "emergency" in lower:
                    self.result_ready.emit(text, "", "", "emergency")
                    self._wait_for_reply()
                    continue

                if "start vision" in lower:
                    self.result_ready.emit(text, "", "", "start_vision")
                    self._wait_for_reply()
                    continue

                if "stop vision" in lower:
                    self.result_ready.emit(text, "", "", "stop_vision")
                    self._wait_for_reply()
                    continue

                if any(p in lower for p in ("read text", "read this", "read the text")):
                    self.result_ready.emit(text, "", "", "read_text")
                    self._wait_for_reply()
                    continue

                self.status_changed.emit("Understanding your request...")

                try:
                    from ai_brain import generate_response
                    response, emotion = generate_response(text)
                    if not response:
                        response = "I'm here. Could you say that again?"
                    if not emotion:
                        emotion = "unknown"
                except Exception as error:
                    print(f"AI response error: {error}")
                    response, emotion = "I'm having trouble reaching my assistant right now.", "unknown"

                self.result_ready.emit(text, response, emotion, "none")
                self._wait_for_reply()

        except Exception as error:
            print(f"Continuous speech worker error: {error}")
            self.status_changed.emit(f"Speech error: {error}")

        finally:
            self.status_changed.emit("Speech listening stopped")


class OCRWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, frame):
        super().__init__()
        self.frame = frame.copy()

    def run(self):
        try:
            from ocr_module import extract_text
            result = extract_text(self.frame)
            self.result_ready.emit(result if result else "No readable text found.")
        except Exception as error:
            self.result_ready.emit(f"OCR error: {error}")


class AccessibilityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 760)
        self.setMinimumSize(850, 620)

        self.vision_worker = None
        self.speech_worker = None
        self.ocr_worker = None
        self.latest_frame = None
        self.sound_active = False
        self.emergency_active = False
        self.last_spoken_message = ""

        self.build_interface()
        self.apply_styles()
        self.update_status("System ready")

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(24, 20, 24, 20)
        main.setSpacing(16)

        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QLabel("Your intelligent everyday accessibility assistant")
        subtitle.setObjectName("subtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout)
        header.addStretch()
        self.status_label = QLabel("● System ready")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)
        main.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)

        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_layout = QVBoxLayout(camera_panel)
        camera_title = QLabel("LIVE VISION")
        camera_title.setObjectName("sectionTitle")
        camera_layout.addWidget(camera_title)

        self.camera_label = QLabel("Camera is currently inactive\n\nSelect Vision Mode to begin")
        self.camera_label.setObjectName("camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(450, 320)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout.addWidget(self.camera_label, 1)

        self.scene_label = QLabel("Scene information will appear here.")
        self.scene_label.setObjectName("scene")
        self.scene_label.setWordWrap(True)
        camera_layout.addWidget(self.scene_label)
        content.addWidget(camera_panel, 3)

        control_panel = QFrame()
        control_panel.setObjectName("panel")
        controls = QVBoxLayout(control_panel)
        control_title = QLabel("ASSISTANT CONTROLS")
        control_title.setObjectName("sectionTitle")
        controls.addWidget(control_title)

        self.vision_btn = QPushButton("👁  Start Vision")
        self.speech_btn = QPushButton("🎤  Start Conversation")
        self.ocr_btn = QPushButton("📖  Read Text")
        self.sound_btn = QPushButton("🔊  Start Sound Monitoring")
        self.emergency_btn = QPushButton("🚨  Emergency")
        self.stop_btn = QPushButton("⛔  Stop All")

        self.vision_btn.clicked.connect(self.start_vision)
        self.speech_btn.clicked.connect(self.handle_speech)
        self.ocr_btn.clicked.connect(self.handle_ocr)
        self.sound_btn.clicked.connect(self.toggle_sound)
        self.emergency_btn.clicked.connect(self.handle_emergency)
        self.stop_btn.clicked.connect(self.stop_all)

        for button in (
            self.vision_btn, self.speech_btn, self.ocr_btn,
            self.sound_btn, self.emergency_btn, self.stop_btn
        ):
            button.setMinimumHeight(46)
            controls.addWidget(button)

        controls.addStretch()
        content.addWidget(control_panel, 2)
        main.addLayout(content, 3)

        output_panel = QFrame()
        output_panel.setObjectName("panel")
        output_layout = QVBoxLayout(output_panel)
        output_title = QLabel("ASSISTANT ACTIVITY")
        output_title.setObjectName("sectionTitle")
        output_layout.addWidget(output_title)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            "Recognised speech, AI responses, OCR text and system announcements will appear here."
        )
        self.output.setMinimumHeight(130)
        output_layout.addWidget(self.output)
        main.addWidget(output_panel, 1)

        footer = QLabel(
            "Experimental assistive technology • Always verify your surroundings independently."
        )
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        main.addWidget(footer)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #101522; }
            QWidget { color: #edf1f7; font-family: Segoe UI; font-size: 10pt; }
            #appTitle { font-size: 21pt; font-weight: 700; color: #ffffff; }
            #subtitle { color: #9ba8bb; font-size: 10pt; }
            #status { color: #6ed9a0; font-weight: 600; }
            #panel { background-color: #1a2333; border: 1px solid #2a3549; border-radius: 12px; }
            #sectionTitle { color: #8eafff; font-size: 10pt; font-weight: 700; padding: 5px; }
            #camera { background-color: #0b101a; border: 1px solid #344157; border-radius: 8px; color: #8d9bb0; font-size: 12pt; }
            #scene { color: #d8e2f4; background-color: #111b2b; border-radius: 7px; padding: 10px; }
            QPushButton { background-color: #283850; border: 1px solid #3c5274; border-radius: 8px; padding: 9px; text-align: left; font-weight: 600; }
            QPushButton:hover { background-color: #354b6c; }
            QPushButton:pressed { background-color: #1c2b42; }
            QTextEdit { background-color: #0e1521; border: 1px solid #344157; border-radius: 8px; padding: 10px; color: #e4ebf5; }
            #footer { color: #7f8ba0; font-size: 9pt; }
        """)

    def update_status(self, message):
        self.status_label.setText("● " + message)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.output.append(f"[{timestamp}] {message}")

    # ---------------- VISION ----------------

    def start_vision(self):
        if self.vision_worker is not None and self.vision_worker.isRunning():
            self.log("Vision is already running.")
            return

        self.last_spoken_message = ""
        self.vision_worker = VisionWorker(CAMERA_INDEX)
        self.vision_worker.frame_ready.connect(self.display_frame)
        self.vision_worker.announcement_ready.connect(self.handle_vision_announcement)
        self.vision_worker.status_changed.connect(self.update_status)
        self.vision_worker.error_occurred.connect(self.handle_error)
        self.vision_worker.start()
        self.vision_btn.setText("👁  Vision Running")
        self.log("Vision mode requested.")
        self.start_sound()

    def display_frame(self, image, original_frame):
        self.latest_frame = original_frame.copy()
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(
            pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def handle_vision_announcement(self, message):
        self.scene_label.setText(message)
        if message != self.last_spoken_message:
            self.last_spoken_message = message
            self.log("Vision: " + message)
            speak(message)

    # ---------------- CONTINUOUS SPEECH ----------------

    def handle_speech(self):
        if self.speech_worker is not None and self.speech_worker.isRunning():
            self.stop_speech()
            return

        if not wait_until_ready(timeout=15):
            self.log("TTS could not initialise. Check the terminal for the SAPI5 error.")
            self.update_status("Voice output unavailable")
            return

        self.speech_worker = SpeechWorker()
        self.speech_worker.result_ready.connect(self.handle_speech_result)
        self.speech_worker.status_changed.connect(self.update_status)
        self.speech_worker.finished.connect(self._speech_finished)

        self.speech_btn.setText("🎤  Stop Listening")
        self.log("Conversation started. Speak after the listening prompt. Say 'stop listening' to finish.")
        speak("I'm listening. You can speak now.")
        self.speech_worker.start()

    def stop_speech(self):
        worker = self.speech_worker
        if worker is None or not worker.isRunning():
            self._speech_finished()
            return
        worker.stop()
        self.update_status("Stopping speech listening...")
        self.log("Speech listening stop requested.")

    def _speech_finished(self):
        self.speech_btn.setText("🎤  Start Conversation")
        if self.speech_worker is None or not self.speech_worker.isRunning():
            self.update_status("System ready")

    def handle_speech_result(self, text, response, emotion, command):
        if text:
            self.log("Recognised speech: " + text)

        if command == "stop_listening":
            if response:
                speak(response)
            self.log("Voice conversation ended.")
            if self.speech_worker is not None:
                self.speech_worker.stop()
            return

        if command == "emergency":
            self.handle_emergency()
            if self.speech_worker is not None:
                self.speech_worker.release_after_reply()
            return

        if command == "start_vision":
            self.start_vision()
            speak("Vision mode started.")
            if self.speech_worker is not None:
                self.speech_worker.release_after_reply()
            return

        if command == "stop_vision":
            if self.vision_worker is not None and self.vision_worker.isRunning():
                self.vision_worker.stop()
            self.vision_btn.setText("👁  Start Vision")
            self.stop_sound()
            speak("Vision mode stopped.")
            if self.speech_worker is not None:
                self.speech_worker.release_after_reply()
            return

        if command == "read_text":
            self.handle_ocr()
            if self.speech_worker is not None:
                self.speech_worker.release_after_reply()
            return

        if response:
            self.log("Emotion: " + str(emotion))
            self.log("Assistant: " + response)
            self.update_status("Speaking response...")
            speak(response)

        if self.speech_worker is not None:
            self.speech_worker.release_after_reply()

    # ---------------- OCR ----------------

    def handle_ocr(self):
        if self.latest_frame is None:
            self.log("Start vision mode before using OCR.")
            return

        if self.ocr_worker is not None and self.ocr_worker.isRunning():
            self.log("OCR is already processing an image.")
            return

        self.ocr_worker = OCRWorker(self.latest_frame)
        self.ocr_worker.result_ready.connect(self.handle_ocr_result)
        self.ocr_worker.start()
        self.log("Reading text from camera...")
        self.update_status("OCR processing...")

    def handle_ocr_result(self, text):
        self.log("OCR result: " + text)
        self.update_status("OCR completed")
        if text and not text.startswith("OCR failed:") and not text.startswith("OCR error:"):
            speak(text)

    # ---------------- SOUND ----------------

    def start_sound(self):
        if self.sound_active:
            return
        try:
            start_sound_detection()
            self.sound_active = True
            self.sound_btn.setText("🔊  Stop Sound Monitoring")
            self.log("Environmental sound monitoring started.")
        except Exception as error:
            self.log(f"Sound monitoring error: {error}")

    def stop_sound(self):
        if not self.sound_active:
            return
        try:
            stop_sound_detection()
        except Exception as error:
            self.log(f"Sound stop error: {error}")
        self.sound_active = False
        self.sound_btn.setText("🔊  Start Sound Monitoring")
        self.log("Environmental sound monitoring stopped.")

    def toggle_sound(self):
        if self.sound_active:
            self.stop_sound()
        else:
            self.start_sound()

    # ---------------- EMERGENCY ----------------

    def handle_emergency(self):
        self.log("EMERGENCY ACTIVATED.")
        self.update_status("EMERGENCY ACTIVE")
        self.emergency_active = True
        emergency_alert()
        try:
            from emergency_module import start_emergency
            start_emergency()
        except Exception as error:
            self.log(f"Emergency alarm error: {error}")
        self.log("Local emergency alarm activated. Press Stop All to silence it.")

    def stop_emergency(self):
        if not self.emergency_active:
            return
        try:
            from emergency_module import stop_emergency
            stop_emergency()
        except Exception as error:
            self.log(f"Emergency shutdown error: {error}")
        self.emergency_active = False

    # ---------------- STOP ALL ----------------

    def stop_all(self):
        self.log("Stopping active features...")
        self.stop_speech()

        if self.vision_worker is not None and self.vision_worker.isRunning():
            self.vision_worker.stop()
            self.vision_worker.wait(3000)

        self.vision_btn.setText("👁  Start Vision")
        self.latest_frame = None
        self.camera_label.clear()
        self.camera_label.setText("Camera is currently inactive")

        self.stop_sound()
        self.stop_emergency()
        self.update_status("System ready")
        self.log("Active features stopped.")
        speak("Stopped.")

    def handle_error(self, message):
        self.log("ERROR: " + message)
        self.update_status("An error occurred")

    def closeEvent(self, event):
        self.log("Shutting down application...")

        if self.vision_worker is not None and self.vision_worker.isRunning():
            self.vision_worker.stop()
            self.vision_worker.wait(3000)

        if self.speech_worker is not None and self.speech_worker.isRunning():
            self.speech_worker.stop()
            self.speech_worker.wait(9000)

        self.stop_sound()
        self.stop_emergency()

        try:
            shutdown_tts(wait=False)
        except Exception as error:
            print(f"TTS shutdown warning: {error}")

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = AccessibilityApp()
    window.show()
    sys.exit(app.exec_())
