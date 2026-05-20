import sys
import cv2
import time

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

from vision_module import process_frame
from speech_module import speech_to_text
from tts_module import speak
from sound_module import start_sound_detection, stop_sound_detection, emergency_alert
from ai_brain import generate_response


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Accessibility Companion")
        self.setGeometry(200, 50, 700, 800)

        layout = QVBoxLayout()

        self.camera = QLabel()
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.mode = None
        self.last_message = ""

        # Buttons
        self.vision_btn = QPushButton("👁️ Vision Mode")
        self.speech_btn = QPushButton("🎤 Speak")
        self.stop_btn = QPushButton("⛔ Stop")
        self.emergency_btn = QPushButton("🚨 Emergency")

        layout.addWidget(self.camera)
        layout.addWidget(self.output)
        layout.addWidget(self.vision_btn)
        layout.addWidget(self.speech_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.emergency_btn)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Button connections
        self.vision_btn.clicked.connect(self.start_vision)
        self.speech_btn.clicked.connect(self.handle_speech)
        self.stop_btn.clicked.connect(self.stop_all)
        self.emergency_btn.clicked.connect(self.handle_emergency)

    # ---------------- VISION ----------------

    def start_vision(self):
        self.mode = "vision"
        speak("Vision mode started")
        start_sound_detection()
        self.timer.start(120)

    def stop_all(self):
        self.mode = None
        self.timer.stop()
        stop_sound_detection()
        speak("Stopped")

    def update_frame(self):
        if self.mode != "vision":
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame, desc, nav, speak_flag, message = process_frame(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(img))

        if nav:
            self.output.setText("🧭 " + nav)
        elif desc:
            self.output.setText("👁️ " + desc)

        if speak_flag and message:
            if self.last_message != message:
                self.last_message = message
                speak(message)

    # ---------------- SPEECH ----------------

    def handle_speech(self):
        speak("Listening")

        text = speech_to_text()

        if not text:
            speak("Please try again")
            return

        text_lower = text.lower()

        # Show input
        self.output.setText(f"🗣️ {text}")

        # 🔥 COMMANDS (priority)
        if "emergency" in text_lower:
            self.handle_emergency()
            return

        if "start vision" in text_lower:
            self.start_vision()
            return

        if "stop vision" in text_lower:
            self.stop_all()
            return

        # 🤖 AI RESPONSE
        response, emotion = generate_response(text)

        self.output.setText(
            f"🗣️ {text}\n\n🧠 Emotion: {emotion}\n🤖 {response}"
        )

        # 🔥 Smooth delay (natural feel)
        time.sleep(0.3)
        speak(response)

    # ---------------- EMERGENCY ----------------

    def handle_emergency(self):
        self.output.setText("🚨 Emergency")
        speak("Emergency activated")
        emergency_alert()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())