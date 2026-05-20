from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
import cv2
import threading

from ultralytics import YOLO
from assistant_core import analyze_scene
from tts_module import speak

model = YOLO("yolov8n.pt")


# 🎨 Card Widget
class Card(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.1, 0.12, 0.2, 1)
            self.rect = RoundedRectangle(radius=[20])
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


class MainUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=15, spacing=15, **kwargs)

        # 🧠 Title
        self.title = Label(
            text="AI Accessibility Companion",
            font_size=22,
            bold=True,
            size_hint=(1, 0.1)
        )
        self.add_widget(self.title)

        # 📊 Status Card
        self.status_card = Card(size_hint=(1, 0.15), padding=10)
        self.status = Label(text="System Ready", font_size=16)
        self.status_card.add_widget(self.status)
        self.add_widget(self.status_card)

        # 📷 Camera Card
        self.camera_card = Card(size_hint=(1, 0.5))
        self.img = Image()
        self.camera_card.add_widget(self.img)
        self.add_widget(self.camera_card)

        # 🎛 Buttons
        btn_layout = BoxLayout(size_hint=(1, 0.25), spacing=10)

        self.start_btn = Button(
            text="Start",
            background_color=(0.3, 0.4, 1, 1)
        )
        self.start_btn.bind(on_press=self.start_camera)

        self.stop_btn = Button(
            text="Stop",
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.stop_btn.bind(on_press=self.stop_camera)

        self.emergency_btn = Button(
            text="EMERGENCY",
            background_color=(1, 0.2, 0.2, 1)
        )
        self.emergency_btn.bind(on_press=self.emergency)

        btn_layout.add_widget(self.start_btn)
        btn_layout.add_widget(self.stop_btn)
        btn_layout.add_widget(self.emergency_btn)

        self.add_widget(btn_layout)

        self.capture = None
        self.running = False

    # 📷 Start Camera
    def start_camera(self, instance):
        if self.running:
            return

        self.capture = cv2.VideoCapture(0)
        self.running = True
        Clock.schedule_interval(self.update_frame, 1.0 / 24)

    # 🛑 Stop Camera
    def stop_camera(self, instance):
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    # 🚨 Emergency
    def emergency(self, instance):
        self.status.text = "🚨 EMERGENCY"
        threading.Thread(target=speak, args=("Emergency! Please help!",), daemon=True).start()

    # 🔄 Frame Update
    def update_frame(self, dt):
        if not self.running or not self.capture:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        height, width, _ = frame.shape
        results = model(frame)

        detected = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2

                if center_x < width / 3:
                    pos = "left"
                elif center_x < 2 * width / 3:
                    pos = "center"
                else:
                    pos = "right"

                detected.append((label, pos))

        if detected:
            guidance = analyze_scene(detected)
            self.status.text = guidance
            threading.Thread(target=speak, args=(guidance,), daemon=True).start()

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture


class MobileApp(App):
    def build(self):
        return MainUI()


if __name__ == "__main__":
    MobileApp().run()