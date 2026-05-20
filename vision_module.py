from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

last_message = ""
last_time = 0
NAV_DELAY = 4  # 🔥 increased delay


def get_position(cx, w):
    if cx < w * 0.35:
        return "left"
    elif cx > w * 0.65:
        return "right"
    return "front"


def get_distance(size):
    if size > 320:
        return "very_close"
    elif size > 180:
        return "close"
    return "far"


def describe(label, pos, dist):
    # 🔥 Clean natural phrasing (NO repetition)
    if dist == "very_close":
        if pos == "front":
            return f"a {label} right in front of you"
        else:
            return f"a {label} very close on your {pos}"

    if dist == "close":
        if pos == "front":
            return f"a {label} ahead"
        else:
            return f"a {label} on your {pos}"

    # far
    if pos == "front":
        return f"a {label} further ahead"
    else:
        return f"a {label} on your {pos}"


def process_frame(frame):
    global last_message, last_time

    results = model(frame, verbose=False)

    frame_width = frame.shape[1]
    objects = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            size = float(x2 - x1)

            pos = get_position(cx, frame_width)
            dist = get_distance(size)

            objects.append((label, pos, dist, size))

    # 🔹 No objects
    if not objects:
        msg = "The path looks clear, you can move forward."
        if msg != last_message:
            last_message = msg
            return frame, "", msg, True, msg
        return frame, "", "", False, ""

    # 🔹 Sort by closeness
    objects.sort(key=lambda x: -x[3])

    descriptions = []
    danger = False

    # 🔥 Avoid duplicate labels
    seen = set()

    for obj in objects[:3]:
        label, pos, dist, _ = obj

        if label in seen:
            continue
        seen.add(label)

        if dist == "very_close":
            danger = True

        descriptions.append(describe(label, pos, dist))

    scene = ", ".join(descriptions)

    # 🔥 NATURAL RESPONSE
    if danger:
        msg = f"There’s {scene}. Just pause for a moment."
    else:
        msg = f"I can see {scene}. You’re okay to move."

    # 🔥 ANTI-REPETITION FILTER
    current_time = time.time()

    if (
        msg != last_message and
        current_time - last_time > NAV_DELAY
    ):
        last_message = msg
        last_time = current_time
        return frame, scene, msg, True, msg

    return frame, scene, "", False, ""