
"""
Daily Life Accessibility Companion
Natural Scene Understanding and Vision Module

Features:
- YOLOv8m object detection
- Lazy model initialisation
- Different confidence thresholds by object type
- Stable detection across consecutive frames
- Bounding-box visualisation
- Person/object spatial relationship estimation
- Natural scene descriptions
- Controlled voice announcement cooldown
- Existing five-value process_frame interface

IMPORTANT:
The system estimates visual relationships from bounding boxes.
It cannot reliably determine physical distance or prove that
a person is holding an object.
This is an experimental assistive prototype, not a certified
mobility or obstacle-avoidance system.
"""

import time
import threading

import cv2
from ultralytics import YOLO


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "yolov8m.pt"

IMAGE_SIZE = 640

PERSON_CONFIDENCE = 0.25
HANDHELD_CONFIDENCE = 0.20
OTHER_CONFIDENCE = 0.35

REQUIRED_STABLE_FRAMES = 3

ANNOUNCEMENT_COOLDOWN = 2.0
REPEAT_ANNOUNCEMENT_INTERVAL = 12.0

MAX_OBJECTS_TO_DESCRIBE = 3

# Objects that someone might commonly hold.
HANDHELD_OBJECTS = {
    "book",
    "bottle",
    "cell phone",
    "cup",
    "remote",
    "umbrella",
    "handbag",
    "backpack",
    "laptop",
    "sports ball",
    "keyboard",
    "mouse",
    "scissors",
    "toothbrush",
    "wine glass",
}

# ==================================================
# INTERNAL STATE
# ==================================================

_model = None
_model_lock = threading.Lock()

_detection_history = {}

_frame_count = 0

_last_announced_message = ""
_last_announcement_time = 0.0


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():

    global _model

    if _model is not None:
        return _model

    with _model_lock:

        if _model is not None:
            return _model

        print("\nLoading YOLOv8m vision model...")

        try:

            _model = YOLO(MODEL_PATH)

            print("YOLOv8m model loaded successfully.")

            return _model

        except Exception as error:

            print(f"Vision model loading error: {error}")

            raise RuntimeError(
                "Unable to load YOLO vision model."
            ) from error


# ==================================================
# OBJECT POSITION
# ==================================================

def get_position(center_x, frame_width):

    if frame_width <= 0:
        return "front"

    if center_x < frame_width * 0.35:
        return "left"

    if center_x > frame_width * 0.65:
        return "right"

    return "front"


# ==================================================
# OBJECT CONFIDENCE
# ==================================================

def get_confidence_threshold(label):

    if label == "person":
        return PERSON_CONFIDENCE

    if label in HANDHELD_OBJECTS:
        return HANDHELD_CONFIDENCE

    return OTHER_CONFIDENCE


# ==================================================
# LOAD OBJECT DETECTIONS
# ==================================================

def detect_objects(frame, model):

    frame_height, frame_width = frame.shape[:2]

    results = model.predict(
        source=frame,
        conf=0.15,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    objects = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            try:

                confidence = float(
                    box.conf[0].item()
                )

                class_id = int(
                    box.cls[0].item()
                )

                label = model.names[class_id]

                threshold = get_confidence_threshold(label)

                if confidence < threshold:
                    continue

                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value in coordinates
                ]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                position = get_position(
                    center_x,
                    frame_width
                )

                objects.append({

                    "label": label,
                    "confidence": confidence,
                    "position": position,

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "center": (
                        center_x,
                        center_y
                    ),

                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1)

                })

            except Exception as error:

                print(
                    f"Detection processing warning: {error}"
                )

    return objects


# ==================================================
# STABLE DETECTION
# ==================================================

def confirm_detections(objects):

    global _frame_count
    global _detection_history

    _frame_count += 1

    current_keys = set()

    for obj in objects:

        key = (
            obj["label"],
            obj["position"]
        )

        current_keys.add(key)

        previous = _detection_history.get(key)

        if (
            previous is not None
            and previous["last_seen"] == _frame_count - 1
        ):

            count = previous["count"] + 1

        else:

            count = 1

        _detection_history[key] = {
            "count": count,
            "last_seen": _frame_count
        }

    # Remove old detections.

    expired = [
        key
        for key, value in _detection_history.items()
        if value["last_seen"] < _frame_count - 1
    ]

    for key in expired:
        del _detection_history[key]

    stable = []

    for obj in objects:

        key = (
            obj["label"],
            obj["position"]
        )

        history = _detection_history.get(key)

        if (
            history is not None
            and history["count"] >= REQUIRED_STABLE_FRAMES
        ):

            stable.append(obj)

    return stable


# ==================================================
# PERSON / OBJECT RELATIONSHIP
# ==================================================

def object_is_inside_person(obj, person):

    """
    Estimates whether an object is spatially associated
    with a person's bounding box.

    This does NOT prove that the person is holding it.
    """

    ox1, oy1, ox2, oy2 = obj["box"]
    px1, py1, px2, py2 = person["box"]

    object_area = max(
        1,
        (ox2 - ox1) * (oy2 - oy1)
    )

    ix1 = max(ox1, px1)
    iy1 = max(oy1, py1)
    ix2 = min(ox2, px2)
    iy2 = min(oy2, py2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)

    intersection_area = (
        intersection_width * intersection_height
    )

    overlap_ratio = intersection_area / object_area

    center_x, center_y = obj["center"]

    center_inside = (
        px1 <= center_x <= px2
        and py1 <= center_y <= py2
    )

    return (
        overlap_ratio >= 0.35
        and center_inside
    )


# ==================================================
# NATURAL SCENE DESCRIPTION
# ==================================================

def build_scene_description(objects):

    if not objects:
        return "No objects confidently detected."

    people = [
        obj for obj in objects
        if obj["label"] == "person"
    ]

    handheld = [
        obj for obj in objects
        if obj["label"] in HANDHELD_OBJECTS
    ]

    descriptions = []

    used_objects = set()

    # ----------------------------------------------
    # PEOPLE AND OBJECTS
    # ----------------------------------------------

    for person in people:

        held_items = []

        for obj in handheld:

            if id(obj) in used_objects:
                continue

            if object_is_inside_person(obj, person):

                held_items.append(obj)

                used_objects.add(id(obj))

        if held_items:

            item_names = list(dict.fromkeys(
                obj["label"]
                for obj in held_items
            ))

            if len(item_names) == 1:

                description = (
                    f"A person appears to be holding "
                    f"a {item_names[0]} "
                    f"{person['position']}."
                )

            else:

                joined = ", ".join(item_names)

                description = (
                    f"A person appears to be holding "
                    f"{joined} {person['position']}."
                )

        else:

            if person["position"] == "front":

                description = (
                    "A person is standing in front of you."
                )

            else:

                description = (
                    f"A person is visible on your "
                    f"{person['position']}."
                )

        descriptions.append(description)

    # ----------------------------------------------
    # OBJECTS NOT ASSOCIATED WITH PEOPLE
    # ----------------------------------------------

    for obj in objects:

        if obj["label"] == "person":
            continue

        if id(obj) in used_objects:
            continue

        label = obj["label"]
        position = obj["position"]

        if position == "front":

            description = (
                f"A {label} is visible in front of you."
            )

        else:

            description = (
                f"A {label} is visible on your {position}."
            )

        descriptions.append(description)

    # ----------------------------------------------
    # LIMIT DESCRIPTION LENGTH
    # ----------------------------------------------

    descriptions = descriptions[
        :MAX_OBJECTS_TO_DESCRIBE
    ]

    return " ".join(descriptions)


# ==================================================
# ANNOUNCEMENT FILTER
# ==================================================

def should_announce(message):

    global _last_announced_message
    global _last_announcement_time

    current_time = time.monotonic()

    elapsed = (
        current_time - _last_announcement_time
    )

    if not message:
        return False

    changed = (
        message != _last_announced_message
    )

    if changed and elapsed >= ANNOUNCEMENT_COOLDOWN:

        _last_announced_message = message
        _last_announcement_time = current_time

        return True

    if (
        not changed
        and elapsed >= REPEAT_ANNOUNCEMENT_INTERVAL
    ):

        _last_announcement_time = current_time

        return True

    return False


# ==================================================
# DRAW DETECTIONS
# ==================================================

def draw_detections(frame, objects):

    for obj in objects:

        x1, y1, x2, y2 = obj["box"]

        label = obj["label"]
        confidence = obj["confidence"]

        text = (
            f"{label} {confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 220, 100),
            2
        )

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 100),
            2
        )

    return frame


# ==================================================
# MAIN FRAME PROCESSING
# ==================================================

def process_frame(frame):

    """
    Maintains compatibility with main_app.py.

    Returns:

        frame,
        description,
        navigation_message,
        speak_flag,
        spoken_message
    """

    if frame is None or frame.size == 0:

        return (
            frame,
            "",
            "",
            False,
            ""
        )

    try:

        model = load_model()

        # ------------------------------------------
        # DETECT OBJECTS
        # ------------------------------------------

        objects = detect_objects(
            frame,
            model
        )

        # ------------------------------------------
        # DRAW CURRENT DETECTIONS
        # ------------------------------------------

        frame = draw_detections(
            frame,
            objects
        )

        # ------------------------------------------
        # CONFIRM STABLE OBJECTS
        # ------------------------------------------

        stable_objects = confirm_detections(
            objects
        )

        if not stable_objects:

            return (
                frame,
                "Checking the scene...",
                "",
                False,
                ""
            )

        # ------------------------------------------
        # SORT BY CONFIDENCE
        # ------------------------------------------

        stable_objects.sort(
            key=lambda obj: obj["confidence"],
            reverse=True
        )

        # ------------------------------------------
        # NATURAL SCENE
        # ------------------------------------------

        scene = build_scene_description(
            stable_objects
        )

        message = scene

        # ------------------------------------------
        # VOICE ANNOUNCEMENT
        # ------------------------------------------

        if should_announce(message):

            print(
                "Vision announcement:",
                message
            )

            return (
                frame,
                scene,
                message,
                True,
                message
            )

        return (
            frame,
            scene,
            "",
            False,
            ""
        )

    except Exception as error:

        print(
            f"Vision processing error: {error}"
        )

        return (
            frame,
            "",
            "",
            False,
            ""
        )


# ==================================================
# MODEL STATUS
# ==================================================

def is_model_loaded():

    return _model is not None
