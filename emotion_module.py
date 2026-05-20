from deepface import DeepFace

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False  # 🔥 prevents crashes
        )
        return result[0]['dominant_emotion']
    except:
        return None