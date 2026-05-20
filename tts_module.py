import pyttsx3

def speak(text):
    try:
        engine = pyttsx3.init()

        # 🔥 Voice tuning
        engine.setProperty('rate', 160)   # slower = calmer
        engine.setProperty('volume', 1.0)

        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)  # female voice (usually softer)

        print("SPEAK:", text)

        engine.say(text)
        engine.runAndWait()
        engine.stop()

    except Exception as e:
        print("TTS Error:", e)