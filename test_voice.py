import pyttsx3

engine = pyttsx3.init('sapi5')   # 👈 IMPORTANT for Windows

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # or voices[1]

engine.setProperty('rate', 170)

engine.say("Voice test working now")
engine.runAndWait()