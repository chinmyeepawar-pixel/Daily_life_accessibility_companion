
from speech_module import speech_to_text, is_model_loaded

print("\nDaily Life Accessibility Companion")
print("Speech Recognition Test")

print("\nWhisper loaded before recording:", is_model_loaded())

print("\nPlease speak clearly after recording starts.")

recognized_text = speech_to_text()

print("\nRecognized result:")

if recognized_text:
    print(recognized_text)
else:
    print("No speech recognised. Please try again.")

print("\nWhisper loaded after recording:", is_model_loaded())

print("\nSpeech test completed.")
