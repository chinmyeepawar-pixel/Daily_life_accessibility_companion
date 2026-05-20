# assistant_core.py

def analyze_scene(text):
    # Now the vision module already generates smart sentences
    return text


def process_command(text):
    text = text.lower()

    if "hello" in text:
        return "Hello! How can I assist you?"

    elif "who are you" in text:
        return "I am your AI Accessibility Companion."

    elif "help" in text:
        return "I can help you with vision, speech, and accessibility."

    else:
        return "Command not recognized"