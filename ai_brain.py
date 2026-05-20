import requests
from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

OLLAMA_URL = "http://localhost:11434/api/generate"

# 🔥 Conversation memory
chat_history = []

# 🔥 Personality + behavior
SYSTEM_PROMPT = (
    "You are a calm, supportive accessibility assistant and navigation guide. "
    "You speak like a human companion. "
    "Keep replies short (1 sentence). "
    "Be helpful, warm, and clear. "
    "If user feels emotional, respond gently. "
    "If user asks something, respond naturally."
)


def build_prompt(user_input):
    global chat_history

    # Add user input to history
    chat_history.append(f"User: {user_input}")

    # Keep only last 5 messages (memory limit)
    history = "\n".join(chat_history[-5:])

    prompt = f"{SYSTEM_PROMPT}\n{history}\nAssistant:"

    return prompt


def generate_response(text):
    try:
        result = emotion_model(text)
        emotion = result[0]['label']

        prompt = build_prompt(text)

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 25,
                    "temperature": 0.6,
                    "repeat_penalty": 1.2,
                    "stop": ["User:", "Assistant:"]
                }
            }
        )

        reply = response.json()["response"].strip()

        # 🔥 Save assistant reply
        chat_history.append(f"Assistant: {reply}")

        return reply, emotion

    except Exception as e:
        print("AI Error:", e)
        return "I'm here. Tell me what you need.", "unknown"