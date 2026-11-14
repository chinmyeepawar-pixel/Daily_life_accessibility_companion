"""
Daily Life Accessibility Companion - AI Brain

Capabilities:
- Current time and date in India Standard Time (fixed UTC+05:30)
- Live weather for Indian cities using Open-Meteo (no API key)
- Defaults to Kolhapur, Maharashtra when no city is mentioned
- Ollama Phi-3 for general conversational questions
- No Transformers/emotion model is loaded
- Network timeouts prevent indefinite waiting

Windows compatibility:
Uses a fixed IST offset instead of ZoneInfo, so Python installations
without the optional IANA timezone database (tzdata) still work.
"""

import re
import threading
from datetime import datetime, timezone, timedelta

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_CITY = "Kolhapur"
INDIA_TZ = timezone(timedelta(hours=5, minutes=30), name="IST")

SYSTEM_PROMPT = (
    "You are a calm, supportive accessibility assistant and navigation guide. "
    "Speak naturally, warmly, and clearly. Keep replies concise, usually one or two sentences. "
    "Never claim to have checked live information unless a live tool has actually supplied it. "
    "If live weather, time, or date is requested, use the supplied real data exactly."
)

chat_history = []
_history_lock = threading.Lock()
REQUEST_TIMEOUT = (5, 12)
OLLAMA_TIMEOUT = (3, 25)
MAX_HISTORY_MESSAGES = 8

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle",
    53: "moderate drizzle", 55: "dense drizzle", 56: "light freezing drizzle",
    57: "dense freezing drizzle", 61: "slight rain", 63: "moderate rain",
    65: "heavy rain", 66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains", 80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers",
    86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _remember(user_text, assistant_text):
    with _history_lock:
        chat_history.append(f"User: {user_text}")
        chat_history.append(f"Assistant: {assistant_text}")
        del chat_history[:-MAX_HISTORY_MESSAGES]


def _get_city_from_question(text):
    match = re.search(
        r"\b(?:weather|temperature|forecast)\b.*?\b(?:in|for|at)\s+"
        r"([a-zA-Z][a-zA-Z .'-]{1,45}?)(?:[?.!,]|$)",
        text,
        flags=re.IGNORECASE
    )
    if match:
        city = match.group(1).strip()
        city = re.sub(
            r"\b(today|tomorrow|right now|now|currently|please|india)\b",
            "",
            city,
            flags=re.IGNORECASE
        ).strip(" ,.")
        if city:
            return city
    return DEFAULT_CITY


def _get_weather(city):
    try:
        geo_response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": city,
                "count": 10,
                "language": "en",
                "format": "json",
                "countryCode": "IN",
            },
            timeout=REQUEST_TIMEOUT,
        )
        geo_response.raise_for_status()
        results = geo_response.json().get("results") or []

        place = next(
            (item for item in results if item.get("country_code") == "IN"),
            None
        )
        if place is None:
            return f"I couldn't find {city} in India. Please say another Indian city."

        latitude = place["latitude"]
        longitude = place["longitude"]
        place_name = place.get("name", city)
        admin = place.get("admin1", "")

        weather_response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,precipitation"
                ),
                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min",
                "forecast_days": 1,
                "timezone": "Asia/Kolkata",
            },
            timeout=REQUEST_TIMEOUT,
        )
        weather_response.raise_for_status()
        payload = weather_response.json()

        current = payload.get("current", {})
        daily = payload.get("daily", {})
        if "temperature_2m" not in current:
            return f"I couldn't retrieve current weather for {place_name} just now."

        code = current.get("weather_code")
        description = WEATHER_CODES.get(code, "conditions not specified")
        temperature = round(current["temperature_2m"])
        feels_like = round(current.get("apparent_temperature", temperature))
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        rain_chance = (daily.get("precipitation_probability_max") or [None])[0]
        high = (daily.get("temperature_2m_max") or [None])[0]
        low = (daily.get("temperature_2m_min") or [None])[0]

        location = f"{place_name}, {admin}" if admin else place_name
        parts = [
            f"Current weather in {location}: {temperature} degrees Celsius and {description}.",
            f"It feels like {feels_like} degrees."
        ]
        if humidity is not None:
            parts.append(f"Humidity is {humidity} percent.")
        if wind is not None:
            parts.append(f"Wind speed is {round(wind)} kilometres per hour.")
        if rain_chance is not None:
            parts.append(f"Today's maximum chance of precipitation is {round(rain_chance)} percent.")
        if low is not None and high is not None:
            parts.append(f"Today's expected temperature range is {round(low)} to {round(high)} degrees.")
        return " ".join(parts)

    except requests.exceptions.Timeout:
        return "The weather service is taking too long to respond. Please try again shortly."
    except requests.exceptions.ConnectionError:
        return "I can't reach the weather service. Please check your internet connection."
    except (requests.exceptions.RequestException, ValueError, KeyError) as error:
        print(f"Weather lookup error: {error}")
        return "I couldn't retrieve the weather right now. Please try again."


def _is_weather_question(text):
    return bool(re.search(
        r"\b(weather|temperature|forecast|rain|raining|hot outside|cold outside)\b",
        text, re.IGNORECASE
    ))


def _is_time_question(text):
    return bool(re.search(
        r"\b(what time|current time|time is it|tell me the time)\b",
        text, re.IGNORECASE
    ))


def _is_date_question(text):
    return bool(re.search(
        r"\b(what date|today's date|todays date|what day is it|which day is it)\b",
        text, re.IGNORECASE
    ))


def _build_prompt(user_input):
    with _history_lock:
        history = "\n".join(chat_history[-MAX_HISTORY_MESSAGES:])
    return f"{SYSTEM_PROMPT}\n{history}\nUser: {user_input}\nAssistant:"


def _general_ai_response(user_text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": _build_prompt(user_text),
            "stream": False,
            "options": {
                "num_predict": 55,
                "temperature": 0.6,
                "repeat_penalty": 1.2,
                "stop": ["User:"]
            }
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    reply = str(response.json().get("response", "")).strip()
    return reply or "I'm here. Could you tell me a little more?"


def generate_response(text):
    """Return (response, emotion_label), compatible with the current main_app.py."""
    user_text = str(text).strip() if text is not None else ""
    if not user_text:
        return "I didn't catch that. Could you say it again?", "not_used"

    now = datetime.now(INDIA_TZ)

    if _is_weather_question(user_text):
        reply = _get_weather(_get_city_from_question(user_text))
        _remember(user_text, reply)
        return reply, "not_used"

    if _is_time_question(user_text):
        reply = f"The current time in India is {now.strftime('%I:%M %p')} IST."
        _remember(user_text, reply)
        return reply, "not_used"

    if _is_date_question(user_text):
        reply = f"Today is {now.strftime('%A, %d %B %Y')} in India."
        _remember(user_text, reply)
        return reply, "not_used"

    try:
        reply = _general_ai_response(user_text)
        _remember(user_text, reply)
        print(f"AI response generated: {reply}")
        return reply, "not_used"

    except requests.exceptions.ConnectionError as error:
        print(f"AI connection error: {error}")
        return "I can't connect to the local AI right now. Please check that Ollama is running.", "not_used"
    except requests.exceptions.Timeout as error:
        print(f"AI request timed out: {error}")
        return "The AI is taking too long to respond. Please try asking me again.", "not_used"
    except requests.exceptions.RequestException as error:
        print(f"AI request error: {error}")
        return "I'm having trouble generating a response right now. Please try again.", "not_used"
    except (ValueError, KeyError) as error:
        print(f"AI response parsing error: {error}")
        return "I received an unexpected response from the local AI. Please try again.", "not_used"
    except Exception as error:
        print(f"Unexpected AI error: {error}")
        return "Something went wrong while preparing my response. Please try again.", "not_used"


def clear_chat_history():
    with _history_lock:
        chat_history.clear()
