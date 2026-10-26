"""Central settings for MediGuide AI.

Holds the API-key loader, all form options, urgency styles, colour palettes,
red-flag safety rules and sample test scenarios. Nothing in here calls the LLM.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Secrets: the key is read from .env (never hard-coded in the source code)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_env_api_key() -> str:
    """Return the OpenAI key from the environment / .env file (or empty string)."""
    return os.getenv("OPENAI_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------
APP_NAME = "MediGuide AI"
APP_TAGLINE = "AI-powered symptom guidance - educational, safety-first, never a diagnosis."
APP_DESCRIPTION = (
    "Enter basic information and symptoms and get structured, preliminary "
    "guidance generated with LangChain + OpenAI."
)

DISCLAIMER_SHORT = (
    "Educational AI prototype - NOT a doctor. It never gives a confirmed diagnosis. "
    "Always consult a qualified healthcare professional."
)
DISCLAIMER_LONG = (
    "MediGuide AI is an educational prototype. It is NOT a replacement for a licensed "
    "doctor, professional diagnosis, emergency service or medical treatment. "
    "If you think you may be having an emergency, contact your local emergency "
    "number immediately."
)
EMERGENCY_TEXT = (
    "Call your local emergency number NOW (for example 1122 in Pakistan, "
    "911 in the US, 112 in the EU) or go to the nearest emergency department."
)

# ---------------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------------
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Form options
# ---------------------------------------------------------------------------
GENDERS = ["Male", "Female", "Other", "Prefer not to say"]

SYMPTOM_OPTIONS = [
    "Fever", "Chills", "Cough", "Sore throat", "Runny nose", "Headache", "Fatigue",
    "Nausea", "Vomiting", "Diarrhea", "Constipation", "Abdominal pain", "Heartburn",
    "Chest pain", "Shortness of breath", "Palpitations", "Dizziness", "Fainting",
    "Muscle aches", "Joint pain", "Back pain", "Skin rash", "Itching", "Ear pain",
    "Eye redness", "Blurred vision", "Swollen glands", "Loss of taste or smell",
    "Loss of appetite", "Weight loss", "Night sweats", "Frequent urination",
    "Burning urination", "Numbness or tingling", "Swelling in legs", "Confusion",
    "Slurred speech", "Facial drooping", "Severe bleeding", "Anxiety", "Insomnia",
]

DURATION_OPTIONS = [
    "Less than 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "2-4 weeks",
    "More than a month",
]

LANGUAGES = [
    "English", "Urdu", "Hindi", "Arabic", "Spanish", "French", "German",
    "Portuguese", "Turkish", "Bengali", "Indonesian",
]
RTL_LANGUAGES = {"Urdu", "Arabic"}
# Languages the built-in PDF font (Latin-1) can print. Others get an HTML report.
PDF_SAFE_LANGUAGES = {"English", "Spanish", "French", "German", "Portuguese", "Indonesian"}

# ---------------------------------------------------------------------------
# Urgency levels
# ---------------------------------------------------------------------------
URGENCY_LEVELS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]
URGENCY_STYLE = {
    "LOW": {
        "emoji": "🟢", "c1": "#22c55e", "c2": "#0d9488",
        "advice": "Symptoms look mild. Rest, hydrate and monitor how you feel.",
    },
    "MEDIUM": {
        "emoji": "🟡", "c1": "#f59e0b", "c2": "#ea580c",
        "advice": "Consider booking a visit with a healthcare professional in the next few days.",
    },
    "HIGH": {
        "emoji": "🟠", "c1": "#f97316", "c2": "#dc2626",
        "advice": "Please contact a healthcare professional or urgent care today.",
    },
    "EMERGENCY": {
        "emoji": "🔴", "c1": "#ef4444", "c2": "#9f1239",
        "advice": "Seek emergency medical help immediately. Do not wait.",
    },
}

# ---------------------------------------------------------------------------
# Local safety pre-screen (runs BEFORE the model is called)
# Each rule = (list of keyword groups, message). A rule fires when EVERY group
# has at least one keyword present in the user's text.
# ---------------------------------------------------------------------------
_CHEST = ["chest pain", "chest tightness", "pressure in chest", "pain in my chest"]
_BREATH = ["shortness of breath", "difficulty breathing", "can't breathe", "cannot breathe",
           "breathless", "trouble breathing"]
_STROKE = ["facial drooping", "face drooping", "slurred speech", "weakness on one side",
           "sudden confusion", "one side of my body"]
_BLEED = ["severe bleeding", "vomiting blood", "coughing blood", "coughing up blood",
          "bleeding heavily"]
_CONSCIOUS = ["fainting", "fainted", "loss of consciousness", "unconscious", "seizure"]
_ALLERGY = ["throat closing", "swollen tongue", "swelling of the throat", "anaphylaxis"]

RED_FLAG_RULES = [
    ([_CHEST, _BREATH], "Chest pain together with breathing difficulty can be a medical emergency."),
    ([_STROKE], "These symptoms can be signs of a stroke, which is time-critical."),
    ([_BLEED], "Severe bleeding needs urgent medical attention."),
    ([_CONSCIOUS], "Fainting, seizures or loss of consciousness need urgent medical evaluation."),
    ([_ALLERGY], "Throat or tongue swelling can signal a severe allergic reaction."),
]

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "self harm",
    "hurt myself", "want to die",
]
CRISIS_MESSAGE = (
    "You deserve support right now. If you are thinking about harming yourself, please "
    "contact your local emergency number or a crisis line immediately - findahelpline.com "
    "lists free services by country - or reach out to someone you trust."
)

# ---------------------------------------------------------------------------
# Colour palettes for the animated theme engine
# Each palette has 4 gradient colours for dark mode and for light mode.
# ---------------------------------------------------------------------------
PALETTES = {
    "Aurora": {
        "accent": "#7c4dff", "accent2": "#00d4ff",
        "dark": ("#0f0c29", "#302b63", "#24243e", "#0b6e8a"),
        "light": ("#e0f7ff", "#efe6ff", "#e8fff3", "#fff0f6"),
    },
    "Sunset": {
        "accent": "#ff6b6b", "accent2": "#feca57",
        "dark": ("#2b0a3d", "#6a1b4d", "#b03a48", "#3d1a78"),
        "light": ("#fff1eb", "#ffe0e9", "#fff6d6", "#ffd9c9"),
    },
    "Ocean": {
        "accent": "#0072ff", "accent2": "#00e0c6",
        "dark": ("#021b3a", "#05445e", "#0b6e8a", "#03396c"),
        "light": ("#e6f7ff", "#d6f0ff", "#e0fff9", "#e8f0ff"),
    },
    "Forest": {
        "accent": "#2ecc71", "accent2": "#a8e063",
        "dark": ("#06201a", "#0b3d2e", "#14532d", "#1d4e5f"),
        "light": ("#effff3", "#e3fbe8", "#f3ffe0", "#e0f7f2"),
    },
}

# ---------------------------------------------------------------------------
# One-click sample scenarios (map directly onto the form's session keys)
# ---------------------------------------------------------------------------
SAMPLE_SCENARIOS = {
    "🤧 Common cold": {
        "f_age": "25", "f_gender": "Female", "f_symptoms": ["Runny nose", "Sore throat"],
        "f_free": "", "f_duration": "1-3 days", "f_severity": 2,
        "f_cond": "", "f_meds": "", "f_notes": "", "f_lang": "English",
    },
    "🌡️ Fever & cough": {
        "f_age": "40", "f_gender": "Male", "f_symptoms": ["Fever", "Cough"],
        "f_free": "", "f_duration": "4-7 days", "f_severity": 6,
        "f_cond": "", "f_meds": "", "f_notes": "Feeling very tired", "f_lang": "English",
    },
    "🫀 Chest pain": {
        "f_age": "58", "f_gender": "Male", "f_symptoms": ["Chest pain", "Shortness of breath"],
        "f_free": "", "f_duration": "Less than 24 hours", "f_severity": 9,
        "f_cond": "High blood pressure", "f_meds": "", "f_notes": "", "f_lang": "English",
    },
    "🌐 Urdu cold": {
        "f_age": "25", "f_gender": "Female", "f_symptoms": ["Runny nose", "Sore throat"],
        "f_free": "", "f_duration": "1-3 days", "f_severity": 2,
        "f_cond": "", "f_meds": "", "f_notes": "", "f_lang": "Urdu",
    },
}
