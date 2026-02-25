import os
import json
import re
import string
import nltk
from flask import Flask, render_template, request, jsonify, session
from fuzzywuzzy import fuzz, process

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "moviebot-secret-2024")

# ── Load character database ───────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "characters.json")
with open(DB_PATH, "r", encoding="utf-8") as f:
    CHARACTERS: list[dict] = json.load(f)

# Build a lookup table: all names + aliases → character dict
CHAR_LOOKUP: dict[str, dict] = {}
for char in CHARACTERS:
    CHAR_LOOKUP[char["name"].lower()] = char
    for alias in char.get("aliases", []):
        CHAR_LOOKUP[alias.lower()] = char

ALL_NAMES = list(CHAR_LOOKUP.keys())

# Common English stop-words
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

# ── Intent keywords ───────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "actor": [
        "who played", "who plays", "actor", "actress", "voice", "portrayed by",
        "who voices", "cast", "who is the actor",
    ],
    "personality": [
        "personality", "character", "traits", "describe", "what kind",
        "what type", "how is", "nature", "attitude", "behavior",
    ],
    "movies": [
        "movies", "films", "appear", "appeared in", "which movie",
        "what movie", "which film", "filmography", "feature",
    ],
    "facts": [
        "fact", "trivia", "interesting", "did you know", "fun fact",
        "tell me something", "little known", "cool",
    ],
    "background": [
        "who is", "tell me about", "background", "history", "story",
        "origin", "about", "explain", "describe", "what do you know",
        "give me info", "information about",
    ],
}


def detect_intent(text: str) -> str:
    """Return the most likely intent based on keyword matching."""
    lower = text.lower()
    for intent, keywords in INTENT_PATTERNS.items():
        for kw in keywords:
            if kw in lower:
                return intent
    return "background"  # default


def extract_character(text: str) -> dict | None:
    """
    Try to find a matching character from the text.
    Uses exact substring matching first, then fuzzy matching.
    """
    lower_text = text.lower()

    # 1. Exact substring match (longest match wins)
    matched_key = None
    matched_len = 0
    for key in ALL_NAMES:
        if key in lower_text and len(key) > matched_len:
            matched_key = key
            matched_len = len(key)
    if matched_key:
        return CHAR_LOOKUP[matched_key]

    # 2. Fuzzy match — extract meaningful words
    tokens = word_tokenize(lower_text)
    content_words = [
        t for t in tokens
        if t not in STOP_WORDS
        and t not in string.punctuation
        and len(t) > 2
    ]
    if not content_words:
        return None

    # Try pairs then single words
    candidates = []
    for i in range(len(content_words) - 1):
        candidates.append(f"{content_words[i]} {content_words[i+1]}")
    candidates.extend(content_words)

    best_match = None
    best_score = 0
    for candidate in candidates:
        result = process.extractOne(candidate, ALL_NAMES, scorer=fuzz.token_set_ratio)
        if result and result[1] > best_score:
            best_score = result[1]
            best_match = result[0]

    if best_match and best_score >= 72:
        return CHAR_LOOKUP[best_match]

    return None


def build_response(char: dict, intent: str) -> str:
    """Build a natural-language response from character data."""
    name = char["name"]

    if intent == "actor":
        actor = char.get("actor", "Unknown")
        movies_str = ", ".join(char.get("movies", [])[:2])
        return (
            f"🎭 <b>{name}</b> is portrayed by <b>{actor}</b>. "
            f"Notable appearances include {movies_str}."
        )

    elif intent == "personality":
        personality = char.get("personality", "No personality data available.")
        return f"🧠 <b>Personality — {name}:</b><br>{personality}"

    elif intent == "movies":
        movies = char.get("movies", [])
        if not movies:
            return f"🎬 I don't have movie data for <b>{name}</b>."
        movie_list = "".join(f"<li>{m}</li>" for m in movies)
        return f"🎬 <b>{name}</b> appears in:<ul>{movie_list}</ul>"

    elif intent == "facts":
        facts = char.get("facts", [])
        if not facts:
            return f"💡 I don't have trivia about <b>{name}</b> yet."
        facts_list = "".join(f"<li>{f}</li>" for f in facts[:3])
        return f"💡 <b>Fun Facts about {name}:</b><ul>{facts_list}</ul>"

    else:  # background / default
        background = char.get("background", "No background information available.")
        actor = char.get("actor", "Unknown")
        movies = char.get("movies", [])
        first_movie = movies[0] if movies else "N/A"
        return (
            f"🎬 <b>{name}</b><br>"
            f"<b>Actor:</b> {actor}<br>"
            f"<b>First Appearance:</b> {first_movie}<br><br>"
            f"{background}"
        )


def greet_response(text: str) -> str | None:
    """Return a greeting response if the input is a greeting."""
    greetings = ["hi", "hello", "hey", "howdy", "hiya", "sup", "greetings"]
    lower = text.lower().strip()
    if any(lower.startswith(g) for g in greetings):
        return (
            "👋 Hello! I'm <b>MovieBot</b> — your AI movie character expert!<br>"
            "You can ask me things like:<br>"
            "<i>\"Tell me about Tony Stark\"</i><br>"
            "<i>\"Who played the Joker?\"</i><br>"
            "<i>\"What is Harry Potter's personality?\"</i>"
        )
    return None


def help_response(text: str) -> str | None:
    """Return a help response if user asks for help."""
    if re.search(r"\bhelp\b|\bwhat can you do\b|\bcommands\b", text.lower()):
        return (
            "🤖 <b>I can answer questions about movie characters!</b><br>"
            "Try asking:<br>"
            "• <i>Who is [character name]?</i><br>"
            "• <i>Who played [character name]?</i><br>"
            "• <i>What is the personality of [character]?</i><br>"
            "• <i>What movies does [character] appear in?</i><br>"
            "• <i>Tell me a fun fact about [character]</i><br><br>"
            "Characters in my database include Tony Stark, Joker, Harry Potter, Batman, "
            "Darth Vader, James Bond, Spider-Man, Hermione Granger, and many more!"
        )
    return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request. Send JSON with 'message' key."}), 400

    user_msg: str = data["message"].strip()
    if not user_msg:
        return jsonify({"response": "Please type a message! 😊"})

    # Store in session history
    history = session.get("history", [])
    history.append({"role": "user", "content": user_msg})

    # ── Special responses ──────────────────────────────────────────────────────
    greeting = greet_response(user_msg)
    if greeting:
        history.append({"role": "bot", "content": greeting})
        session["history"] = history[-20:]  # keep last 20 turns
        return jsonify({"response": greeting})

    help_msg = help_response(user_msg)
    if help_msg:
        history.append({"role": "bot", "content": help_msg})
        session["history"] = history[-20:]
        return jsonify({"response": help_msg})

    # ── NLP pipeline ───────────────────────────────────────────────────────────
    intent = detect_intent(user_msg)
    character = extract_character(user_msg)

    if character is None:
        # Check if user listed a name we partially know
        response = (
            "🤔 I'm not sure which movie character you're asking about. "
            "Could you be more specific? For example:<br>"
            "<i>\"Tell me about Tony Stark\"</i> or <i>\"Who is the Joker?\"</i><br><br>"
            "My database includes characters like Tony Stark, Batman, Joker, Harry Potter, "
            "Darth Vader, James Bond, Spider-Man, Hermione Granger, and more!"
        )
    else:
        response = build_response(character, intent)

    history.append({"role": "bot", "content": response})
    session["history"] = history[-20:]

    return jsonify({"response": response, "character": character["name"] if character else None})


@app.route("/characters", methods=["GET"])
def list_characters():
    """Return all character names for the suggestion chips."""
    names = [c["name"] for c in CHARACTERS]
    return jsonify({"characters": names})


@app.route("/clear", methods=["POST"])
def clear_history():
    session["history"] = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
