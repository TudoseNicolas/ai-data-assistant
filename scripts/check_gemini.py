"""Check which Gemini models your API key can actually use.

Run from the project folder (where .env is):   python scripts/check_gemini.py
It sends a tiny "Reply with OK" to each text model and prints what Google answers.
"""
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key:
    sys.exit("No GEMINI_API_KEY found. Run this from the folder that contains your .env file.")

client = genai.Client(api_key=key)
print("Key loaded (ends in …" + key[-4:] + "). Listing models...\n")

try:
    models = []
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if (not actions or "generateContent" in actions) and "gemini" in name and not any(
            w in name for w in ("image", "tts", "audio", "live", "embedding", "robotics")
        ):
            models.append(name)
except Exception as exc:
    sys.exit(f"Could not list models: {exc}")

print(f"{len(models)} text models found. Testing each one:\n")
working = []
for name in sorted(models, reverse=True):
    try:
        reply = client.models.generate_content(model=name, contents="Reply with OK")
        print(f"  OK     {name}: {(reply.text or '').strip()[:20]}")
        working.append(name)
    except Exception as exc:
        code = getattr(exc, "code", "")
        print(f"  FAIL   {name}: {code} {str(exc)[:120]}")

print()
if working:
    print(f"Working models: {', '.join(working)}")
    print(f"Put this line in your .env file:   GEMINI_MODEL={working[0]}")
else:
    print("No model worked with this key. Check your quota in Google AI Studio, or use Local (Ollama).")
