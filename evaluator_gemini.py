"""
Ad image vision analysis via Gemini.
Reference implementation — requires a Google AI Studio API key (free tier available).
Sends an ad image to Gemini and returns structured violation, UX, and audience data.

Setup:
    pip install google-generativeai
    Add GEMINI_API_KEY=your_key to .env  (get one free at aistudio.google.com)

Flow:
    1. Load and resize the image (path or URL → PIL Image)
    2. Configure the Gemini client with the API key
    3. Send the image + prompt to the model
    4. Strip any markdown fences from the raw response text
    5. Extract the JSON object and parse it
    6. Return the structured result dict (or an error dict on failure)
"""
import io
import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import requests
from PIL import Image

from policies import VIOLATIONS

# Step 1 — Choose the model.
# gemini-2.0-flash is the recommended free-tier model: fast, multimodal, handles images well.
MODEL = "gemini-2.0-flash"

# Step 2 — Build the violation list from policies.py so the prompt stays in sync
# with any policy changes automatically.
VIOLATION_LIST = "\n".join(
    f'- {v["id"]}: {v["description"]}' for v in VIOLATIONS
)

# Step 3 — Define the evaluation prompt.
# The same structured prompt used by the Ollama evaluator, formatted for Gemini.
# Gemini accepts the prompt as a plain string alongside the image object.
PROMPT = f"""You are a Search Ads policy compliance expert and UX evaluator.

Analyze the provided ad image and return a structured evaluation across three areas.

---

PART 1 — POLICY VIOLATIONS
Check for each of the following violations:
{VIOLATION_LIST}

For each violation, report whether it was detected and provide a one-sentence justification.

PART 2 — UX SCORE
Rate the overall user experience of the ad image on a scale of 1–10, considering:
- Visual clarity and composition
- Message legibility
- Use of space
- Overall ad effectiveness

PART 3 — AUDIENCE SUITABILITY
Classify the image content for the following age groups and indicate whether it is appropriate for each:
- under_13
- 13_to_17
- 18_plus
- all_ages

Also suggest the most appropriate target audience based on the visual content and tone.

---

Reply with JSON only — no other text:
{{
  "violations": [
    {{
      "id": "<violation id>",
      "detected": <true|false>,
      "evidence": "<one sentence>"
    }}
  ],
  "ux_score": <float 1.0-10.0>,
  "ux_notes": "<two to three sentences on strengths and weaknesses>",
  "audience": {{
    "under_13": <true|false>,
    "13_to_17": <true|false>,
    "18_plus": <true|false>,
    "all_ages": <true|false>,
    "recommended_targeting": "<one sentence>"
  }},
  "overall_verdict": "<PASS|FAIL>"
}}

Set overall_verdict to FAIL if any HIGH severity violation is detected, otherwise PASS."""

# Step 4 — Maximum image dimension before sending to the model.
# Resizing keeps memory usage low and speeds up the API call.
MAX_PX = 768


def _load_image(source: str) -> Image.Image:
    """Load an image from a path or URL, resize to MAX_PX on the long edge.

    Unlike the local evaluator (which returns JPEG bytes), Gemini's SDK
    accepts a PIL Image object directly, so we return the Image here.

    Steps:
        a. Fetch bytes from URL (with content-type check) or open a local file.
        b. Reject PDFs early with a clear error — PIL cannot open them.
        c. Resize to fit within MAX_PX × MAX_PX while preserving aspect ratio.
        d. Convert to RGB to strip alpha channels (e.g. PNGs with transparency).
    """
    # Step a — Load from URL or local file
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(
                f"URL did not return an image (Content-Type: {content_type!r}). "
                "Right-click the image in your browser and choose 'Copy image address' "
                "to get the direct image URL."
            )
        img = Image.open(io.BytesIO(response.content))
    else:
        path = Path(source)
        # Step b — Reject PDFs; PIL would raise an unhelpful error otherwise
        if path.suffix.lower() == ".pdf":
            raise ValueError("PDF files are not supported. Please provide a JPG, PNG, GIF, or WEBP image.")
        img = Image.open(path)

    # Steps c & d — Resize and normalize color mode
    img.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
    return img.convert("RGB")


def evaluate(source: str, api_key: str | None = None) -> dict:
    """Evaluate an ad image using Gemini. Returns violations, UX score, audience suitability, and verdict.

    Steps:
        1. Resolve the API key (argument → env var → error).
        2. Configure the Gemini SDK and instantiate the model.
        3. Load and resize the image.
        4. Send [prompt, image] to the model and get the raw text response.
        5. Strip markdown code fences that the model sometimes wraps around JSON.
        6. Extract the JSON object with a regex and parse it.
        7. Return the parsed dict, or an error dict if parsing fails.
    """
    # Step 1 — Resolve API key
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Get a free key at aistudio.google.com "
            "and add it to your .env file."
        )

    # Step 2 — Configure the SDK and create the model instance
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL)

    # Step 3 — Load and resize the image
    image = _load_image(source)

    # Step 4 — Call the model.
    # Gemini's multimodal API accepts a list where each element is either
    # a text string or a PIL Image. The model sees both together.
    response = model.generate_content([PROMPT, image])

    # Step 5 — Strip markdown fences (```json ... ```) that Gemini sometimes adds
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    # Step 6 — Extract the first {...} block and parse it as JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _error_result(f"Model returned unparseable response: {raw[:200]}")

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as e:
        return _error_result(f"Model returned invalid JSON ({e}): {match.group()[:200]}")

    # Step 7 — Return the structured result
    return result


def _error_result(reason: str) -> dict:
    """Return a safe error dict that callers and display_results() can handle gracefully."""
    return {
        "violations": [],
        "ux_score": None,
        "ux_notes": reason,
        "audience": {},
        "overall_verdict": "ERROR",
    }
