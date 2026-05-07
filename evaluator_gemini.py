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
    3. Make two model calls — one for violations, one for UX and audience
    4. Strip markdown fences, extract and parse JSON from each response
    5. Merge results, compute verdict from HIGH_RISK violations
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

from policies import VIOLATIONS, HIGH_RISK

# gemini-2.0-flash is the recommended free-tier model: fast, multimodal, handles images well.
MODEL = "gemini-2.0-flash"

# Build the violation list from policies.py so the prompt stays in sync
# with any policy changes automatically.
VIOLATION_LIST = "\n".join(
    f'- {v["id"]}: {v["description"]}' for v in VIOLATIONS
)

# --- Prompt 1: Policy violations ---
# Focused solely on detecting violations. Keeping this separate from the UX
# prompt gives the model a narrower task and produces more accurate results.
# Each violation returns a confidence score (0.0–1.0) so callers can filter
# by certainty rather than treating all detections equally.
PROMPT_VIOLATIONS = f"""You are a Search Ads policy compliance expert.

Analyze the provided ad image for policy violations.

Check for each of the following violations:
{VIOLATION_LIST}

For each violation return:
- detected: true if the violation is present, false otherwise
- confidence: a float from 0.0 to 1.0 representing how certain you are
- evidence: one sentence explaining what you saw

Reply with JSON only — no other text:
{{
  "violations": [
    {{
      "id": "<violation id>",
      "detected": <true|false>,
      "confidence": <float 0.0-1.0>,
      "evidence": "<one sentence>"
    }}
  ]
}}"""

# --- Prompt 2: UX and audience ---
# Separated from violations so the model focuses purely on quality and
# audience fit, without being distracted by compliance checking.
PROMPT_UX = """You are a UX evaluator for Search Ad images.

Analyze the provided ad image across two areas.

PART 1 — UX SCORE
Rate the overall user experience on a scale of 1–10, considering:
- Visual clarity and composition
- Message legibility
- Use of space
- Overall ad effectiveness

PART 2 — AUDIENCE SUITABILITY
Classify whether the image is appropriate for each age group:
- under_13
- 13_to_17
- 18_plus
- all_ages

Also suggest the most appropriate target audience based on the visual content and tone.

Reply with JSON only — no other text:
{
  "ux_score": <float 1.0-10.0>,
  "ux_notes": "<two to three sentences on strengths and weaknesses>",
  "audience": {
    "under_13": <true|false>,
    "13_to_17": <true|false>,
    "18_plus": <true|false>,
    "all_ages": <true|false>,
    "recommended_targeting": "<one sentence>"
  }
}"""

# Images larger than this are resized before sending to the model.
# 768px is enough for the model to read text and assess composition,
# and keeps API payload size low.
MAX_PX = 768


def _load_image(source: str) -> Image.Image:
    """Load an image from a path or URL, resize to MAX_PX on the long edge.

    Unlike the local evaluator (which returns JPEG bytes), Gemini's SDK
    accepts a PIL Image object directly, so we return the Image here.

    Steps:
        a. Fetch bytes from URL (with content-type check) or open a local file.
        b. Reject PDFs early — PIL cannot open them.
        c. Resize to fit within MAX_PX × MAX_PX while preserving aspect ratio.
        d. Convert to RGB to strip alpha channels (e.g. PNGs with transparency).
    """
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
        if path.suffix.lower() == ".pdf":
            raise ValueError("PDF files are not supported. Please provide a JPG, PNG, GIF, or WEBP image.")
        img = Image.open(path)

    img.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
    return img.convert("RGB")


def _call(model, prompt: str, image: Image.Image):
    """Send one prompt + image to Gemini and return parsed JSON, or None on failure.

    Gemini's API accepts a list where each element is either a text string
    or a PIL Image — both are passed together in a single generate_content call.
    """
    response = model.generate_content([prompt, image])

    # Strip markdown fences (```json ... ```) that Gemini sometimes adds.
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", response.text.strip())

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def evaluate(source: str, api_key: str | None = None) -> dict:
    """Evaluate an ad image using Gemini. Returns violations, UX score, audience suitability, and verdict.

    Makes two sequential model calls (violations, then UX/audience), merges
    the results, and computes the overall verdict based on HIGH severity hits.

    Steps:
        1. Resolve the API key (argument → env var → error).
        2. Configure the Gemini SDK and instantiate the model.
        3. Load and resize the image.
        4. Call 1 — violations prompt → violations list with confidence scores.
        5. Call 2 — UX prompt → ux_score, ux_notes, audience flags.
        6. Merge results and compute verdict from policies.HIGH_RISK.
        7. Return the merged dict, or an error dict if either call fails.
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

    # Step 4 — Call 1: violations
    violations_data = _call(model, PROMPT_VIOLATIONS, image)
    if violations_data is None:
        return _error_result("Violations call returned unparsable response.")

    # Step 5 — Call 2: UX and audience
    ux_data = _call(model, PROMPT_UX, image)
    if ux_data is None:
        return _error_result("UX call returned unparsable response.")

    # Step 6 — Compute verdict from HIGH_RISK set rather than trusting the model
    violations = violations_data.get("violations", [])
    detected_ids = {v["id"] for v in violations if v.get("detected")}
    verdict = "FAIL" if detected_ids & HIGH_RISK else "PASS"

    # Step 7 — Return merged result
    return {
        "violations": violations,
        "ux_score": ux_data.get("ux_score"),
        "ux_notes": ux_data.get("ux_notes", ""),
        "audience": ux_data.get("audience", {}),
        "overall_verdict": verdict,
    }


def _error_result(reason: str) -> dict:
    """Return a safe fallback dict when a model call fails."""
    return {
        "violations": [],
        "ux_score": None,
        "ux_notes": reason,
        "audience": {},
        "overall_verdict": "ERROR",
    }
