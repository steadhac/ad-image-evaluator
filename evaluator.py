"""
Ad image vision analysis via Ollama.
Sends an ad image to a local vision model and returns structured violation, UX, and audience data.

Two separate model calls are made per image:
  Call 1 — policy violations + confidence scores
  Call 2 — UX score + audience suitability
Results are merged into a single dict before returning.
"""
import io
import json
import os
import re
from pathlib import Path

import ollama
import requests
from PIL import Image

from policies import VIOLATIONS, HIGH_RISK

# Build the violation list dynamically from policies.py so the prompt
# stays in sync automatically when violations are added or changed.
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
# and keeps memory usage low on CPU-only machines.
MAX_PX = 768


def _load_image(source: str) -> bytes:
    """Load an image from a file path or direct URL and return resized JPEG bytes.

    Steps:
        1. Fetch from URL (with content-type validation) or open from disk.
        2. Reject PDFs early — PIL cannot open them.
        3. Resize to fit within MAX_PX × MAX_PX, preserving aspect ratio.
        4. Convert to RGB (strips alpha channels from PNGs) and encode as JPEG.
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

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _call(model: str, prompt: str, image: bytes) -> dict | None:
    """Send one prompt + image to the model and return the parsed JSON response.

    Returns None if the model response cannot be parsed as JSON, so the
    caller can handle the failure without raising an exception.
    """
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image],
        }],
        options={"num_predict": 2048},
    )

    # Strip markdown code fences (```json ... ```) that the model sometimes adds.
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", response.message.content.strip())

    # Extract the first {...} block from the response and parse it.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def evaluate(source: str) -> dict:
    """Evaluate an ad image for policy violations, UX quality, and audience suitability.

    Makes two sequential model calls (violations, then UX/audience), merges
    the results, and computes the overall verdict based on HIGH severity hits.

    Returns a dict with keys:
        violations      — list of violation results with detected, confidence, evidence
        ux_score        — float 1.0–10.0 or None on error
        ux_notes        — qualitative summary of strengths and weaknesses
        audience        — suitability flags per age group
        overall_verdict — PASS, FAIL, or ERROR
    """
    model = os.environ.get("OLLAMA_MODEL", "llava")
    image = _load_image(source)

    # Call 1 — violations
    violations_data = _call(model, PROMPT_VIOLATIONS, image)
    if violations_data is None:
        return _error_result("Violations call returned unparseable response.")

    # Call 2 — UX and audience
    ux_data = _call(model, PROMPT_UX, image)
    if ux_data is None:
        return _error_result("UX call returned unparseable response.")

    # Verdict is computed here rather than by the model, so it stays consistent
    # with the HIGH_RISK set defined in policies.py.
    violations = violations_data.get("violations", [])
    detected_ids = {v["id"] for v in violations if v.get("detected")}
    verdict = "FAIL" if detected_ids & HIGH_RISK else "PASS"

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
