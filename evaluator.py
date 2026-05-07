"""
Ad image vision analysis via Ollama.
Sends an ad image to a local vision model and returns structured violation, UX, and audience data.
"""
import io
import json
import os
import re
from pathlib import Path

import ollama
import requests
from PIL import Image

from policies import VIOLATIONS

VIOLATION_LIST = "\n".join(
    f'- {v["id"]}: {v["description"]}' for v in VIOLATIONS
)

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


MAX_PX = 768


def _load_image(source: str) -> bytes:
    """Load an image from a path or URL, resize to MAX_PX on the long edge, return JPEG bytes."""
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


def evaluate(source: str) -> dict:
    """Evaluate an ad image. Returns violations, UX score, audience suitability, and verdict."""
    model = os.environ.get("OLLAMA_MODEL", "llava")
    image = _load_image(source)

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image],
        }],
        options={"num_predict": 2048},
    )

    raw = re.sub(r"```(?:json)?\s*|\s*```", "", response.message.content.strip())

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _error_result(f"Model returned unparseable response: {raw[:200]}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        return _error_result(f"Model returned invalid JSON ({e}): {match.group()[:200]}")


def _error_result(reason: str) -> dict:
    return {
        "violations": [],
        "ux_score": None,
        "ux_notes": reason,
        "audience": {},
        "overall_verdict": "ERROR",
    }
