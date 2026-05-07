"""
Ad image policy violation catalog.

Each violation maps to a real Search Ads policy constraint.
Severity: HIGH violations cause immediate disapproval; MEDIUM are flagged for review.
"""

VIOLATIONS = [
    {
        "id":          "visual_quality",
        "name":        "Poor Visual Quality",
        "severity":    "MEDIUM",
        "description": "Image is blurry, pixelated, heavily compressed, or too low resolution to be legible.",
    },
    {
        "id":          "blank_area",
        "name":        "Excessive Blank Area",
        "severity":    "MEDIUM",
        "description": "Image contains large empty or white regions that waste ad real estate and reduce effectiveness.",
    },
    {
        "id":          "incorrect_orientation",
        "name":        "Incorrect Orientation",
        "severity":    "HIGH",
        "description": "Image orientation or aspect ratio does not match the intended ad placement (e.g. portrait image in a landscape slot).",
    },
    {
        "id":          "profanity",
        "name":        "Profanity or Offensive Language",
        "severity":    "HIGH",
        "description": "Image contains profane, offensive, or vulgar text or symbols.",
    },
    {
        "id":          "clickbait",
        "name":        "Clickbait or Misleading Content",
        "severity":    "HIGH",
        "description": "Image uses sensationalist headlines, exaggerated claims, or misleading visuals designed to trick users into clicking.",
    },
    {
        "id":          "weapons",
        "name":        "Weapons",
        "severity":    "HIGH",
        "description": "Image depicts firearms, knives, or other weapons in a promotional context.",
    },
    {
        "id":          "cta_button",
        "name":        "CTA Text on Button",
        "severity":    "HIGH",
        "description": (
            "Image contains a button-style element with call-to-action text (e.g. 'Shop Now', 'Click Here', 'Buy Now'). "
            "Simulating interactive UI controls inside a static image is a Search Ads policy violation."
        ),
    },
    {
        "id":          "interactive_elements",
        "name":        "Simulated Interactive Elements",
        "severity":    "HIGH",
        "description": (
            "Image contains fake interactive UI elements such as search bars, prefilled search boxes, "
            "dropdown menus, or form fields. These simulate native platform UI and are prohibited."
        ),
    },
]

HIGH_RISK = {v["id"] for v in VIOLATIONS if v["severity"] == "HIGH"}
