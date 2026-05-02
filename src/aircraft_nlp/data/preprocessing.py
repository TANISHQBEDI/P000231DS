
import json
import re
from importlib import resources

def _load_abbreviations() -> dict[str, str]:
    with resources.files("aircraft_nlp.config").joinpath("abbreviations.json").open("r", encoding="utf-8") as f:
        return json.load(f)

_ABBREVIATIONS = _load_abbreviations()
_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, _ABBREVIATIONS)) + r")\b") if _ABBREVIATIONS else None

def normalize_text(text: str) -> str:
    text = text.lower()
    if _PATTERN:
        text = _PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)
    return text