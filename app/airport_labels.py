"""Display-only labels; never translate or replace model airport codes."""
import json
from pathlib import Path


def load_airport_names(path: Path) -> dict[str, str]:
    try:
        entries = json.loads(path.read_text())['airports']
        return {code: entry['name'] for code, entry in entries.items()
                if isinstance(entry.get('name'), str) and entry['name'].strip()}
    except (OSError, ValueError, KeyError):
        # Optional presentation data must not prevent a prediction.
        return {}


def airport_label(code: str, names: dict[str, str]) -> str:
    name = names.get(code)
    return f'{code} — {name}' if name else code
