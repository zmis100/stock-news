"""관심종목 알림 설정 저장소.

구조: {
    "channel": "discord" | "email" | "off",
    "rules": [
        {"name": "삼성전자", "threshold_pct": 5.0, "last_alerted_at": "...", "last_close": 0},
        ...
    ]
}
"""
import json
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PATH = _DATA_DIR / "alerts.json"


def _default() -> dict:
    return {"channel": "off", "rules": []}


def load() -> dict:
    if not _PATH.exists():
        return _default()
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default()
        data.setdefault("channel", "off")
        data.setdefault("rules", [])
        return data
    except (json.JSONDecodeError, OSError):
        return _default()


def save(data: dict):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_channel(channel: str):
    data = load()
    data["channel"] = channel
    save(data)


def add_rule(name: str, threshold_pct: float = 5.0):
    name = (name or "").strip()
    if not name:
        return
    data = load()
    if any(r["name"] == name for r in data["rules"]):
        return
    data["rules"].append({
        "name": name,
        "threshold_pct": float(threshold_pct),
        "last_alerted_at": None,
        "last_close": 0,
    })
    save(data)


def remove_rule(name: str):
    data = load()
    data["rules"] = [r for r in data["rules"] if r["name"] != name]
    save(data)


def update_rule(name: str, **fields):
    data = load()
    for r in data["rules"]:
        if r["name"] == name:
            r.update(fields)
            break
    save(data)


def mark_alerted(name: str, close_price: int):
    update_rule(
        name,
        last_alerted_at=datetime.now().isoformat(),
        last_close=int(close_price),
    )
