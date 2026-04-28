"""검색 히스토리 (LRU 큐). 최근 분석한 종목명 N개 보관."""
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PATH = _DATA_DIR / "history.json"
_MAX = 10


def load() -> list[str]:
    if not _PATH.exists():
        return []
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data if x][:_MAX]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save(items: list[str]):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def push(name: str) -> None:
    """가장 최근 분석한 종목을 맨 위로 이동 (중복 제거)."""
    name = (name or "").strip()
    if not name:
        return
    items = load()
    if name in items:
        items.remove(name)
    items.insert(0, name)
    _save(items[:_MAX])


def remove(name: str) -> None:
    items = load()
    if name in items:
        items.remove(name)
        _save(items)


def clear() -> None:
    _save([])
