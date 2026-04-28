"""검색 히스토리 (LRU) — 브라우저 LocalStorage 기반."""
import json
import time

from storage._browser import get_ls

_KEY = "stock_news_history"
_MAX = 10


def _read() -> list[str]:
    raw = get_ls().getItem(_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data if x][:_MAX]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _write(items: list[str]):
    payload = json.dumps(items, ensure_ascii=False)
    get_ls().setItem(_KEY, payload, key=f"set_hist_{time.time_ns()}")


def load() -> list[str]:
    return _read()


def push(name: str) -> None:
    """가장 최근 분석한 종목을 맨 위로 (중복 제거)."""
    name = (name or "").strip()
    if not name:
        return
    items = _read()
    if name in items:
        items.remove(name)
    items.insert(0, name)
    _write(items[:_MAX])


def remove(name: str) -> None:
    items = _read()
    if name in items:
        items.remove(name)
        _write(items)


def clear() -> None:
    _write([])
