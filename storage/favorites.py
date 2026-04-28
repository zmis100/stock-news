"""관심 종목 (즐겨찾기) — 브라우저 LocalStorage 기반.

각 사용자(브라우저)별로 분리 저장.
Streamlit Cloud 배포 시 모든 방문자가 자기 데이터만 보게 됨.
"""
import json
import time

from storage._browser import get_ls

_KEY = "stock_news_favorites"
_MAX_FAVORITES = 30


def _read() -> list[str]:
    raw = get_ls().getItem(_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data if x][:_MAX_FAVORITES]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _write(items: list[str]):
    payload = json.dumps(items, ensure_ascii=False)
    get_ls().setItem(_KEY, payload, key=f"set_fav_{time.time_ns()}")


def load() -> list[str]:
    return _read()


def add(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "종목명이 비어있습니다."

    items = _read()
    if name in items:
        return False, f"'{name}'은(는) 이미 즐겨찾기에 있습니다."
    if len(items) >= _MAX_FAVORITES:
        return False, f"즐겨찾기는 최대 {_MAX_FAVORITES}개까지 저장 가능합니다."

    items.append(name)
    _write(items)
    return True, f"⭐ '{name}' 추가됨"


def remove(name: str) -> tuple[bool, str]:
    items = _read()
    if name not in items:
        return False, f"'{name}'은(는) 즐겨찾기에 없습니다."
    items.remove(name)
    _write(items)
    return True, f"'{name}' 제거됨"


def is_favorited(name: str) -> bool:
    return name in _read()


def toggle(name: str) -> tuple[bool, str]:
    """추가/제거 토글. (등록상태, 메시지) 반환."""
    if is_favorited(name):
        ok, msg = remove(name)
        return False, msg
    return add(name)
