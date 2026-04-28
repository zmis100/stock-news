"""관심 종목 (즐겨찾기) JSON 저장소.

단일 사용자/로컬 환경 가정. 동시성 이슈 없음.
Streamlit Cloud 배포 시 ephemeral filesystem이라 휘발됨 → 추후 sqlite/cloud로 이전 가능.
"""
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FAV_PATH = _DATA_DIR / "favorites.json"
_MAX_FAVORITES = 30


def _ensure_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def load() -> list[str]:
    """즐겨찾기 목록 로드. 없으면 빈 리스트."""
    if not _FAV_PATH.exists():
        return []
    try:
        data = json.loads(_FAV_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data if x]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save(items: list[str]):
    _ensure_dir()
    _FAV_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add(name: str) -> tuple[bool, str]:
    """즐겨찾기 추가. (성공여부, 메시지) 반환."""
    name = (name or "").strip()
    if not name:
        return False, "종목명이 비어있습니다."

    items = load()
    if name in items:
        return False, f"'{name}'은(는) 이미 즐겨찾기에 있습니다."
    if len(items) >= _MAX_FAVORITES:
        return False, f"즐겨찾기는 최대 {_MAX_FAVORITES}개까지 저장 가능합니다."

    items.append(name)
    _save(items)
    return True, f"⭐ '{name}' 추가됨"


def remove(name: str) -> tuple[bool, str]:
    items = load()
    if name not in items:
        return False, f"'{name}'은(는) 즐겨찾기에 없습니다."
    items.remove(name)
    _save(items)
    return True, f"'{name}' 제거됨"


def is_favorited(name: str) -> bool:
    return name in load()


def toggle(name: str) -> tuple[bool, str]:
    """추가/제거 토글. (현재 등록상태, 메시지) 반환."""
    if is_favorited(name):
        ok, msg = remove(name)
        return False, msg
    ok, msg = add(name)
    return ok, msg
