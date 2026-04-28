"""텍스트 정리 유틸"""
import re
import html


def clean_html(text: str) -> str:
    """HTML 태그 제거 + 엔티티 디코딩"""
    text = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(text)


def safe_int(s, default: int = 0) -> int:
    try:
        return int(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return default


def safe_float(s, default: float = 0.0) -> float:
    try:
        return float(str(s).replace(",", "").replace("%", "").replace("+", ""))
    except (ValueError, TypeError):
        return default
