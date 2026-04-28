"""애플리케이션 로깅 일원 설정.

app.py 진입점에서 setup_logging()을 한 번만 호출.
환경변수 LOG_LEVEL로 레벨 조정 (기본 INFO).
"""
import logging
import os
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
_INITIALIZED = False


def setup_logging(log_to_file: bool = True) -> None:
    """루트 로거 설정. 콘솔 + (옵션) 파일 핸들러."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_to_file:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                _LOG_DIR / "app.log",
                encoding="utf-8",
            )
            handlers.append(file_handler)
        except OSError:
            pass

    logging.basicConfig(level=level, format=fmt, datefmt=date_fmt, handlers=handlers)

    # 외부 라이브러리 노이즈 줄이기
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)

    _INITIALIZED = True
