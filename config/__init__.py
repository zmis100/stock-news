"""환경변수 / 시크릿 로딩"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key, "")


NAVER_CLIENT_ID = get_secret("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = get_secret("NAVER_CLIENT_SECRET")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


def api_status() -> tuple[bool, list[str]]:
    """API 키 유효성 체크. (전체OK, 누락키목록) 반환."""
    missing = []
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        missing.append("NAVER")
    if not GEMINI_API_KEY:
        missing.append("GEMINI")
    return (len(missing) == 0, missing)
