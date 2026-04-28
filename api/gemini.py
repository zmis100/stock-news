"""Gemini API 폴백 체인 + 외부 프롬프트 로더."""
import json
import logging
import re
from pathlib import Path

import streamlit as st
from google import genai

from config import GEMINI_API_KEY
from utils.kst import now, today_str

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


GEMINI_MODELS = [
    {"id": "models/gemini-2.5-pro",        "label": "Gemini 2.5 Pro 🏆"},
    {"id": "models/gemini-2.5-flash",      "label": "Gemini 2.5 Flash ⚡"},
    {"id": "models/gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite 💨"},
    {"id": "models/gemini-2.0-flash",      "label": "Gemini 2.0 Flash 🆓"},
]

_FALLBACK_KEYWORDS = (
    "429", "quota", "resource_exhausted", "rate limit",
    "exceeded", "unavailable", "503",
)


def _load_prompt(name: str) -> str:
    """prompts/{name}.md 로드 (없으면 KeyError 가까운 RuntimeError)"""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise RuntimeError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def call_with_fallback(prompt: str) -> tuple[str, str]:
    """Gemini 모델 폴백 체인 호출. (응답텍스트, 사용모델라벨) 반환."""
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY가 설정되지 않았습니다.", "N/A"

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        log.exception("Gemini client init failed")
        return f"❌ Gemini 클라이언트 초기화 실패: {e}", "FAILED"

    error_log = []
    for m in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(model=m["id"], contents=prompt)
            if resp and resp.text:
                log.info("Gemini call ok via %s", m["label"])
                return resp.text, m["label"]
            error_log.append(f"• {m['label']}: 빈 응답")
        except Exception as e:
            err_str = str(e).lower()
            error_log.append(f"• {m['label']}: {str(e)[:120]}")
            log.warning("Gemini %s failed: %s", m["label"], e)
            if any(k in err_str for k in _FALLBACK_KEYWORDS):
                continue
            continue

    fail_msg = (
        "❌ 모든 Gemini 모델 호출 실패\n\n"
        "**시도 내역:**\n" + "\n".join(error_log) + "\n\n"
        "잠시 후 다시 시도하거나 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키를 확인하세요."
    )
    return fail_msg, "FAILED"


# ── 프롬프트 빌더 ─────────────────────────────────────────────

def _format_news_block(news_titles_and_descs: tuple) -> str:
    return "\n\n".join([
        f"[기사 {i+1}]\n제목: {title}\n내용: {desc}"
        for i, (title, desc) in enumerate(news_titles_and_descs)
    ])


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_news(label: str, news_titles_and_descs: tuple, mode: str = "stock") -> tuple[str, str]:
    """뉴스 요약. mode='stock' 또는 'today'. 1시간 캐시."""
    if not news_titles_and_descs:
        return "요약할 뉴스 기사가 없습니다.", "N/A"

    news_text = _format_news_block(news_titles_and_descs)

    if mode == "today":
        template = _load_prompt("today_market")
        prompt = template.format(today=today_str(), news_text=news_text)
    else:
        template = _load_prompt("stock_summary")
        prompt = template.format(label=label, news_text=news_text)

    return call_with_fallback(prompt)


@st.cache_data(ttl=600, show_spinner=False)
def analyze_closing_bet(
    stocks_tuple: tuple,
    news_tuple: tuple,
    session_label: str = "장 마감 임박",
) -> tuple[str, str]:
    """종가베팅 분석. 10분 캐시."""
    if not stocks_tuple:
        return "분석할 특징주가 없습니다.", "N/A"

    stocks_summary = "\n".join([
        f"- {name} ({code}): +{ratio:.2f}%, 거래대금 {amt:,.0f}억"
        for name, code, ratio, amt in stocks_tuple
    ])

    news_block = ""
    for name, items in news_tuple:
        if not items:
            continue
        news_block += f"\n■ {name}\n"
        for title, desc in items:
            news_block += f"  · {title}\n    {desc[:120]}\n"

    template = _load_prompt("closing_bet")
    prompt = template.format(
        now=now().strftime("%H:%M"),
        session_label=session_label,
        stocks_summary=stocks_summary,
        news_block=news_block,
    )
    return call_with_fallback(prompt)


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_movers_reasons(stocks_tuple: tuple, news_tuple: tuple) -> tuple[dict[str, str], str]:
    """거래대금 + 등락률 상위 종목들의 상승/하락 사유 분석.

    Returns: ({code: reason}, used_model). 30분 캐시.
    """
    if not stocks_tuple:
        return {}, "N/A"

    stocks_summary = "\n".join([
        f"- [{code}] {name}: {ratio:+.2f}%, 거래대금 {amt:,.0f}억"
        for name, code, ratio, amt in stocks_tuple
    ])

    news_block = ""
    for name, items in news_tuple:
        if not items:
            continue
        news_block += f"\n■ {name}\n"
        for title, desc in items[:3]:
            news_block += f"  · {title}\n    {desc[:120]}\n"

    template = _load_prompt("movers_reasons")
    prompt = template.format(stocks_summary=stocks_summary, news_block=news_block)

    text, used_model = call_with_fallback(prompt)
    if used_model == "FAILED":
        return {}, used_model

    # JSON 추출
    code_to_reason: dict[str, str] = {}
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"\s*```", "", cleaned)
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for item in data:
                code = str(item.get("code", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if code and reason:
                    code_to_reason[code] = reason
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("analyze_movers_reasons JSON parse failed: %s", e)

    return code_to_reason, used_model
