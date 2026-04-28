"""yfinance 기반 시장 데이터 (세계증시 + 국내 종목 OHLC 차트)"""
import logging
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import yfinance as yf

from api.naver import fetch_trading_volume_top

log = logging.getLogger(__name__)


# ── 세계 증시 ────────────────────────────────────────────────

WORLD_INDICES = [
    {"name": "다우산업",   "ticker": "^DJI"},
    {"name": "나스닥종합", "ticker": "^IXIC"},
    {"name": "S&P 500",    "ticker": "^GSPC"},
    {"name": "코스피",     "ticker": "^KS11"},
    {"name": "코스닥",     "ticker": "^KQ11"},
    {"name": "닛케이225",  "ticker": "^N225"},
    {"name": "상해종합",   "ticker": "000001.SS"},
    {"name": "항셍",       "ticker": "^HSI"},
]


def _fetch_index(item: dict):
    try:
        t = yf.Ticker(item["ticker"])
        daily = t.history(period="5d", interval="1d")
        if len(daily) < 2:
            return None
        prev_close = float(daily["Close"].iloc[-2])

        intraday = t.history(period="1d", interval="5m")
        if not intraday.empty:
            current = float(intraday["Close"].iloc[-1])
            chart_prices = [float(p) for p in intraday["Close"].tolist()]
        else:
            current = float(daily["Close"].iloc[-1])
            chart_prices = [float(p) for p in daily["Close"].tolist()]

        change = current - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0

        return {
            "name":         item["name"],
            "ticker":       item["ticker"],
            "current":      current,
            "prev_close":   prev_close,
            "change":       change,
            "change_pct":   change_pct,
            "chart_prices": chart_prices,
        }
    except Exception as e:
        log.warning("yfinance index %s failed: %s", item["ticker"], e)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_world_indices() -> list[dict]:
    """세계 증시 지수 (5분 캐시)"""
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_fetch_index, WORLD_INDICES))
    return [r for r in results if r is not None]


# ── 국내 종목 OHLC 차트 ──────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _build_ticker_index() -> dict[str, tuple[str, str]]:
    """종목명 → (code, market) 매핑. 거래대금 TOP100 기반 (10분 캐시)."""
    top, _ = fetch_trading_volume_top(100)
    return {s["name"]: (s["code"], s["market"]) for s in top}


def name_to_yf_ticker(name_or_code: str, market_hint: str | None = None) -> str | None:
    """
    종목명 또는 코드를 yfinance 티커로 변환.
    - "삼성전자" → "005930.KS"
    - "005930" + market_hint="KOSPI" → "005930.KS"
    """
    s = (name_or_code or "").strip()
    if not s:
        return None

    code = s
    market = market_hint

    if not s.isdigit():
        idx = _build_ticker_index()
        hit = idx.get(s)
        if not hit:
            return None
        code, market = hit
    else:
        if not market:
            return None

    suffix = ".KS" if market == "KOSPI" else ".KQ"
    return f"{code}{suffix}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker: str, period: str = "6mo", interval: str = "1d"):
    """OHLC 캔들 데이터 (5분 캐시). pandas DataFrame 반환.

    interval='1y' (년봉)은 yfinance가 미지원이라 월봉을 받아 연 단위로 리샘플링.
    """
    try:
        t = yf.Ticker(ticker)

        if interval == "1y":
            df = t.history(period="max", interval="1mo")
            if df.empty:
                return None
            df = df.resample("YE").agg({
                "Open":   "first",
                "High":   "max",
                "Low":    "min",
                "Close":  "last",
                "Volume": "sum",
            }).dropna()
            return df if not df.empty else None

        df = t.history(period=period, interval=interval)
        return df if not df.empty else None
    except Exception as e:
        log.warning("fetch_ohlc %s failed: %s", ticker, e)
        return None
