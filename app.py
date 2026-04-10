import streamlit as st
import requests
import os
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
import yfinance as yf
import plotly.graph_objects as go

# ── 환경변수 로딩 ──────────────────────────────────────────────
load_dotenv()

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key, "")

NAVER_CLIENT_ID     = get_secret("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = get_secret("NAVER_CLIENT_SECRET")
GEMINI_API_KEY      = get_secret("GEMINI_API_KEY")

GEMINI_MODEL = "models/gemini-2.5-flash-lite"  # (legacy - 폴백 체인으로 대체됨)

# ── Gemini 모델 폴백 체인 (1순위: 최고 품질 → 4순위: free tier 너그러움) ──
GEMINI_MODELS = [
    {"id": "models/gemini-2.5-pro",        "label": "Gemini 2.5 Pro 🏆"},
    {"id": "models/gemini-2.5-flash",      "label": "Gemini 2.5 Flash ⚡"},
    {"id": "models/gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite 💨"},
    {"id": "models/gemini-2.0-flash",      "label": "Gemini 2.0 Flash 🆓"},
]

# 폴백 트리거가 되는 에러 키워드 (대소문자 무시)
_GEMINI_FALLBACK_KEYWORDS = (
    "429", "quota", "resource_exhausted", "rate limit",
    "exceeded", "unavailable", "503",
)


def call_gemini_with_fallback(prompt: str) -> tuple[str, str]:
    """
    Gemini 모델 폴백 체인 호출.
    상위 모델부터 차례로 시도하다가 quota/rate-limit 에러 나면 다음 모델로 자동 폴백.

    Returns:
        (응답 텍스트, 사용된 모델 라벨)
    """
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY가 설정되지 않았습니다.", "N/A"

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"❌ Gemini 클라이언트 초기화 실패: {e}", "FAILED"

    error_log = []
    for m in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(model=m["id"], contents=prompt)
            if resp and resp.text:
                return resp.text, m["label"]
            error_log.append(f"• {m['label']}: 빈 응답")
        except Exception as e:
            err_str = str(e).lower()
            error_log.append(f"• {m['label']}: {str(e)[:120]}")
            if any(k in err_str for k in _GEMINI_FALLBACK_KEYWORDS):
                continue
            # 그 외 에러도 일단 다음 모델 시도 (네트워크 일시 장애 등)
            continue

    fail_msg = (
        "❌ 모든 Gemini 모델 호출 실패\n\n"
        "**시도 내역:**\n" + "\n".join(error_log) + "\n\n"
        "잠시 후 다시 시도하거나 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키를 확인하세요."
    )
    return fail_msg, "FAILED"

# ── 한국 시간 동적 계산 ────────────────────────────────────────
KST = timezone(timedelta(hours=9))

def get_kst_now():
    """항상 한국 시간 기준 현재 시각을 반환"""
    return datetime.now(KST)

def get_kst_today():
    """항상 한국 시간 기준 오늘 날짜를 반환"""
    return get_kst_now().date()

def get_kst_today_str():
    return get_kst_today().strftime("%Y년 %m월 %d일")


# ══════════════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════════════

def clean_html(text: str) -> str:
    """HTML 태그 및 엔티티 제거"""
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_naver_news(query: str, display: int = 10) -> list[dict]:
    """네이버 뉴스 검색 API 호출 (1시간 캐시)"""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id":     NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": "date"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        if response.status_code != 200:
            return []
        items = response.json().get("items", [])
        return [{
            "title":       clean_html(item.get("title", "")),
            "link":        item.get("link", ""),
            "description": clean_html(item.get("description", "")),
            "pubDate":     item.get("pubDate", ""),
        } for item in items]
    except requests.exceptions.Timeout:
        return []
    except Exception:
        return []


def fetch_multiple_keywords(keywords: list[str], display: int = 5) -> list[dict]:
    """여러 키워드를 병렬로 뉴스 수집"""
    all_news = []
    with ThreadPoolExecutor(max_workers=min(len(keywords), 5)) as executor:
        futures = {
            executor.submit(fetch_naver_news, f"{kw} 오늘", display): kw
            for kw in keywords
        }
        for future in as_completed(futures):
            try:
                news = future.result()
                all_news.extend(news)
            except Exception:
                pass

    # 중복 제거
    seen = set()
    unique = []
    for n in all_news:
        if n["title"] not in seen:
            seen.add(n["title"])
            unique.append(n)
    return unique


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trading_volume_top(limit: int = 100) -> tuple[list[dict], str]:
    """네이버 금융 API로 KOSPI+KOSDAQ 거래대금 상위 종목 조회 (1시간 캐시)"""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from requests.adapters import HTTPAdapter

        session = requests.Session()
        session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

        def fetch_page(market, page):
            url = f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize=100"
            r = session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return market, data.get("stocks", []), data.get("totalCount", 0)

        def parse_num(s):
            try:
                return int(str(s).replace(",", ""))
            except Exception:
                return 0

        # 1단계: 첫 페이지 2개 동시 요청으로 totalCount 파악
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(fetch_page, m, 1): m for m in ["KOSPI", "KOSDAQ"]}
            first = {}
            for f in as_completed(futures):
                market, stocks, total = f.result()
                first[market] = (stocks, total)

        # 2단계: 나머지 전 페이지 병렬 요청
        tasks = [
            (market, p)
            for market, (_, total) in first.items()
            for p in range(2, (total + 99) // 100 + 1)
        ]
        all_stocks = [
            (s, m) for m, (stocks, _) in first.items()
            for s in stocks if s.get("stockEndType") == "stock"
        ]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(fetch_page, m, p): (m, p) for m, p in tasks}
            for f in as_completed(futures):
                market, stocks, _ = f.result()
                all_stocks.extend([(s, market) for s in stocks if s.get("stockEndType") == "stock"])

        all_stocks.sort(key=lambda x: parse_num(x[0].get("accumulatedTradingValue", 0)), reverse=True)

        result = []
        for s, market in all_stocks[:limit]:
            amount_mil = parse_num(s.get("accumulatedTradingValue", 0))
            result.append({
                "rank": len(result) + 1,
                "name": s.get("stockName", ""),
                "code": s.get("itemCode", ""),
                "market": market,
                "close": parse_num(s.get("closePrice", 0)),
                "change_ratio": float(s.get("fluctuationsRatio", 0) or 0),
                "volume": parse_num(s.get("accumulatedTradingVolume", 0)),
                "amount": amount_mil * 1_000_000,
                "marcap": parse_num(s.get("marketValue", 0)) * 1_000_000,
            })
        return result, ""
    except Exception as e:
        return [], str(e)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_world_indices() -> list[dict]:
    """주요 세계 증시 지수 조회 (5분 캐시) - yfinance 기반"""
    indices = [
        {"name": "다우산업",   "ticker": "^DJI"},
        {"name": "나스닥종합", "ticker": "^IXIC"},
        {"name": "S&P 500",    "ticker": "^GSPC"},
        {"name": "코스피",     "ticker": "^KS11"},
        {"name": "코스닥",     "ticker": "^KQ11"},
        {"name": "닛케이225",  "ticker": "^N225"},
        {"name": "상해종합",   "ticker": "000001.SS"},
        {"name": "항셍",       "ticker": "^HSI"},
    ]

    def fetch_one(item):
        try:
            t = yf.Ticker(item["ticker"])
            # 일봉 5일치로 전일 종가 확보
            daily = t.history(period="5d", interval="1d")
            if len(daily) < 2:
                return None
            prev_close = float(daily["Close"].iloc[-2])

            # 분봉 1일치로 인트라데이 차트 확보
            intraday = t.history(period="1d", interval="5m")
            if not intraday.empty:
                current = float(intraday["Close"].iloc[-1])
                chart_prices = [float(p) for p in intraday["Close"].tolist()]
            else:
                # 분봉이 없으면 일봉으로 폴백
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
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_one, indices))

    return [r for r in results if r is not None]


# ══════════════════════════════════════════════════════════════
# 종가베팅용 데이터 함수들
# ══════════════════════════════════════════════════════════════

# 한국 주요 ETF 브랜드 prefix (이 prefix로 시작하는 종목명은 ETF로 판단)
_ETF_BRAND_PREFIXES = (
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "KOSEF", "HANARO",
    "ACE", "KINDEX", "SOL", "HK", "KCGI", "RISE", "WOORI",
    "BNK", "KIWOOM", "TIMEFOLIO", "PLUS",
)


def is_etf_or_spac(name: str) -> bool:
    """ETF / ETN / 스팩(SPAC) 여부 판별 (종목명 기반)"""
    if not name:
        return False
    upper = name.upper().strip()
    if any(upper.startswith(p) for p in _ETF_BRAND_PREFIXES):
        return True
    if "스팩" in name or "SPAC" in upper:
        return True
    if "ETN" in upper:
        return True
    return False


@st.cache_data(ttl=120, show_spinner=False)
def fetch_realtime_top_gainers(limit: int = 30) -> list[dict]:
    """
    네이버 금융 HTML 스크래핑 - 실시간 등락률 상위 (KOSPI + KOSDAQ)
    장중에도 실시간으로 갱신됨. 2분 캐시.
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    all_stocks = []
    for sosok, market in [("0", "KOSPI"), ("1", "KOSDAQ")]:
        try:
            url = f"https://finance.naver.com/sise/sise_rise.naver?sosok={sosok}"
            r = requests.get(url, headers=headers, timeout=8)
            r.encoding = "euc-kr"
            soup = BeautifulSoup(r.text, "html.parser")

            for row in soup.select("table.type_2 tr"):
                cols = row.select("td")
                if len(cols) < 10:
                    continue

                a_tag = cols[1].select_one("a.tltle")
                if not a_tag:
                    continue

                try:
                    name   = a_tag.text.strip()
                    href   = a_tag.get("href", "")
                    code   = href.split("code=")[-1] if "code=" in href else ""
                    price  = int(cols[2].text.strip().replace(",", ""))
                    chg    = float(cols[4].text.strip().replace("%", "").replace("+", ""))
                    volume = int(cols[5].text.strip().replace(",", ""))
                    amt_m  = int(cols[6].text.strip().replace(",", ""))  # 백만원 단위
                except (ValueError, IndexError):
                    continue

                all_stocks.append({
                    "name":         name,
                    "code":         code,
                    "market":       market,
                    "current":      price,
                    "change_ratio": chg,
                    "volume":       volume,
                    "amount":       amt_m * 1_000_000,
                })
        except Exception:
            continue

    all_stocks.sort(key=lambda x: x["change_ratio"], reverse=True)
    for i, s in enumerate(all_stocks[:limit], 1):
        s["display_rank"] = i
    return all_stocks[:limit]


def fetch_news_per_stock(stocks: list[dict], display: int = 3) -> dict[str, list[dict]]:
    """특징주 각각의 최신 뉴스를 병렬 수집"""
    result: dict[str, list[dict]] = {}
    if not stocks:
        return result
    with ThreadPoolExecutor(max_workers=min(len(stocks), 8)) as ex:
        futures = {
            ex.submit(fetch_naver_news, s["name"], display): s["name"]
            for s in stocks
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                result[name] = f.result()
            except Exception:
                result[name] = []
    return result


@st.cache_data(ttl=600, show_spinner=False)
def analyze_closing_bet(stocks_tuple: tuple, news_tuple: tuple) -> tuple[str, str]:
    """
    종가베팅 관점 LLM 분석 (10분 캐시).
    stocks_tuple: ((name, code, change_ratio, amount_eok), ...)
    news_tuple:   ((name, ((title, desc), ...)), ...)
    Returns: (분석텍스트, 사용모델라벨)
    """
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

    now_str = get_kst_now().strftime("%H:%M")

    prompt = f"""당신은 한국 주식 단타/종가베팅 전문가입니다.
현재 시각: {now_str} KST (장 마감 임박)

## 오늘의 거래대금 + 상승률 상위 특징주
{stocks_summary}

## 각 종목 최신 뉴스
{news_block}

위 데이터를 바탕으로 **종가베팅 관점**에서 분석해 주세요.

### 🔥 오늘의 주도 테마 TOP 3
- **1순위 테마**: 핵심 대장주(들)와 1줄 근거
- **2순위 테마**: 동일 형식
- **3순위 테마**: 동일 형식

### 📊 특징주 상승 사유
- 종목별로 핵심 모멘텀(뉴스/이슈) 1~2줄

### 🎯 종가베팅 코멘트
- 장 막판 주목할만한 종목 (모멘텀·수급 관점)
- 주의해야 할 리스크 (오버행, 단기과열, 재료 노출 등)

### ⚠️ 면책
본 분석은 정보 제공용이며 투자 권유가 아닙니다.

답변은 한국어 마크다운으로 작성해 주세요."""

    return call_gemini_with_fallback(prompt)


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_with_gemini(label: str, news_titles_and_descs: tuple, mode: str = "stock") -> tuple[str, str]:
    """Gemini API 호출 (1시간 캐시, 같은 뉴스면 재호출 안함). (요약, 사용모델) 튜플 반환."""
    if not news_titles_and_descs:
        return "요약할 뉴스 기사가 없습니다.", "N/A"

    news_text = "\n\n".join([
        f"[기사 {i+1}]\n제목: {title}\n내용: {desc}"
        for i, (title, desc) in enumerate(news_titles_and_descs)
    ])

    today_str = get_kst_today_str()

    if mode == "theme":
        prompt = f"""당신은 주식 시장 및 투자 테마 분석 전문가입니다.
아래는 '{label}' 테마와 관련된 최신 뉴스 기사들입니다.

{news_text}

위 뉴스들을 바탕으로 '{label}' 테마에 대한 **심층 분석**을 다음 형식으로 작성해 주세요:

### 📌 테마 개요
- '{label}' 테마의 현재 시장 위치와 중요성을 2~3줄로 설명

### 🏆 대장주 & 관련주 랭킹
아래 뉴스 기사들에서 **가장 많이 언급된 종목**을 기준으로 대장주와 관련주를 분류해 주세요.
- **🥇 대장주 1위**: 종목명 — 언급 빈도가 가장 높은 핵심 종목. 왜 대장주인지 근거(뉴스 언급 빈도, 시가총액, 테마 대표성)를 1줄로 설명
- **🥈 대장주 2위**: 종목명 — 근거 1줄 설명
- **🥉 대장주 3위**: 종목명 — 근거 1줄 설명
- **관련주**: 위 대장주 외에 뉴스에서 언급된 관련 종목들을 나열하고 각각 한줄 설명

### 📈 핵심 동향
- 최근 주요 뉴스에서 파악되는 핵심 트렌드 3~5가지를 불릿 포인트로 정리

### ⚡ 호재 vs 악재
- **호재**: 긍정적 요인 정리
- **악재**: 부정적 요인 및 리스크 정리

### 🔮 향후 전망
- 단기(1~2주) 및 중기(1~3개월) 전망을 간략히 서술

### ⚠️ 투자 유의사항
- 해당 테마 투자 시 주의할 점
- 본 분석은 정보 제공 목적이며 투자 권유가 아님을 명시

답변은 한국어로, 마크다운 형식으로 작성해 주세요."""
    elif mode == "today":
        prompt = f"""당신은 주식 시장 및 경제 분석 전문가입니다.
아래는 오늘({today_str}) 수집된 주요 뉴스 기사들입니다.

{news_text}

위 뉴스들을 바탕으로 **오늘의 주요 시장 동향**을 다음 조건에 맞게 요약해 주세요:
1. 오늘 시장에 영향을 미치는 **핵심 이슈 3가지**를 불릿 포인트로 정리
2. 전반적인 시장 분위기(강세/약세/혼조)를 한 줄로 평가
3. 투자자가 오늘 특히 주목해야 할 점을 마지막에 강조
4. 투자 권유가 아닌 정보 제공 목적임을 명시

답변은 한국어로, 마크다운 형식으로 작성해 주세요."""
    else:
        prompt = f"""당신은 주식 시장 분석 전문가입니다.
아래는 '{label}' 종목과 관련된 최신 뉴스 기사들입니다.

{news_text}

위 뉴스들을 바탕으로 '{label}' 종목의 **최근 시장 동향(Market Trend)**을 요약해 주세요:
1. 3~5줄로 명확하게 요약
2. 현재 시장 평가 서술
3. 주요 이슈나 모멘텀(상승/하락 요인) 분석
4. 전반적인 시장 분위기(호재/악재/중립) 마무리
5. 투자 권유가 아닌 정보 제공 목적임을 명시

답변은 한국어로, 마크다운 형식으로 작성해 주세요."""

    return call_gemini_with_fallback(prompt)


# ══════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════

def inject_custom_css():
    """프리미엄 UI 커스텀 CSS 주입"""
    st.markdown("""
    <style>
    /* ── 기본 폰트 & 배경 ── */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* ── 헤더 영역 ── */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .hero-header h1 {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        color: #fff;
        font-size: 0.75rem;
        margin-top: 0.8rem;
    }

    /* ── 탭 스타일 ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #fff !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── 카드 컨테이너 ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .summary-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(102,126,234,0.1);
    }
    .summary-card h3 {
        color: #a78bfa;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .news-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    .news-card:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(102,126,234,0.3);
        transform: translateY(-1px);
    }
    .news-card .news-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
        line-height: 1.4;
    }
    .news-card .news-desc {
        color: rgba(255,255,255,0.55);
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    .news-card .news-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
    }
    .news-card .news-date {
        color: rgba(255,255,255,0.35);
        font-size: 0.75rem;
    }
    .news-card a {
        color: #818cf8;
        font-size: 0.8rem;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    .news-card a:hover { color: #a78bfa; }
    .news-count-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .section-title {
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .keyword-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 0.5rem 0;
    }
    .keyword-chip {
        background: rgba(102,126,234,0.15);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ── 버튼 ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102,126,234,0.5) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }
    .stButton > button[disabled] {
        background: rgba(255,255,255,0.1) !important;
        box-shadow: none !important;
        color: rgba(255,255,255,0.3) !important;
    }

    /* ── 입력 필드 ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
    }

    /* ── 사이드바 ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.7); }

    /* ── 사이드바 카드 ── */
    .sidebar-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .sidebar-card h4 {
        color: #a78bfa;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-card p, .sidebar-card li {
        color: rgba(255,255,255,0.6);
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .sidebar-model {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-model span {
        color: #a78bfa;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sidebar-time {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        margin-bottom: 1rem;
    }
    .footer-notice {
        background: rgba(234,179,8,0.08);
        border: 1px solid rgba(234,179,8,0.15);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        color: rgba(234,179,8,0.7);
        font-size: 0.75rem;
        text-align: center;
        margin-top: 1rem;
    }

    /* ── 스피너 ── */
    .stSpinner > div { color: #a78bfa !important; }

    /* ── 경고/에러 박스 ── */
    .stAlert { border-radius: 12px !important; }

    /* ── Expander 숨기기 (뉴스를 커스텀 카드로 대체) ── */

    /* ── 모바일 반응형 ── */
    @media (max-width: 768px) {
        .hero-header { padding: 1.2rem 1rem; border-radius: 12px; }
        .hero-header h1 { font-size: 1.3rem; }
        .hero-header p { font-size: 0.8rem; }
        .glass-card, .summary-card { padding: 1rem; border-radius: 12px; }
        .news-card { padding: 0.8rem; }
        .news-card .news-title { font-size: 0.88rem; }
        .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 0.5rem 0.6rem; }
        .stButton > button { font-size: 0.88rem !important; padding: 0.5rem 1rem !important; }
        .section-title { font-size: 1rem; }
    }
    @media (max-width: 480px) {
        .hero-header h1 { font-size: 1.1rem; }
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
        .stTabs [data-baseweb="tab"] { font-size: 0.72rem; padding: 0.4rem 0.5rem; flex: 1; min-width: 0; text-align: center; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """상단 히어로 헤더"""
    kst_now = get_kst_now()
    time_str = kst_now.strftime("%H:%M KST")
    st.markdown(f"""
    <div class="hero-header">
        <h1>Stock News AI Analyzer</h1>
        <p>AI 기반 실시간 주식 뉴스 분석 & 시장 동향 리포트</p>
        <div class="hero-badge">Powered by Gemini 2.5 Flash Lite &middot; {time_str}</div>
    </div>
    """, unsafe_allow_html=True)


def render_summary(title: str, content: str, used_model: str = ""):
    """AI 요약 결과를 프리미엄 카드로 렌더링 (+ 사용된 모델 배지)"""
    st.markdown(f'<div class="section-title">&#129504; {title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-card">', unsafe_allow_html=True)
    st.markdown(content)
    if used_model and used_model not in ("N/A", "FAILED"):
        st.markdown(
            f'<div style="margin-top:1rem; padding-top:0.8rem; '
            f'border-top:1px solid rgba(255,255,255,0.08); '
            f'color:rgba(255,255,255,0.45); font-size:0.75rem; text-align:right;">'
            f'분석 모델: <strong style="color:#a78bfa;">{used_model}</strong></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_news_list(news_list: list[dict]):
    """뉴스 목록을 카드 형태로 렌더링"""
    st.markdown(
        f'<div class="section-title">&#128240; '
        f'수집된 뉴스 <span class="news-count-badge">{len(news_list)}건</span></div>',
        unsafe_allow_html=True,
    )
    for news in news_list:
        title = news["title"]
        desc = news["description"]
        link = news["link"]
        pub = news.get("pubDate", "")
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{title}</div>
            <div class="news-desc">{desc}</div>
            <div class="news-meta">
                <span class="news-date">{pub}</span>
                <a href="{link}" target="_blank">원문 보기 &#8594;</a>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_keyword_chips(keywords: list[str]):
    """키워드 칩 렌더링"""
    chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in keywords)
    st.markdown(f'<div class="keyword-chips">{chips}</div>', unsafe_allow_html=True)


def render_world_index_card(idx: dict):
    """네이버 스타일 세계 증시 카드 (그리드용 컴팩트 버전)"""
    name       = idx["name"]
    current    = idx["current"]
    change     = idx["change"]
    change_pct = idx["change_pct"]
    prices     = idx["chart_prices"]

    is_up = change >= 0
    color      = "#ef4444" if is_up else "#3b82f6"  # 한국식: 상승 빨강 / 하락 파랑
    fill_color = "rgba(239,68,68,0.18)" if is_up else "rgba(59,130,246,0.18)"
    arrow      = "▲" if is_up else "▼"
    sign       = "+" if is_up else ""

    # 컴팩트 카드 헤더
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;
                padding:0.7rem 0.9rem 0.2rem 0.9rem;
                margin-top:0.4rem;">
        <div style="color:#e2e8f0; font-weight:600; font-size:0.85rem; margin-bottom:0.2rem;">{name}</div>
        <div style="color:{color}; font-weight:700; font-size:1.05rem;">{current:,.2f}</div>
        <div style="color:{color}; font-weight:600; font-size:0.75rem;">
            {arrow} {abs(change):,.2f} ({sign}{change_pct:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 미니 차트 (plotly area chart) - 컴팩트
    if prices:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=prices,
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=1.5),
            fillcolor=fill_color,
            hovertemplate="%{y:,.2f}<extra></extra>",
        ))
        ymin, ymax = min(prices), max(prices)
        margin = (ymax - ymin) * 0.15 if ymax > ymin else max(ymax * 0.001, 1)
        fig.update_yaxes(
            range=[ymin - margin, ymax + margin],
            visible=False,
        )
        fig.update_xaxes(visible=False)
        fig.update_layout(
            height=70,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main():
    st.set_page_config(
        page_title="Stock News AI Analyzer",
        page_icon="📈",
        layout="centered",
    )

    inject_custom_css()
    render_hero()

    # ── API 키 유효성 체크 ──────────────────────────────────────
    api_ok = True
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        st.warning("네이버 API 키가 설정되지 않았습니다.")
        api_ok = False
    if not GEMINI_API_KEY:
        st.warning("Gemini API 키가 설정되지 않았습니다.")
        api_ok = False

    tab1, tab2, tab4, tab5, tab6 = st.tabs([
        "  종목 검색  ", "  시장 동향  ", "  거래대금  ", "  세계 증시  ", "  🎯 종가베팅  "
    ])

    # ════════════════════════════════
    # TAB 1: 종목 검색
    # ════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">&#128270; 종목명으로 뉴스 검색</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            stock_name = st.text_input(
                label="종목명",
                placeholder="삼성전자, SK하이닉스, 카카오 ...",
                label_visibility="collapsed",
                key="stock_input",
            )
        with col2:
            search_btn = st.button("분석", use_container_width=True,
                                   disabled=not api_ok, key="stock_btn")

        if search_btn and stock_name.strip():
            query = f"{stock_name.strip()} 주식"
            with st.status(f"'{stock_name}' 분석 중...", expanded=True) as status:
                status.update(label="네이버 뉴스 수집 중...", state="running")
                news_list = fetch_naver_news(query, display=10)

                if not news_list:
                    status.update(label="뉴스 수집 실패", state="error")
                    st.error("뉴스를 가져오지 못했습니다. API 키와 검색어를 확인하세요.")
                else:
                    st.write(f"✅ 뉴스 {len(news_list)}건 수집 완료")
                    status.update(label="Gemini AI가 핵심을 분석 중입니다...", state="running")
                    news_tuple = tuple((n["title"], n["description"]) for n in news_list)
                    summary, used_model = summarize_with_gemini(stock_name, news_tuple, mode="stock")
                    st.write(f"✅ AI 분석 완료 ({used_model})")
                    status.update(label="분석 완료!", state="complete")

            if news_list:
                render_summary(f"AI 시장 동향 요약 -- {stock_name}", summary, used_model)
                render_news_list(news_list)

        elif search_btn and not stock_name.strip():
            st.warning("종목명을 입력해 주세요.")

    # ════════════════════════════════
    # TAB 2: 오늘의 시장 동향
    # ════════════════════════════════
    with tab2:
        today_str = get_kst_today_str()

        st.markdown(
            f'<div class="section-title">&#128197; {today_str} 시장 동향</div>',
            unsafe_allow_html=True,
        )

        market_choice = st.radio(
            "시장 선택",
            options=["🇰🇷 국내 증시", "🇺🇸 미국 증시"],
            horizontal=True,
            key="market_choice",
        )

        # 선택에 따른 키워드 매핑
        if market_choice.startswith("🇰🇷"):
            keywords = ["코스피", "코스닥", "한국증시"]
            market_label = "국내 증시"
        else:
            keywords = ["미국증시", "나스닥", "다우존스", "S&P500"]
            market_label = "미국 증시"

        render_keyword_chips(keywords)

        today_btn = st.button(f"{market_label} 시장 동향 분석", use_container_width=True,
                              disabled=not api_ok, key="today_btn")

        if today_btn:
            with st.status(f"{market_label} 시장 동향 분석 중...", expanded=True) as status:
                status.update(label=f"네이버 뉴스 수집 중... ({len(keywords)}개 키워드 병렬 처리)", state="running")
                unique_news = fetch_multiple_keywords(keywords, display=5)

                if not unique_news:
                    status.update(label="뉴스 수집 실패", state="error")
                    st.error("뉴스를 가져오지 못했습니다.")
                else:
                    st.write(f"✅ 뉴스 {len(unique_news)}건 수집 완료 (중복 제거)")
                    status.update(label="Gemini AI가 오늘의 시장을 분석 중입니다...", state="running")
                    news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
                    summary, used_model = summarize_with_gemini(
                        ", ".join(keywords), news_tuple, mode="today"
                    )
                    st.write(f"✅ AI 분석 완료 ({used_model})")
                    status.update(label="분석 완료!", state="complete")

            if unique_news:
                render_summary(f"{market_label} AI 시장 동향 요약", summary, used_model)
                render_news_list(unique_news)

    # ════════════════════════════════
    # TAB 4: 거래대금 TOP 100
    # ════════════════════════════════
    with tab4:
        today_str = get_kst_today_str()

        st.markdown(
            f'<div class="section-title">&#128293; {today_str} 거래대금 상위 종목</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
            'KRX 전종목 거래대금 기준 (KOSPI + KOSDAQ)</p>',
            unsafe_allow_html=True,
        )

        # 장 운영 시간 체크 (평일 09:00~15:30)
        now_kst = get_kst_now()
        is_weekday = now_kst.weekday() < 5  # 월~금
        market_open = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now_kst.replace(hour=15, minute=30, second=0, microsecond=0)
        is_market_open = is_weekday and market_open <= now_kst <= market_close

        if is_market_open:
            st.warning(
                "현재 장 운영 시간(09:00~15:30)입니다. "
                "거래대금 데이터는 **전일 장 마감 기준**으로 제공되며, "
                "당일 실시간 데이터는 지원되지 않습니다. "
                "장 마감(15:30) 이후 조회하시면 당일 데이터를 확인할 수 있습니다."
            )

        vol_btn = st.button("거래대금 TOP 100 조회", use_container_width=True, key="vol_btn")

        if vol_btn:
            with st.status("KRX 거래대금 데이터 조회 중...", expanded=True) as status:
                status.update(label="FinanceDataReader에서 전종목 데이터 수집 중...", state="running")
                top_list, err_msg = fetch_trading_volume_top(100)

                if not top_list:
                    status.update(label="데이터 조회 실패", state="error")
                    st.error("거래대금 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
                    if err_msg:
                        st.code(f"오류 내용: {err_msg}", language="text")
                else:
                    data_label = "전일 장 마감 기준" if is_market_open else "최근 장 마감 기준"
                    st.write(f"✅ {len(top_list)}개 종목 거래대금 데이터 수집 완료 ({data_label})")
                    status.update(label="조회 완료!", state="complete")

            if top_list:
                st.session_state["vol_data"] = top_list

        if "vol_data" in st.session_state and st.session_state["vol_data"]:
            top_list = st.session_state["vol_data"]

            # 정렬 옵션
            sort_options = {
                "거래대금 높은순": ("amount", True),
                "거래대금 낮은순": ("amount", False),
                "등락률 높은순": ("change_ratio", True),
                "등락률 낮은순": ("change_ratio", False),
            }
            selected_sort = st.selectbox(
                "정렬 기준",
                options=list(sort_options.keys()),
                key="vol_sort",
            )
            sort_key, sort_desc = sort_options[selected_sort]
            sorted_list = sorted(top_list, key=lambda x: x[sort_key], reverse=sort_desc)

            # 순위 재부여
            for i, item in enumerate(sorted_list, 1):
                item["display_rank"] = i

            # 전체 카드 렌더링
            st.markdown(
                f'<div class="section-title">&#127942; {selected_sort} TOP 100</div>',
                unsafe_allow_html=True,
            )
            for item in sorted_list:
                amt_billions = item["amount"] / 1e8
                cap_trillions = item["marcap"] / 1e12
                chg = item["change_ratio"]
                chg_color = "#ef4444" if chg < 0 else "#22c55e" if chg > 0 else "rgba(255,255,255,0.5)"
                chg_sign = "+" if chg > 0 else ""

                st.markdown(f"""
                <div class="news-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div style="flex:1; min-width:150px;">
                        <span style="color:#667eea; font-weight:700; font-size:1.1rem; margin-right:0.5rem;">{item["display_rank"]}</span>
                        <span class="news-title" style="display:inline; font-size:1rem;">{item["name"]}</span>
                        <span style="color:rgba(255,255,255,0.3); font-size:0.75rem; margin-left:0.4rem;">{item["code"]} &middot; {item["market"]}</span>
                    </div>
                    <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;">
                        <div style="text-align:right;">
                            <div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">거래대금</div>
                            <div style="color:#e2e8f0; font-weight:600; font-size:0.95rem;">{amt_billions:,.0f}억</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">등락률</div>
                            <div style="color:{chg_color}; font-weight:600; font-size:0.95rem;">{chg_sign}{chg:.2f}%</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">종가</div>
                            <div style="color:#e2e8f0; font-size:0.9rem;">{item["close"]:,}원</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">시총</div>
                            <div style="color:rgba(255,255,255,0.6); font-size:0.85rem;">{cap_trillions:.1f}조</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════
    # TAB 5: 세계 증시 현황
    # ════════════════════════════════
    with tab5:
        st.markdown(
            '<div class="section-title">&#127760; 세계 주요 증시 현황</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
            '미국 / 한국 / 아시아 주요 지수 (5분 캐시 · yfinance)</p>',
            unsafe_allow_html=True,
        )

        col_filter, col_refresh = st.columns([3, 1])
        with col_filter:
            region_filter = st.selectbox(
                "지역 필터",
                options=["🌐 전체", "🇺🇸 미국", "🇰🇷 한국", "🌏 아시아"],
                key="world_region_filter",
                label_visibility="collapsed",
            )
        with col_refresh:
            world_btn = st.button("🔄 새로고침", use_container_width=True, key="world_refresh_btn")

        if world_btn:
            fetch_world_indices.clear()

        with st.spinner("세계 증시 데이터 수집 중..."):
            world_data = fetch_world_indices()

        if not world_data:
            st.error("데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            # 미국 / 한국 / 아시아 그룹핑
            us_names    = {"다우산업", "나스닥종합", "S&P 500"}
            kr_names    = {"코스피", "코스닥"}
            asia_names  = {"닛케이225", "상해종합", "항셍"}

            us_group    = [d for d in world_data if d["name"] in us_names]
            kr_group    = [d for d in world_data if d["name"] in kr_names]
            asia_group  = [d for d in world_data if d["name"] in asia_names]

            def render_grid(items: list, cols: int = 3):
                """N개 카드를 N열 그리드로 렌더링"""
                for i in range(0, len(items), cols):
                    row = items[i:i + cols]
                    col_objs = st.columns(cols)
                    for j, item in enumerate(row):
                        with col_objs[j]:
                            render_world_index_card(item)

            show_all = region_filter == "🌐 전체"

            if show_all or "미국" in region_filter:
                if us_group:
                    st.markdown(
                        '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">&#127482;&#127480; 미국 증시</div>',
                        unsafe_allow_html=True,
                    )
                    render_grid(us_group, cols=3)

            if show_all or "한국" in region_filter:
                if kr_group:
                    st.markdown(
                        '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">&#127472;&#127479; 한국 증시</div>',
                        unsafe_allow_html=True,
                    )
                    render_grid(kr_group, cols=2)

            if show_all or "아시아" in region_filter:
                if asia_group:
                    st.markdown(
                        '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">&#127759; 아시아 증시</div>',
                        unsafe_allow_html=True,
                    )
                    render_grid(asia_group, cols=3)

    # ════════════════════════════════
    # TAB 6: 오후 시황 & 종가베팅
    # ════════════════════════════════
    with tab6:
        st.markdown(
            '<div class="section-title">&#127919; 오후 시황 & 종가베팅 대시보드</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
            '장 막판 특징주 + 주도 테마를 빠르게 파악 (14:30~15:20 활용 추천)</p>',
            unsafe_allow_html=True,
        )

        # ── 자동 새로고침 토글 (기본 OFF) ──
        col_auto, col_info = st.columns([2, 3])
        with col_auto:
            auto_refresh = st.toggle(
                "⚡ 1분 자동 새로고침",
                value=False,
                key="bet_auto_refresh",
                help="장중에만 켜는 것을 추천. 캐시 ttl 안에서는 추가 API 호출 없음.",
            )
        with col_info:
            if auto_refresh:
                try:
                    from streamlit_autorefresh import st_autorefresh
                    refresh_count = st_autorefresh(interval=60_000, key="bet_autorefresh_counter")
                    st.caption(f"🔄 #{refresh_count}회 새로고침 · {get_kst_now().strftime('%H:%M:%S')} KST")
                except ImportError:
                    st.caption("⚠️ streamlit-autorefresh 패키지 미설치")

        # ── 서브탭 (2개) ──
        bet_tab1, bet_tab2 = st.tabs([
            "  🔥 특징주 (실시간/마감)  ",
            "  🤖 AI 종가베팅 분석  ",
        ])

        # ── BET-TAB1: 특징주 (등락률 +10% 이상, ETF/스팩 제외, 최대 50개) ──
        with bet_tab1:
            st.caption("실시간 등락률 +10% 이상 종목 · ETF/스팩/ETN 제외 · 최대 50개")

            feat_btn = st.button("특징주 조회", use_container_width=True, key="feat_btn")
            if feat_btn:
                with st.spinner("특징주 수집 중..."):
                    raw = fetch_realtime_top_gainers(150)
                    feature_list = [
                        s for s in raw
                        if s["change_ratio"] >= 10.0 and not is_etf_or_spac(s["name"])
                    ][:50]
                    # display_rank 재부여
                    for i, s in enumerate(feature_list, 1):
                        s["display_rank"] = i
                    source_label = "실시간 +10% 이상"

                if not feature_list:
                    st.warning("등락률 +10% 이상 특징주가 없습니다.")
                else:
                    st.success(f"✅ {len(feature_list)}개 특징주 ({source_label})")
                    # 세션에 저장 → AI 분석 탭에서 재사용
                    st.session_state["bet_features"] = feature_list
                    st.session_state["bet_source_label"] = source_label

                    for s in feature_list:
                        chg = s["change_ratio"]
                        amt_eok = s["amount"] / 1e8
                        st.markdown(f"""
                        <div class="news-card" style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="flex:1;">
                                <span style="color:#667eea; font-weight:700; margin-right:0.5rem;">{s['display_rank']}</span>
                                <strong>{s['name']}</strong>
                                <span style="color:rgba(255,255,255,0.4); font-size:0.75rem; margin-left:0.4rem;">{s['code']} · {s.get('market', '')}</span>
                            </div>
                            <div style="display:flex; gap:1.2rem;">
                                <span style="color:#e2e8f0;">{s.get('current', 0):,}원</span>
                                <span style="color:#e2e8f0;">{amt_eok:,.0f}억</span>
                                <span style="color:#ef4444; font-weight:600;">+{chg:.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 이전 조회 결과가 있으면 안내
            if "bet_features" in st.session_state and not feat_btn:
                st.info(
                    f"💾 이전 조회 결과 {len(st.session_state['bet_features'])}개 종목이 "
                    f"AI 분석 탭에서 사용 가능합니다 (소스: {st.session_state.get('bet_source_label', '?')})."
                )

        # ── BET-TAB2: AI 종가베팅 분석 ──
        with bet_tab2:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.6); font-size:0.85rem;">'
                '특징주 + 종목별 뉴스를 Gemini에게 던져서 <strong>주도 테마 TOP3</strong> + '
                '<strong>상승 사유</strong> + <strong>종가베팅 코멘트</strong> 분석</p>',
                unsafe_allow_html=True,
            )

            features_in_state = st.session_state.get("bet_features", [])

            if not features_in_state:
                st.warning("⚠️ 먼저 옆의 [특징주] 탭에서 특징주를 조회해 주세요.")
            else:
                src = st.session_state.get("bet_source_label", "?")
                analyze_count = min(10, len(features_in_state))
                st.markdown(
                    f"📋 분석 대상: **{len(features_in_state)}개 종목** ({src}) → "
                    f"상위 **{analyze_count}개** 분석"
                )

                ai_btn = st.button("🚀 AI 종가베팅 분석 실행", use_container_width=True,
                                   disabled=not api_ok, key="ai_bet_btn")

                if ai_btn:
                    targets = features_in_state[:analyze_count]
                    with st.status("AI 종가베팅 분석 진행 중...", expanded=True) as status:
                        status.update(label="① 종목별 최신 뉴스 수집 중...", state="running")
                        news_map = fetch_news_per_stock(targets, display=3)
                        st.write(f"✅ {len(news_map)}개 종목 뉴스 수집 완료")

                        status.update(label="② Gemini AI 분석 중 (폴백 체인 적용)...", state="running")
                        stocks_tuple = tuple(
                            (s["name"], s["code"], s["change_ratio"], s["amount"] / 1e8)
                            for s in targets
                        )
                        news_tuple = tuple(
                            (name, tuple((n["title"], n["description"]) for n in items))
                            for name, items in news_map.items()
                        )
                        bet_summary, bet_model = analyze_closing_bet(stocks_tuple, news_tuple)
                        st.write(f"✅ AI 분석 완료 ({bet_model})")
                        status.update(label="완료!", state="complete")

                    render_summary("🔥 종가베팅 AI 리포트", bet_summary, bet_model)

    # ── 사이드바 ────────────────────────────────────────────────
    with st.sidebar:
        kst_sidebar = get_kst_now()

        st.markdown(f"""
        <div class="sidebar-model"><span>Gemini Pro → Flash → Lite → 2.0 (자동 폴백)</span></div>
        <div class="sidebar-time">{kst_sidebar.strftime('%Y-%m-%d %H:%M:%S')} KST</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>&#128270; 종목 검색</h4>
            <p>종목명 입력 &rarr; 분석 클릭 &rarr; AI 요약</p>
        </div>
        <div class="sidebar-card">
            <h4>&#128197; 시장 동향</h4>
            <p>키워드 선택/입력 &rarr; 동향 분석 &rarr; 오늘의 리포트</p>
        </div>
        <div class="sidebar-card">
            <h4>&#128293; 거래대금</h4>
            <p>KRX 전종목 거래대금 TOP 100 실시간 조회</p>
        </div>
        <div class="sidebar-card">
            <h4>&#127760; 세계 증시</h4>
            <p>미국·한국·아시아 주요 지수 실시간 차트</p>
        </div>
        <div class="sidebar-card">
            <h4>&#127919; 종가베팅</h4>
            <p>특징주 + AI 분석 + 자동 새로고침. 14:30~15:20 활용 추천</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>&#9881; 기술 스택</h4>
            <p>
            <strong>UI</strong> &middot; Streamlit<br>
            <strong>News</strong> &middot; Naver Search API<br>
            <strong>AI</strong> &middot; Google Gemini 2.5 Flash Lite
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer-notice">
            &#9888; 본 서비스는 정보 제공 목적이며,<br>투자 권유가 아닙니다.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
