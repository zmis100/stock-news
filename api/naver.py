"""네이버 뉴스 API + 네이버 금융 스크래핑"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
from utils.text import clean_html, safe_int, safe_float

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ── 뉴스 API ─────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news(query: str, display: int = 10) -> list[dict]:
    """네이버 뉴스 검색 (1시간 캐시)"""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id":     NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": "date"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            log.warning("Naver news API %s: %s", r.status_code, r.text[:200])
            return []
        items = r.json().get("items", [])
        return [{
            "title":       clean_html(it.get("title", "")),
            "link":        it.get("link", ""),
            "description": clean_html(it.get("description", "")),
            "pubDate":     it.get("pubDate", ""),
        } for it in items]
    except requests.exceptions.Timeout:
        log.warning("Naver news API timeout for %s", query)
        return []
    except Exception as e:
        log.exception("Naver news API error: %s", e)
        return []


def fetch_news_multi(keywords: list[str], display: int = 5) -> list[dict]:
    """여러 키워드 병렬 수집 + 중복 제거"""
    all_news = []
    with ThreadPoolExecutor(max_workers=min(len(keywords), 5)) as ex:
        futures = {
            ex.submit(fetch_news, f"{kw} 오늘", display): kw
            for kw in keywords
        }
        for f in as_completed(futures):
            try:
                all_news.extend(f.result())
            except Exception as e:
                log.warning("fetch_news_multi failed: %s", e)

    seen = set()
    unique = []
    for n in all_news:
        if n["title"] not in seen:
            seen.add(n["title"])
            unique.append(n)
    return unique


def fetch_news_per_stock(
    stocks: list[dict],
    display: int = 3,
    priority_keyword: str | None = None,
) -> dict[str, list[dict]]:
    """각 종목별 뉴스 병렬 수집.

    priority_keyword 지정 시: 제목/본문에 해당 키워드가 들어간 뉴스를 결과 상단으로 정렬.
    """
    result: dict[str, list[dict]] = {}
    if not stocks:
        return result
    with ThreadPoolExecutor(max_workers=min(len(stocks), 8)) as ex:
        futures = {ex.submit(fetch_news, s["name"], display): s["name"] for s in stocks}
        for f in as_completed(futures):
            name = futures[f]
            try:
                result[name] = f.result()
            except Exception as e:
                log.warning("fetch_news_per_stock %s failed: %s", name, e)
                result[name] = []

    if priority_keyword:
        for name, items in result.items():
            preferred, others = [], []
            for n in items:
                if (priority_keyword in n.get("title", "")
                        or priority_keyword in n.get("description", "")):
                    preferred.append(n)
                else:
                    others.append(n)
            result[name] = preferred + others
    return result


# ── 거래대금 TOP 100 (네이버 모바일 금융 API) ─────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_trading_volume_top(limit: int = 100) -> tuple[list[dict], str]:
    """KOSPI+KOSDAQ 거래대금 상위 (5분 캐시)"""
    try:
        session = requests.Session()
        session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
        session.headers.update({"User-Agent": _USER_AGENT})

        def fetch_page(market, page):
            url = f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize=100"
            r = session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return market, data.get("stocks", []), data.get("totalCount", 0)

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(fetch_page, m, 1): m for m in ["KOSPI", "KOSDAQ"]}
            first = {}
            for f in as_completed(futures):
                market, stocks, total = f.result()
                first[market] = (stocks, total)

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

        all_stocks.sort(key=lambda x: safe_int(x[0].get("accumulatedTradingValue", 0)), reverse=True)

        # 종목코드 기준 중복 제거 (페이지 경계 / KOSPI·KOSDAQ 이중 노출 대비)
        seen_codes: set[str] = set()
        unique_stocks = []
        for s, m in all_stocks:
            code = s.get("itemCode", "")
            if code and code not in seen_codes:
                seen_codes.add(code)
                unique_stocks.append((s, m))
        all_stocks = unique_stocks

        result = []
        for s, market in all_stocks[:limit]:
            amount_mil = safe_int(s.get("accumulatedTradingValue", 0))
            result.append({
                "rank": len(result) + 1,
                "name": s.get("stockName", ""),
                "code": s.get("itemCode", ""),
                "market": market,
                "close": safe_int(s.get("closePrice", 0)),
                "change_ratio": safe_float(s.get("fluctuationsRatio", 0)),
                "volume": safe_int(s.get("accumulatedTradingVolume", 0)),
                "amount": amount_mil * 1_000_000,            # 백만원 → 원
                "marcap": safe_int(s.get("marketValue", 0)) * 100_000_000,  # 억원 → 원
            })
        return result, ""
    except Exception as e:
        log.exception("fetch_trading_volume_top failed")
        return [], str(e)


# ── 실시간 등락률 상위 (네이버 금융 HTML 스크래핑) ────────────

_ETF_BRAND_PREFIXES = (
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "KOSEF", "HANARO",
    "ACE", "KINDEX", "SOL", "HK", "KCGI", "RISE", "WOORI",
    "BNK", "KIWOOM", "TIMEFOLIO", "PLUS",
)


def is_etf_or_spac(name: str) -> bool:
    """ETF/ETN/스팩 여부 (종목명 기반)"""
    if not name:
        return False
    upper = name.upper().strip()
    if any(upper.startswith(p) for p in _ETF_BRAND_PREFIXES):
        return True
    return "스팩" in name or "SPAC" in upper or "ETN" in upper


# ── 외국인/기관 순매수 (네이버 종목 frgn 페이지 스크래핑) ─────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_foreign_institution(code: str) -> dict | None:
    """단일 종목의 가장 최근 거래일 외국인/기관 순매매 수량.

    Returns: {"date": "YYYY.MM.DD", "close": int, "inst_qty": int, "foreign_qty": int}
    음수 = 순매도. 1시간 캐시.
    """
    try:
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=8)
        r.encoding = "euc-kr"
        soup = BeautifulSoup(r.text, "html.parser")

        tables = soup.select("table.type2")
        if len(tables) < 2:
            return None

        # type2 두 번째 테이블이 외국인/기관 순매매 표
        for row in tables[1].select("tr"):
            cells = [c.get_text(strip=True) for c in row.select("td")]
            # 데이터 행: [날짜, 종가, 전일비, 등락률, 거래량, 기관, 외국인, 보유주수, 보유율]
            if len(cells) >= 7 and cells[0] and "." in cells[0]:
                return {
                    "date":        cells[0],
                    "close":       safe_int(cells[1]),
                    "inst_qty":    _parse_signed(cells[5]),
                    "foreign_qty": _parse_signed(cells[6]),
                }
        return None
    except Exception as e:
        log.warning("fetch_foreign_institution %s failed: %s", code, e)
        return None


def _parse_signed(s: str) -> int:
    """+133,425 / -28,964 같은 부호 있는 숫자 파싱"""
    if not s:
        return 0
    s = s.replace(",", "").replace(" ", "")
    sign = -1 if s.startswith("-") else 1
    return sign * safe_int(s.lstrip("+-"))


def fetch_foreign_institution_bulk(codes: list[str]) -> dict[str, dict]:
    """여러 종목 외국인/기관 순매매 병렬 수집. {code: {...}}"""
    result: dict[str, dict] = {}
    if not codes:
        return result

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(fetch_foreign_institution, c): c for c in codes}
        for f in as_completed(futures):
            code = futures[f]
            try:
                data = f.result()
                if data:
                    result[code] = data
            except Exception as e:
                log.warning("fetch_foreign_institution_bulk %s failed: %s", code, e)
    return result


# ── 실시간 등락률 상위 (네이버 금융 HTML 스크래핑) ────────────

@st.cache_data(ttl=120, show_spinner=False)
def fetch_realtime_top_gainers(limit: int = 30) -> list[dict]:
    """실시간 등락률 상위 (KOSPI + KOSDAQ). 2분 캐시."""
    headers = {"User-Agent": _USER_AGENT}
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

                name = a_tag.text.strip()
                href = a_tag.get("href", "")
                code = href.split("code=")[-1] if "code=" in href else ""
                price = safe_int(cols[2].text.strip())
                chg = safe_float(cols[4].text.strip())
                volume = safe_int(cols[5].text.strip())
                amt_m = safe_int(cols[6].text.strip())
                if price == 0:
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
        except Exception as e:
            log.warning("fetch_realtime_top_gainers %s failed: %s", market, e)
            continue

    all_stocks.sort(key=lambda x: x["change_ratio"], reverse=True)
    for i, s in enumerate(all_stocks[:limit], 1):
        s["display_rank"] = i
    return all_stocks[:limit]
