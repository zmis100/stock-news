import streamlit as st
import requests
import os
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
import FinanceDataReader as fdr

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

GEMINI_MODEL = "models/gemini-2.5-flash-lite"

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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trading_volume_top(limit: int = 100) -> list[dict]:
    """KRX 전종목 거래대금 상위 종목 조회 (1시간 캐시)"""
    try:
        df = fdr.StockListing('KRX')
        df['Amount'] = df['Amount'].astype(float)
        df = df[df['Amount'] > 0]
        top = df.nlargest(limit, 'Amount')
        result = []
        for _, row in top.iterrows():
            result.append({
                "rank": len(result) + 1,
                "name": row["Name"],
                "code": row["Code"],
                "market": row["Market"],
                "close": int(float(row["Close"])),
                "change_ratio": round(float(row["ChagesRatio"]), 2),
                "volume": int(float(row["Volume"])),
                "amount": int(float(row["Amount"])),
                "marcap": int(float(row["Marcap"])),
            })
        return result
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_with_gemini(label: str, news_titles_and_descs: tuple, mode: str = "stock") -> str:
    """Gemini API 호출 (1시간 캐시, 같은 뉴스면 재호출 안함)"""
    if not news_titles_and_descs:
        return "요약할 뉴스 기사가 없습니다."

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

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return (
            f"❌ Gemini API 호출 실패\n\n"
            f"**오류:** {e}\n\n"
            f"[Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키를 확인하세요."
        )


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


def render_summary(title: str, content: str):
    """AI 요약 결과를 프리미엄 카드로 렌더링"""
    st.markdown(f'<div class="section-title">&#129504; {title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-card">', unsafe_allow_html=True)
    st.markdown(content)
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

    tab1, tab2, tab3, tab4 = st.tabs(["  종목 검색  ", "  시장 동향  ", "  테마 분석  ", "  거래대금  "])

    # ════════════════════════════════
    # TAB 1: 종목 검색
    # ════════════════════════════════
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

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
                    summary = summarize_with_gemini(stock_name, news_tuple, mode="stock")
                    st.write("✅ AI 분석 완료")
                    status.update(label="분석 완료!", state="complete")

            if news_list:
                render_summary(f"AI 시장 동향 요약 -- {stock_name}", summary)
                render_news_list(news_list)

        elif search_btn and not stock_name.strip():
            st.warning("종목명을 입력해 주세요.")

    # ════════════════════════════════
    # TAB 2: 오늘의 시장 동향
    # ════════════════════════════════
    with tab2:
        today_str = get_kst_today_str()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">&#128197; {today_str} 시장 동향</div>',
            unsafe_allow_html=True,
        )

        preset_keywords = st.multiselect(
            "추천 키워드 선택",
            options=["코스피", "코스닥", "미국증시", "환율", "금리", "반도체", "2차전지", "AI 주식"],
            default=["코스피", "코스닥"],
        )
        custom_input = st.text_input(
            "직접 키워드 입력 (쉼표로 구분)",
            placeholder="전기차, 삼성전자, 유가 ...",
            key="custom_keywords",
        )
        custom_keywords = [k.strip() for k in custom_input.split(",") if k.strip()] if custom_input else []
        keywords = list(dict.fromkeys(preset_keywords + custom_keywords))

        if keywords:
            render_keyword_chips(keywords)

        today_btn = st.button("시장 동향 분석", use_container_width=True,
                              disabled=not api_ok, key="today_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if today_btn:
            if not keywords:
                st.warning("키워드를 하나 이상 선택해 주세요.")
            else:
                with st.status(f"{len(keywords)}개 키워드 시장 동향 분석 중...", expanded=True) as status:
                    status.update(label=f"네이버 뉴스 수집 중... ({len(keywords)}개 키워드 병렬 처리)", state="running")
                    unique_news = fetch_multiple_keywords(keywords, display=5)

                    if not unique_news:
                        status.update(label="뉴스 수집 실패", state="error")
                        st.error("뉴스를 가져오지 못했습니다.")
                    else:
                        st.write(f"✅ 뉴스 {len(unique_news)}건 수집 완료 (중복 제거)")
                        status.update(label="Gemini AI가 오늘의 시장을 분석 중입니다...", state="running")
                        news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
                        summary = summarize_with_gemini(
                            ", ".join(keywords), news_tuple, mode="today"
                        )
                        st.write("✅ AI 분석 완료")
                        status.update(label="분석 완료!", state="complete")

                if unique_news:
                    render_summary("오늘의 AI 시장 동향 요약", summary)
                    render_news_list(unique_news)

    # ════════════════════════════════
    # TAB 3: 테마별 분석
    # ════════════════════════════════
    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">&#128202; 투자 테마 심층 분석</div>', unsafe_allow_html=True)

        all_themes = [
            "2차전지", "태양광", "원자력", "수소에너지", "희토류",
            "반도체", "AI 인공지능", "클라우드", "로봇", "자율주행",
            "바이오", "제약", "헬스케어", "신약개발",
            "금리", "환율", "부동산", "가상화폐", "IPO",
            "미국증시", "중국경제", "유럽증시", "신흥국",
            "조선", "방산", "건설", "항공우주", "전력설비", "방위산업",
        ]

        selected_theme = st.selectbox(
            "추천 테마에서 선택",
            options=all_themes,
            key="theme_select",
        )

        custom_theme = st.text_input(
            "또는 직접 테마 입력",
            placeholder="전력설비, 스마트팩토리, K-뷰티 ...",
            key="custom_theme",
        )

        final_theme = custom_theme.strip() if custom_theme.strip() else selected_theme

        if final_theme:
            render_keyword_chips([final_theme])

        theme_btn = st.button("테마 심층 분석", use_container_width=True,
                              disabled=not api_ok, key="theme_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if theme_btn:
            search_queries = [f"{final_theme} 관련주", f"{final_theme} 시장", f"{final_theme} 전망"]
            with st.status(f"'{final_theme}' 테마 심층 분석 중...", expanded=True) as status:
                all_news = []
                for i, q in enumerate(search_queries, 1):
                    status.update(label=f"네이버 뉴스 수집 중... ({i}/{len(search_queries)}: {q})", state="running")
                    news = fetch_naver_news(q, display=5)
                    all_news.extend(news)

                seen = set()
                unique_news = []
                for n in all_news:
                    if n["title"] not in seen:
                        seen.add(n["title"])
                        unique_news.append(n)

                if not unique_news:
                    status.update(label="뉴스 수집 실패", state="error")
                    st.error("뉴스를 가져오지 못했습니다.")
                else:
                    st.write(f"✅ 뉴스 {len(unique_news)}건 수집 완료 (중복 제거)")
                    status.update(label=f"Gemini AI가 '{final_theme}' 대장주·관련주를 분석 중입니다...", state="running")
                    news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
                    summary = summarize_with_gemini(final_theme, news_tuple, mode="theme")
                    st.write("✅ AI 심층 분석 완료 (대장주 랭킹 포함)")
                    status.update(label="분석 완료!", state="complete")

            if unique_news:
                render_summary(f"'{final_theme}' 테마 심층 분석", summary)
                render_news_list(unique_news)

    # ════════════════════════════════
    # TAB 4: 거래대금 TOP 100
    # ════════════════════════════════
    with tab4:
        today_str = get_kst_today_str()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">&#128293; {today_str} 거래대금 상위 종목</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
            'KRX 전종목 거래대금 기준 (KOSPI + KOSDAQ)</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

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
                top_list = fetch_trading_volume_top(100)

                if not top_list:
                    status.update(label="데이터 조회 실패", state="error")
                    st.error("거래대금 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
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

    # ── 사이드바 ────────────────────────────────────────────────
    with st.sidebar:
        kst_sidebar = get_kst_now()

        st.markdown(f"""
        <div class="sidebar-model"><span>Gemini 2.5 Flash Lite</span></div>
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
            <h4>&#128202; 테마 분석</h4>
            <p>테마 선택/입력 &rarr; 심층 분석 &rarr; 대장주 랭킹</p>
        </div>
        <div class="sidebar-card">
            <h4>&#128293; 거래대금</h4>
            <p>KRX 전종목 거래대금 TOP 100 실시간 조회</p>
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
