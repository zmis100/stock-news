import streamlit as st
import requests
import os
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

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

GEMINI_MODEL = "models/gemini-2.5-flash"

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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_naver_news(query: str, display: int = 10) -> list[dict]:
    """네이버 뉴스 검색 API 호출 (5분 캐시)"""
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


@st.cache_data(ttl=600, show_spinner=False)
def summarize_with_gemini(label: str, news_titles_and_descs: tuple, mode: str = "stock") -> str:
    """Gemini API 호출 (10분 캐시, 같은 뉴스면 재호출 안함)"""
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

### 📈 핵심 동향
- 최근 주요 뉴스에서 파악되는 핵심 트렌드 3~5가지를 불릿 포인트로 정리

### 🏢 관련 주요 종목
- 이 테마와 관련된 대표 종목들을 언급하고 최근 뉴스에서의 동향 설명

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

def main():
    st.set_page_config(
        page_title="주식 뉴스 AI 요약",
        page_icon="📈",
        layout="centered",
    )

    st.title("📈 주식 뉴스 AI 시장 동향 분석기")
    st.caption("종목명 검색 또는 오늘의 시장 동향을 AI가 요약해 드립니다.")
    st.divider()

    # ── API 키 유효성 체크 ──────────────────────────────────────
    api_ok = True
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        st.warning("⚠️ 네이버 API 키가 설정되지 않았습니다. secrets.toml을 확인하세요.")
        api_ok = False
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API 키가 설정되지 않았습니다. secrets.toml을 확인하세요.")
        api_ok = False

    tab1, tab2, tab3 = st.tabs(["🔍 종목명 검색", "📅 오늘의 시장 동향", "📊 테마별 분석"])

    # ════════════════════════════════
    # TAB 1: 종목 검색
    # ════════════════════════════════
    with tab1:
        st.subheader("종목명으로 뉴스 검색 및 AI 요약")

        col1, col2 = st.columns([4, 1])
        with col1:
            stock_name = st.text_input(
                label="종목명",
                placeholder="예: 삼성전자, SK하이닉스, 카카오",
                label_visibility="collapsed",
                key="stock_input",
            )
        with col2:
            search_btn = st.button("🔍 분석", use_container_width=True,
                                   disabled=not api_ok, key="stock_btn")

        if search_btn and stock_name.strip():
            query = f"{stock_name.strip()} 주식"

            with st.spinner(f"📡 '{stock_name}' 관련 최신 뉴스 수집 중..."):
                news_list = fetch_naver_news(query, display=10)

            if not news_list:
                st.error("뉴스를 가져오지 못했습니다. API 키와 검색어를 확인하세요.")
            else:
                # 캐시 가능하도록 tuple로 변환
                news_tuple = tuple((n["title"], n["description"]) for n in news_list)
                with st.spinner("🤖 Gemini AI가 시장 동향 분석 중..."):
                    summary = summarize_with_gemini(stock_name, news_tuple, mode="stock")

                st.subheader(f"🧠 AI 시장 동향 요약 — {stock_name}")
                st.info(summary)
                st.divider()

                st.subheader(f"📰 수집된 뉴스 ({len(news_list)}건)")
                for i, news in enumerate(news_list, 1):
                    with st.expander(f"{i}. {news['title']}"):
                        st.caption(f"🕒 {news.get('pubDate', '')}")
                        st.write(news["description"])
                        st.markdown(f"[🔗 원문 보기]({news['link']})")

        elif search_btn and not stock_name.strip():
            st.warning("종목명을 입력해 주세요.")

    # ════════════════════════════════
    # TAB 2: 오늘의 시장 동향
    # ════════════════════════════════
    with tab2:
        # 항상 한국시간 기준 실시간 날짜 표시
        kst_now = get_kst_now()
        today_str = get_kst_today_str()
        time_str = kst_now.strftime("%H:%M")

        st.subheader(f"📅 {today_str} 주요 시장 동향")
        st.caption(f"한국시간 {time_str} 기준 · 주요 경제·증시 뉴스를 수집해 AI가 요약합니다.")

        preset_keywords = st.multiselect(
            "추천 키워드에서 선택 (복수 선택 가능)",
            options=["코스피", "코스닥", "미국증시", "환율", "금리", "반도체", "2차전지", "AI 주식"],
            default=["코스피", "코스닥"],
        )

        custom_input = st.text_input(
            "직접 키워드 입력 (쉼표로 구분)",
            placeholder="예: 전기차, 삼성전자, 유가",
            key="custom_keywords",
        )

        custom_keywords = [k.strip() for k in custom_input.split(",") if k.strip()] if custom_input else []
        keywords = list(dict.fromkeys(preset_keywords + custom_keywords))

        if keywords:
            st.caption(f"분석 키워드: {', '.join(keywords)}")

        today_btn = st.button("📅 오늘 시장 동향 분석", use_container_width=True,
                              disabled=not api_ok, key="today_btn")

        if today_btn:
            if not keywords:
                st.warning("키워드를 하나 이상 선택해 주세요.")
            else:
                with st.spinner(f"📡 {len(keywords)}개 키워드 뉴스 동시 수집 중..."):
                    unique_news = fetch_multiple_keywords(keywords, display=5)

                if not unique_news:
                    st.error("뉴스를 가져오지 못했습니다.")
                else:
                    news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
                    with st.spinner("🤖 Gemini AI가 오늘의 시장 동향 분석 중..."):
                        summary = summarize_with_gemini(
                            ", ".join(keywords), news_tuple, mode="today"
                        )

                    st.subheader("🧠 오늘의 AI 시장 동향 요약")
                    st.info(summary)
                    st.divider()

                    st.subheader(f"📰 수집된 뉴스 ({len(unique_news)}건)")
                    for i, news in enumerate(unique_news, 1):
                        with st.expander(f"{i}. {news['title']}"):
                            st.caption(f"🕒 {news.get('pubDate', '')}")
                            st.write(news["description"])
                            st.markdown(f"[🔗 원문 보기]({news['link']})")

    # ════════════════════════════════
    # TAB 3: 테마별 분석
    # ════════════════════════════════
    with tab3:
        st.subheader("📊 투자 테마별 심층 분석")
        st.caption("관심 있는 투자 테마를 선택하면 관련 뉴스를 수집하고 AI가 심층 분석합니다.")

        theme_categories = {
            "🔋 에너지·소재": ["2차전지", "태양광", "원자력", "수소에너지", "희토류"],
            "💻 IT·기술": ["반도체", "AI 인공지능", "클라우드", "로봇", "자율주행"],
            "🏥 바이오·헬스": ["바이오", "제약", "헬스케어", "신약개발"],
            "💰 금융·경제": ["금리", "환율", "부동산", "가상화폐", "IPO"],
            "🌍 글로벌": ["미국증시", "중국경제", "유럽증시", "신흥국"],
            "🏗️ 산업·인프라": ["조선", "방산", "건설", "항공우주"],
        }

        selected_category = st.selectbox(
            "테마 카테고리 선택",
            options=list(theme_categories.keys()),
            key="theme_category",
        )

        selected_theme = st.selectbox(
            "분석할 테마 선택",
            options=theme_categories[selected_category],
            key="theme_select",
        )

        theme_btn = st.button("📊 테마 심층 분석", use_container_width=True,
                              disabled=not api_ok, key="theme_btn")

        if theme_btn:
            search_queries = [f"{selected_theme} 관련주", f"{selected_theme} 시장", f"{selected_theme} 전망"]
            all_news = []
            for q in search_queries:
                with st.spinner(f"📡 '{q}' 뉴스 수집 중..."):
                    news = fetch_naver_news(q, display=5)
                    all_news.extend(news)

            seen = set()
            unique_news = []
            for n in all_news:
                if n["title"] not in seen:
                    seen.add(n["title"])
                    unique_news.append(n)

            if not unique_news:
                st.error("뉴스를 가져오지 못했습니다.")
            else:
                news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
                with st.spinner(f"🤖 Gemini AI가 '{selected_theme}' 테마 심층 분석 중..."):
                    summary = summarize_with_gemini(selected_theme, news_tuple, mode="theme")

                st.subheader(f"🧠 '{selected_theme}' 테마 심층 분석")
                st.info(summary)
                st.divider()

                st.subheader(f"📰 수집된 뉴스 ({len(unique_news)}건)")
                for i, news in enumerate(unique_news, 1):
                    with st.expander(f"{i}. {news['title']}"):
                        st.caption(f"🕒 {news.get('pubDate', '')}")
                        st.write(news["description"])
                        st.markdown(f"[🔗 원문 보기]({news['link']})")

    # ── 사이드바 ────────────────────────────────────────────────
    with st.sidebar:
        st.success(f"✅ AI 모델: `{GEMINI_MODEL}`")
        kst_sidebar = get_kst_now()
        st.caption(f"🕒 한국시간: {kst_sidebar.strftime('%Y-%m-%d %H:%M:%S')}")
        st.header("📖 사용 방법")
        st.markdown("""
**🔍 종목명 검색 탭**
1. 분석할 종목명 입력
   예) `삼성전자`, `NAVER`, `현대차`
2. 🔍 분석 버튼 클릭
3. AI가 시장 동향 자동 요약

**📅 오늘의 시장 동향 탭**
1. 관심 키워드 선택
2. 오늘 시장 동향 분석 클릭
3. 오늘 날짜 기준 뉴스 AI 요약

**📊 테마별 분석 탭**
1. 카테고리 선택 (IT, 에너지, 바이오 등)
2. 세부 테마 선택
3. 테마 심층 분석 클릭
4. AI가 호재/악재, 관련종목, 전망 분석
""")
        st.divider()
        st.header("⚙️ 기술 스택")
        st.markdown("""
- **UI**: Streamlit
- **뉴스**: 네이버 검색 Open API
- **AI**: Google Gemini 2.5 Flash
""")
        st.divider()
        st.caption("⚠️ 본 서비스는 정보 제공 목적이며, 투자 권유가 아닙니다.")


if __name__ == "__main__":
    main()
