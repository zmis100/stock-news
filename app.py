import streamlit as st
import requests
import os
from dotenv import load_dotenv
from datetime import date
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

# ✅ 실제 확인된 모델명 고정
GEMINI_MODEL = "models/gemini-2.5-flash"


# ══════════════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════════════

def fetch_naver_news(query: str, display: int = 10) -> list[dict]:
    """네이버 뉴스 검색 API 호출"""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id":     NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query":   query,
        "display": display,
        "sort":    "date",
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code == 200:
        items = response.json().get("items", [])
        import re
        news_list = []
        for item in items:
            def clean(text):
                return re.sub(r"<[^>]+>", "", text).replace("&quot;", '"').replace("&amp;", "&")
            news_list.append({
                "title":       clean(item.get("title", "")),
                "link":        item.get("link", ""),
                "description": clean(item.get("description", "")),
                "pubDate":     item.get("pubDate", ""),
            })
        return news_list
    else:
        st.error(f"네이버 API 오류: {response.status_code} — {response.text}")
        return []


def summarize_with_gemini(label: str, news_list: list[dict], mode: str = "stock") -> str:
    """확인된 모델로 Gemini API 호출"""
    if not news_list:
        return "요약할 뉴스 기사가 없습니다."

    news_text = "\n\n".join([
        f"[기사 {i+1}]\n제목: {n['title']}\n내용: {n['description']}"
        for i, n in enumerate(news_list)
    ])

    if mode == "today":
        prompt = f"""당신은 주식 시장 및 경제 분석 전문가입니다.
아래는 오늘({date.today().strftime('%Y년 %m월 %d일')}) 수집된 주요 뉴스 기사들입니다.

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

    tab1, tab2 = st.tabs(["🔍 종목명 검색", "📅 오늘의 시장 동향"])

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
                with st.spinner("🤖 Gemini AI가 시장 동향 분석 중..."):
                    summary = summarize_with_gemini(stock_name, news_list, mode="stock")

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
        today_str = date.today().strftime("%Y년 %m월 %d일")
        st.subheader(f"📅 {today_str} 주요 시장 동향")
        st.caption("오늘 날짜 기준 주요 경제·증시 뉴스를 수집해 AI가 요약합니다.")

        keywords = st.multiselect(
            "분석할 키워드 선택 (복수 선택 가능)",
            options=["코스피", "코스닥", "미국증시", "환율", "금리", "반도체", "2차전지", "AI 주식"],
            default=["코스피", "코스닥"],
        )

        today_btn = st.button("📅 오늘 시장 동향 분석", use_container_width=True,
                              disabled=not api_ok, key="today_btn")

        if today_btn:
            if not keywords:
                st.warning("키워드를 하나 이상 선택해 주세요.")
            else:
                all_news = []
                for kw in keywords:
                    with st.spinner(f"📡 '{kw}' 뉴스 수집 중..."):
                        news = fetch_naver_news(f"{kw} 오늘", display=5)
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
                    with st.spinner("🤖 Gemini AI가 오늘의 시장 동향 분석 중..."):
                        summary = summarize_with_gemini(
                            ", ".join(keywords), unique_news, mode="today"
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

    # ── 사이드바 ────────────────────────────────────────────────
    with st.sidebar:
        st.success(f"✅ AI 모델: `{GEMINI_MODEL}`")
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