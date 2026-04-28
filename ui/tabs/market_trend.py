"""TAB 2: 오늘의 시장 동향"""
import streamlit as st

from api.naver import fetch_news_multi
from api.gemini import summarize_news
from utils.kst import today_str
from ui.components import render_summary, render_news_list, render_keyword_chips


def render(api_ok: bool):
    st.markdown(
        f'<div class="section-title">&#128197; {today_str()} 시장 동향</div>',
        unsafe_allow_html=True,
    )

    market_choice = st.radio(
        "시장 선택",
        options=["🇰🇷 국내 증시", "🇺🇸 미국 증시"],
        horizontal=True,
        key="market_choice",
    )

    if market_choice.startswith("🇰🇷"):
        keywords = ["코스피", "코스닥", "한국증시"]
        market_label = "국내 증시"
    else:
        keywords = ["미국증시", "나스닥", "다우존스", "S&P500"]
        market_label = "미국 증시"

    render_keyword_chips(keywords)

    today_btn = st.button(
        f"{market_label} 시장 동향 분석",
        use_container_width=True,
        disabled=not api_ok,
        key="today_btn",
    )

    if not today_btn:
        return

    with st.status(f"{market_label} 분석 중...", expanded=True) as status:
        status.update(label=f"네이버 뉴스 수집 중... ({len(keywords)}개 키워드 병렬)", state="running")
        unique_news = fetch_news_multi(keywords, display=5)

        if not unique_news:
            status.update(label="뉴스 수집 실패", state="error")
            st.error("뉴스를 가져오지 못했습니다.")
            return

        st.write(f"✅ 뉴스 {len(unique_news)}건 (중복 제거)")
        status.update(label="Gemini AI 분석 중...", state="running")
        news_tuple = tuple((n["title"], n["description"]) for n in unique_news)
        summary, used_model = summarize_news(", ".join(keywords), news_tuple, mode="today")
        st.write(f"✅ AI 분석 완료 ({used_model})")
        status.update(label="분석 완료!", state="complete")

    render_summary(f"{market_label} AI 시장 동향 요약", summary, used_model)
    render_news_list(unique_news)
