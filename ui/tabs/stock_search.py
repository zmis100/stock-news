"""TAB 1: 종목 검색 (뉴스 + 차트 + AI 분석 + 즐겨찾기)"""
import streamlit as st

from api.naver import fetch_news
from api.gemini import summarize_news
from api.market import name_to_yf_ticker, fetch_ohlc
from storage import favorites, history
from ui.components import (
    render_summary,
    render_news_list,
    render_candle_chart,
    render_volume_chart,
)


# 주기별 (period, interval) 매핑. period는 보여줄 데이터 범위.
_INTERVAL_OPTIONS = {
    "일봉": ("6mo", "1d"),
    "주봉": ("2y",  "1wk"),
    "월봉": ("10y", "1mo"),
    "년봉": ("max", "1y"),  # fetch_ohlc 내부에서 월봉→연봉 리샘플링
}


def _run_search(target: str):
    """뉴스 + AI 요약 가져오기. (news_list, summary, used_model) 반환.
    캐시 적중 시 즉시 반환."""
    query = f"{target} 주식"
    news_list = fetch_news(query, display=10)
    if not news_list:
        return [], "", ""
    news_tuple = tuple((n["title"], n["description"]) for n in news_list)
    summary, used_model = summarize_news(target, news_tuple, mode="stock")
    return news_list, summary, used_model


def render(api_ok: bool):
    st.markdown(
        '<div class="section-title">&#128270; 종목명으로 뉴스 검색 + 차트 분석</div>',
        unsafe_allow_html=True,
    )

    # ── 사이드바 즐겨찾기 클릭으로부터 prefill 적용 ──────
    # text_input이 key를 가지고 있을 땐 value 인자가 무시되므로 session_state에 직접 주입
    prefill = st.session_state.pop("stock_prefill", None)
    if prefill is not None:
        st.session_state["stock_input"] = prefill

    # ── 입력 (form: 엔터키로도 검색됨) ──────────────────
    with st.form("stock_search_form", clear_on_submit=False, border=False):
        col1, col2 = st.columns([5, 1])
        with col1:
            stock_name = st.text_input(
                label="종목명",
                placeholder="삼성전자, SK하이닉스, 카카오 ... (엔터 또는 분석 클릭)",
                label_visibility="collapsed",
                key="stock_input",
            )
        with col2:
            search_btn = st.form_submit_button(
                "🔎 분석",
                use_container_width=True,
                disabled=not api_ok,
            )

        current = (stock_name or "").strip()
        is_fav = bool(current) and favorites.is_favorited(current)
        if not current:
            fav_label = "☆ 즐겨찾기 (먼저 종목명을 입력하세요)"
        elif is_fav:
            fav_label = f"★ '{current}' 관심종목에서 제거"
        else:
            fav_label = f"☆ '{current}' 관심종목에 추가"

        fav_btn = st.form_submit_button(
            fav_label,
            use_container_width=True,
            disabled=not current,
        )

    # ── 즐겨찾기 토글 처리 ──────────────────────────────
    if fav_btn and current:
        _, msg = favorites.toggle(current)
        st.toast(msg)
        st.rerun()

    # ── 검색 트리거 결정 ────────────────────────────────
    auto_search = st.session_state.pop("stock_auto_search", False)
    is_new_search = bool((search_btn or auto_search) and current)

    if is_new_search:
        # 새 검색: 마지막 검색 종목 갱신 + 히스토리 push
        st.session_state["last_searched_stock"] = current
        history.push(current)
        target = current
    else:
        # 새 검색이 아니면 직전 검색 결과 재표시 (캐시에서 즉시 가져옴)
        target = st.session_state.get("last_searched_stock", "")
        if not target:
            if search_btn and not current:
                st.warning("종목명을 입력해 주세요.")
            return

    # ── 데이터 가져오기 (새 검색은 status 표시, 재표시는 조용히) ──
    if is_new_search:
        with st.status(f"'{target}' 분석 중...", expanded=True) as status:
            status.update(label="네이버 뉴스 수집 중...", state="running")
            news_list, summary, used_model = _run_search(target)

            if not news_list:
                status.update(label="뉴스 수집 실패", state="error")
                st.error("뉴스를 가져오지 못했습니다. API 키와 검색어를 확인하세요.")
                # 실패 시 직전 검색 흔적 제거 → 무한 에러 방지
                st.session_state.pop("last_searched_stock", None)
                return
            st.write(f"✅ 뉴스 {len(news_list)}건 수집 완료")
            st.write(f"✅ AI 분석 완료 ({used_model})")
            status.update(label="분석 완료!", state="complete")
    else:
        news_list, summary, used_model = _run_search(target)
        if not news_list:
            st.warning(f"'{target}' 데이터가 캐시에서 사라졌습니다. 다시 분석을 눌러주세요.")
            st.session_state.pop("last_searched_stock", None)
            return

    # ── AI 요약 ──────────────────────────────────────────
    render_summary(f"AI 시장 동향 요약 -- {target}", summary, used_model)

    # ── 차트 ─────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">&#128202; 가격 차트</div>',
        unsafe_allow_html=True,
    )

    interval_label = st.radio(
        "주기",
        options=list(_INTERVAL_OPTIONS.keys()),
        index=0,
        horizontal=True,
        key=f"chart_interval_{target}",
        label_visibility="collapsed",
    )
    period, interval = _INTERVAL_OPTIONS[interval_label]

    ticker = name_to_yf_ticker(target)
    if ticker is None:
        st.info(
            "📊 차트는 거래대금 TOP100 종목만 지원됩니다. "
            "[거래대금 탭]을 한 번 새로고침한 뒤 다시 시도해 주세요."
        )
    else:
        with st.spinner("차트 데이터 수집 중..."):
            df = fetch_ohlc(ticker, period=period, interval=interval)

        if df is None or df.empty:
            st.warning(f"차트 데이터를 가져오지 못했습니다 ({ticker} · {interval_label}).")
        else:
            render_candle_chart(df, title=f"{target} ({ticker}) · {interval_label}")
            with st.expander("📊 거래량 보기", expanded=False):
                render_volume_chart(df)

    # ── 뉴스 리스트 ──────────────────────────────────────
    render_news_list(news_list)
