"""TAB 4: 세계 증시"""
import streamlit as st

from api.market import fetch_world_indices
from ui.components import render_world_index_card


_US_NAMES = {"다우산업", "나스닥종합", "S&P 500"}
_KR_NAMES = {"코스피", "코스닥"}
_ASIA_NAMES = {"닛케이225", "상해종합", "항셍"}


def _render_grid(items: list, cols: int = 3):
    for i in range(0, len(items), cols):
        row = items[i:i + cols]
        col_objs = st.columns(cols)
        for j, item in enumerate(row):
            with col_objs[j]:
                render_world_index_card(item)


def render():
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
        if st.button("🔄 새로고침", use_container_width=True, key="world_refresh_btn"):
            fetch_world_indices.clear()

    with st.spinner("세계 증시 데이터 수집 중..."):
        world_data = fetch_world_indices()

    if not world_data:
        st.error("데이터를 가져오지 못했습니다.")
        return

    us_group   = [d for d in world_data if d["name"] in _US_NAMES]
    kr_group   = [d for d in world_data if d["name"] in _KR_NAMES]
    asia_group = [d for d in world_data if d["name"] in _ASIA_NAMES]

    show_all = region_filter == "🌐 전체"

    if (show_all or "미국" in region_filter) and us_group:
        st.markdown(
            '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">'
            '&#127482;&#127480; 미국 증시</div>',
            unsafe_allow_html=True,
        )
        _render_grid(us_group, cols=3)

    if (show_all or "한국" in region_filter) and kr_group:
        st.markdown(
            '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">'
            '&#127472;&#127479; 한국 증시</div>',
            unsafe_allow_html=True,
        )
        _render_grid(kr_group, cols=2)

    if (show_all or "아시아" in region_filter) and asia_group:
        st.markdown(
            '<div class="section-title" style="font-size:1rem; margin:1rem 0 0.3rem 0;">'
            '&#127759; 아시아 증시</div>',
            unsafe_allow_html=True,
        )
        _render_grid(asia_group, cols=3)
