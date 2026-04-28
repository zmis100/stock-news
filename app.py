"""Stock News AI Analyzer — entry point."""
import streamlit as st

from config import api_status
from storage import favorites, history
from ui.components import inject_styles, render_hero, render_sidebar
from ui.tabs import (
    stock_search,
    market_trend,
    trading_volume,
    world_market,
    alerts_settings,
)
from utils.logging import setup_logging


def _on_favorite_click(name: str):
    st.session_state["stock_prefill"] = name
    st.session_state["stock_auto_search"] = True
    st.rerun()


def _on_favorite_remove(name: str):
    favorites.remove(name)
    st.toast(f"'{name}' 관심종목에서 제거됨")
    st.rerun()


def _on_history_click(name: str):
    st.session_state["stock_prefill"] = name
    st.session_state["stock_auto_search"] = True
    st.rerun()


def _on_history_clear():
    history.clear()
    st.toast("히스토리 삭제됨")
    st.rerun()


def main():
    setup_logging()
    st.set_page_config(
        page_title="Stock News AI Analyzer",
        page_icon="📈",
        layout="centered",
    )

    inject_styles()
    render_hero()

    api_ok, missing = api_status()
    if not api_ok:
        for key in missing:
            st.warning(f"{key} API 키가 설정되지 않았습니다.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  종목 검색  ",
        "  시장 동향  ",
        "  거래대금 / 종가베팅  ",
        "  세계 증시  ",
        "  🔔 알림  ",
    ])

    with tab1:
        stock_search.render(api_ok)
    with tab2:
        market_trend.render(api_ok)
    with tab3:
        trading_volume.render()
    with tab4:
        world_market.render()
    with tab5:
        alerts_settings.render()

    render_sidebar(
        get_favorites_fn=favorites.load,
        on_fav_click=_on_favorite_click,
        on_fav_remove=_on_favorite_remove,
        get_history_fn=history.load,
        on_history_click=_on_history_click,
        on_history_clear=_on_history_clear,
    )


if __name__ == "__main__":
    main()
