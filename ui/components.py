"""공용 UI 컴포넌트"""
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from utils.kst import now

_CSS_PATH = Path(__file__).parent / "styles.css"


def inject_styles():
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero():
    time_str = now().strftime("%H:%M KST")
    st.markdown(f"""
    <div class="hero-header">
        <h1>Stock News AI Analyzer</h1>
        <p>AI 기반 실시간 주식 뉴스 분석 &amp; 시장 동향 리포트</p>
        <div class="hero-badge">Powered by Gemini 2.5 (Pro→Flash 폴백) &middot; {time_str}</div>
    </div>
    """, unsafe_allow_html=True)


def render_summary(title: str, content: str, used_model: str = ""):
    st.markdown(f'<div class="section-title">&#129504; {title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
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
    st.markdown(
        f'<div class="section-title">&#128240; '
        f'수집된 뉴스 <span class="news-count-badge">{len(news_list)}건</span></div>',
        unsafe_allow_html=True,
    )
    for news in news_list:
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{news["title"]}</div>
            <div class="news-desc">{news["description"]}</div>
            <div class="news-meta">
                <span class="news-date">{news.get("pubDate", "")}</span>
                <a href="{news["link"]}" target="_blank">원문 보기 &#8594;</a>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_keyword_chips(keywords: list[str]):
    chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in keywords)
    st.markdown(f'<div class="keyword-chips">{chips}</div>', unsafe_allow_html=True)


def render_world_index_card(idx: dict):
    """세계 증시 컴팩트 카드 + 미니 차트"""
    name       = idx["name"]
    current    = idx["current"]
    change     = idx["change"]
    change_pct = idx["change_pct"]
    prices     = idx["chart_prices"]

    is_up = change >= 0
    color      = "#ef4444" if is_up else "#3b82f6"
    fill_color = "rgba(239,68,68,0.18)" if is_up else "rgba(59,130,246,0.18)"
    arrow      = "▲" if is_up else "▼"
    sign       = "+" if is_up else ""

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
        fig.update_yaxes(range=[ymin - margin, ymax + margin], visible=False)
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


def render_candle_chart(df, title: str = "", show_ma: tuple = (5, 20, 60)):
    """국내 종목 캔들차트 + 이동평균선"""
    if df is None or df.empty:
        st.info("차트 데이터가 없습니다.")
        return

    fig = go.Figure()

    # 캔들 (한국식: 상승 빨강 / 하락 파랑)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#ef4444",
        decreasing_line_color="#3b82f6",
        increasing_fillcolor="#ef4444",
        decreasing_fillcolor="#3b82f6",
        name="가격",
    ))

    # 이동평균선
    ma_colors = {5: "#fbbf24", 20: "#a78bfa", 60: "#34d399"}
    for window in show_ma:
        if len(df) >= window:
            ma = df["Close"].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ma,
                mode="lines",
                name=f"MA{window}",
                line=dict(color=ma_colors.get(window, "#94a3b8"), width=1.2),
                hovertemplate=f"MA{window}: %{{y:,.0f}}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14), x=0.02),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="rgba(255,255,255,0.05)",
            color="rgba(255,255,255,0.6)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            color="rgba(255,255,255,0.6)",
            tickformat=",",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="rgba(255,255,255,0.6)", size=10),
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_volume_chart(df):
    """거래량 막대 차트 (캔들 아래 작게)"""
    if df is None or df.empty:
        return

    colors = [
        "#ef4444" if c >= o else "#3b82f6"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=colors,
        opacity=0.6,
        hovertemplate="거래량: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="rgba(255,255,255,0.5)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="rgba(255,255,255,0.5)", tickformat=",.2s"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_sidebar(
    get_favorites_fn=None,
    on_fav_click=None,
    on_fav_remove=None,
    get_history_fn=None,
    on_history_click=None,
    on_history_clear=None,
):
    """사이드바 (즐겨찾기 + 검색 히스토리 + 가이드 + 면책)"""
    with st.sidebar:
        kst_now = now()

        st.markdown(f"""
        <div class="sidebar-model"><span>Gemini Pro → Flash → Lite → 2.0 (자동 폴백)</span></div>
        <div class="sidebar-time">{kst_now.strftime('%Y-%m-%d %H:%M:%S')} KST</div>
        """, unsafe_allow_html=True)

        # 즐겨찾기 섹션
        if get_favorites_fn is not None:
            favs = get_favorites_fn()
            st.markdown(
                '<div class="sidebar-card"><h4>&#11088; 관심 종목</h4></div>',
                unsafe_allow_html=True,
            )
            if not favs:
                st.caption("종목 검색 탭에서 ☆ 버튼으로 추가하세요")
            else:
                for fav in favs:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        if st.button(
                            f"⭐ {fav}",
                            key=f"fav_btn_{fav}",
                            use_container_width=True,
                        ):
                            if on_fav_click:
                                on_fav_click(fav)
                    with c2:
                        if st.button(
                            "✕",
                            key=f"fav_del_{fav}",
                            use_container_width=True,
                            help=f"{fav} 제거",
                        ):
                            if on_fav_remove:
                                on_fav_remove(fav)

        # 검색 히스토리 섹션
        if get_history_fn is not None:
            hist = get_history_fn()
            if hist:
                col_h1, col_h2 = st.columns([5, 1])
                with col_h1:
                    st.markdown(
                        '<span class="history-card-marker">'
                        '&#128338; 최근 분석'
                        '</span>',
                        unsafe_allow_html=True,
                    )
                with col_h2:
                    if st.button(
                        "🗑",
                        key="history_clear",
                        help="히스토리 비우기",
                        use_container_width=True,
                    ):
                        if on_history_clear:
                            on_history_clear()
                for h in hist:
                    if st.button(
                        f"🕐 {h}",
                        key=f"hist_btn_{h}",
                        use_container_width=True,
                    ):
                        if on_history_click:
                            on_history_click(h)

        st.markdown("""
        <div class="sidebar-card">
            <h4>&#128270; 종목 검색</h4>
            <p>종목명 입력 &rarr; 분석 클릭 &rarr; AI 요약 + 차트</p>
        </div>
        <div class="sidebar-card">
            <h4>&#128197; 시장 동향</h4>
            <p>국내/미국 시장 &rarr; AI가 핵심 이슈 정리</p>
        </div>
        <div class="sidebar-card">
            <h4>&#128293; 거래대금</h4>
            <p>KRX 거래대금 TOP 100 (5분 캐시)</p>
        </div>
        <div class="sidebar-card">
            <h4>&#127760; 세계 증시</h4>
            <p>미국·한국·아시아 주요 지수</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer-notice">
            &#9888; 본 서비스는 정보 제공 목적이며,<br>투자 권유가 아닙니다.
        </div>
        """, unsafe_allow_html=True)
