"""TAB 3: 거래대금 TOP 100 (외국인/기관 + AI 상승 사유 + 종가베팅 자동 패널)"""
import streamlit as st

from api.naver import (
    fetch_trading_volume_top,
    fetch_foreign_institution_bulk,
    fetch_news_per_stock,
)
from api.gemini import analyze_movers_reasons, analyze_closing_bet
from utils.kst import today_str, is_market_open, closing_bet_session, now


_SORT_OPTIONS = {
    "거래대금 높은순": ("amount", True),
    "거래대금 낮은순": ("amount", False),
    "등락률 높은순":   ("change_ratio", True),
    "등락률 낮은순":   ("change_ratio", False),
}

# AI 분석 대상: 등락률 절댓값 N% 이상 (위/아래 둘 다, 개수 무제한)
_AI_THRESHOLD_PCT = 10.0


def _format_money(qty: int, price: int) -> str:
    if qty == 0 or price == 0:
        return "—"
    won = qty * price
    eok = won / 1e8
    sign = "+" if won > 0 else ""
    return f"{sign}{eok:,.0f}억"


def _format_marcap(marcap: int) -> str:
    if not marcap or marcap <= 0:
        return "—"
    if marcap >= 1e13:
        return f"{marcap / 1e12:,.0f}조"
    if marcap >= 1e12:
        return f"{marcap / 1e12:,.1f}조"
    return f"{marcap / 1e8:,.0f}억"


def _flow_color(qty: int) -> str:
    """외국인/기관 수급 색상 — 등락률(빨/파)과 구분되도록 매수=초록 / 매도=주황."""
    if qty > 0:
        return "#22c55e"   # 매수 = 초록
    if qty < 0:
        return "#f97316"   # 매도 = 주황
    return "rgba(255,255,255,0.4)"


def _render_row(item: dict):
    amt_billions = item["amount"] / 1e8
    cap_str = _format_marcap(item["marcap"])
    chg = item["change_ratio"]
    chg_color = "#ef4444" if chg > 0 else "#3b82f6" if chg < 0 else "rgba(255,255,255,0.5)"
    chg_sign = "+" if chg > 0 else ""

    fi = item.get("foreign_institution") or {}
    foreign_qty = fi.get("foreign_qty", 0)
    inst_qty = fi.get("inst_qty", 0)
    fi_close = fi.get("close", item["close"])
    fi_date = fi.get("date", "")
    fi_date_short = fi_date[5:].replace(".", "/") if fi_date else ""

    foreign_html = (
        f'<div style="text-align:right;">'
        f'<div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">외국인'
        + (f' <span style="color:rgba(255,255,255,0.3);">({fi_date_short})</span>' if fi_date_short else '')
        + f'</div>'
        f'<div style="color:{_flow_color(foreign_qty)}; font-weight:600; font-size:0.85rem;">'
        f'{_format_money(foreign_qty, fi_close)}</div>'
        f'</div>'
    )
    inst_html = (
        f'<div style="text-align:right;">'
        f'<div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">기관'
        + (f' <span style="color:rgba(255,255,255,0.3);">({fi_date_short})</span>' if fi_date_short else '')
        + f'</div>'
        f'<div style="color:{_flow_color(inst_qty)}; font-weight:600; font-size:0.85rem;">'
        f'{_format_money(inst_qty, fi_close)}</div>'
        f'</div>'
    )

    # AI 상승/하락 사유 코멘트 (상승=빨강 강조선, 하락=파랑 — 등락률 색과 일치)
    ai_comment = item.get("ai_reason", "")
    comment_block = ""
    if ai_comment:
        accent = "#ef4444" if chg > 0 else "#3b82f6" if chg < 0 else "#a78bfa"
        comment_block = (
            '<div style="margin-top:0.5rem; padding:0.5rem 0.75rem; '
            'background:rgba(167,139,250,0.06); '
            f'border-left:3px solid {accent}; '
            'border-radius:6px; color:rgba(255,255,255,0.85); '
            'font-size:0.82rem; line-height:1.45;">'
            f'🤖 {ai_comment}'
            '</div>'
        )

    st.markdown(f"""
    <div class="news-card" style="display:flex; flex-direction:column; gap:0.3rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem;">
            <div style="flex:1; min-width:160px;">
                <span style="color:#667eea; font-weight:700; font-size:1.1rem; margin-right:0.5rem;">{item.get("display_rank", "")}</span>
                <span class="news-title" style="display:inline; font-size:1rem;">{item["name"]}</span>
                <span style="color:rgba(255,255,255,0.3); font-size:0.75rem; margin-left:0.4rem;">{item["code"]} &middot; {item["market"]}</span>
            </div>
            <div style="display:flex; gap:1.2rem; flex-wrap:wrap; align-items:center;">
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
                {foreign_html}
                {inst_html}
                <div style="text-align:right;">
                    <div style="color:rgba(255,255,255,0.4); font-size:0.7rem;">시총</div>
                    <div style="color:rgba(255,255,255,0.6); font-size:0.85rem;">{cap_str}</div>
                </div>
            </div>
        </div>
        {comment_block}
    </div>
    """, unsafe_allow_html=True)


def _render_closing_bet_panel():
    """정규장(15:00~15:20) / NXT(19:30~19:50) 종가베팅 시간대에만 자동 표시."""
    session = closing_bet_session()
    if session is None:
        return

    session_kind, session_label = session
    accent = "#ef4444" if session_kind == "regular" else "#f59e0b"
    bg = "rgba(239,68,68,0.08)" if session_kind == "regular" else "rgba(245,158,11,0.08)"
    border = "rgba(239,68,68,0.3)" if session_kind == "regular" else "rgba(245,158,11,0.3)"

    st.markdown(
        f'<div style="background:{bg}; border:1px solid {border}; '
        f'border-left:4px solid {accent}; '
        f'border-radius:10px; padding:0.9rem 1rem; margin:0.5rem 0; '
        f'color:rgba(255,255,255,0.95); font-size:0.9rem; line-height:1.6;">'
        f'🎯 <strong>종가베팅 시간대 — {session_label}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1분 자동 새로고침 토글
    col_auto, col_info = st.columns([2, 3])
    with col_auto:
        auto = st.toggle("⚡ 1분 자동 새로고침", value=False, key="bet_auto_refresh")
    with col_info:
        if auto:
            try:
                from streamlit_autorefresh import st_autorefresh
                cnt = st_autorefresh(interval=60_000, key="bet_autorefresh_counter")
                st.caption(f"🔄 #{cnt}회 갱신 · {now().strftime('%H:%M:%S')} KST")
            except ImportError:
                st.caption("⚠️ streamlit-autorefresh 미설치")

    # 거래대금 데이터 필요
    top_list = st.session_state.get("vol_data") or []
    targets = [s for s in top_list if abs(s.get("change_ratio", 0)) >= 10.0]

    if not top_list:
        st.info("먼저 아래 [거래대금 TOP 100 조회]를 누르면 종가베팅 종합 분석이 가능합니다.")
        return

    if not targets:
        st.warning("등락률 ±10% 이상인 종목이 없어 종가베팅 종합 분석을 생략합니다.")
        return

    st.markdown(
        f'<div style="color:rgba(255,255,255,0.7); font-size:0.85rem;">'
        f'분석 대상: 거래대금 TOP100 중 등락률 ±10% 이상 <strong>{len(targets)}개</strong> 종목'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.button("🚀 종가베팅 종합 분석 실행", use_container_width=True, key="bet_run_btn"):
        return

    with st.status("종가베팅 종합 분석 중...", expanded=True) as status:
        status.update(label="① 종목별 뉴스 수집 (특징주 우선)...", state="running")
        news_map = fetch_news_per_stock(targets, display=10, priority_keyword="특징주")
        st.write(f"✅ {len(news_map)}개 종목 뉴스 수집")

        status.update(label="② Gemini 종가베팅 분석 중...", state="running")
        stocks_tuple = tuple(
            (s["name"], s["code"], s["change_ratio"], s["amount"] / 1e8)
            for s in targets
        )
        news_tuple = tuple(
            (name, tuple((n["title"], n["description"]) for n in items))
            for name, items in news_map.items()
        )
        bet_summary, bet_model = analyze_closing_bet(
            stocks_tuple, news_tuple, session_label=session_label,
        )
        st.write(f"✅ AI 분석 완료 ({bet_model})")
        status.update(label="완료!", state="complete")

    # 결과 렌더 (간단한 카드)
    st.markdown(
        '<div class="summary-card" style="margin-top:0.5rem;">',
        unsafe_allow_html=True,
    )
    st.markdown("### 🔥 종가베팅 AI 종합 리포트")
    st.markdown(bet_summary)
    if bet_model not in ("N/A", "FAILED"):
        st.markdown(
            f'<div style="margin-top:0.8rem; padding-top:0.6rem; '
            f'border-top:1px solid rgba(255,255,255,0.08); '
            f'color:rgba(255,255,255,0.45); font-size:0.75rem; text-align:right;">'
            f'분석 모델: <strong style="color:#a78bfa;">{bet_model}</strong></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render():
    st.markdown(
        f'<div class="section-title">&#128293; {today_str()} 거래대금 상위 종목</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
        'KRX 거래대금 + 외국인/기관 순매수 + 🤖 AI 상승/하락 사유 분석</p>',
        unsafe_allow_html=True,
    )

    if is_market_open():
        st.markdown(
            '<div style="background:rgba(34,197,94,0.08); '
            'border:1px solid rgba(34,197,94,0.25); '
            'border-radius:10px; padding:0.7rem 1rem; '
            'color:rgba(255,255,255,0.85); font-size:0.85rem; line-height:1.6;">'
            '🟢 <strong>장 운영 시간 (09:00 - 15:30)</strong><br>'
            '· 거래대금: 장중 누적 (5분 캐시)<br>'
            '· 외국인/기관: <strong>전일 마감 확정치</strong> '
            '(당일 장중 수급은 무료로 제공되지 않음. 장 마감 30분~1시간 후 당일 확정치로 갱신)'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(168,85,247,0.08); '
            'border:1px solid rgba(168,85,247,0.25); '
            'border-radius:10px; padding:0.7rem 1rem; '
            'color:rgba(255,255,255,0.85); font-size:0.85rem; line-height:1.6;">'
            '🌙 <strong>장 마감 시간</strong><br>'
            '· 거래대금: 당일 최종 마감 데이터<br>'
            '· 외국인/기관: <strong>당일 마감 확정치</strong> '
            '(마감 후 30분~1시간 사이엔 직전 거래일 데이터일 수 있음)<br>'
            '<span style="color:rgba(255,255,255,0.5); font-size:0.78rem;">'
            '↳ 카드의 날짜(예: 외국인 (4/27))로 어느 일자 데이터인지 확인 가능'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── 종가베팅 자동 패널 (정규 15:00~15:20 / NXT 19:30~19:50) ──
    _render_closing_bet_panel()

    if st.button("거래대금 TOP 100 조회", use_container_width=True, key="vol_btn"):
        with st.status("KRX 거래대금 + 외국인/기관 + AI 분석 중...", expanded=True) as status:
            # 1단계: 거래대금
            status.update(label="① KRX 거래대금 수집...", state="running")
            top_list, err_msg = fetch_trading_volume_top(100)
            if not top_list:
                status.update(label="조회 실패", state="error")
                st.error("거래대금 데이터를 가져오지 못했습니다.")
                if err_msg:
                    st.code(f"오류: {err_msg}", language="text")
                return
            st.write(f"✅ {len(top_list)}개 종목 거래대금 수집")

            # 2단계: 외국인/기관 (병렬)
            status.update(label="② 외국인/기관 순매수 수집 (병렬)...", state="running")
            codes = [s["code"] for s in top_list if s.get("code")]
            fi_map = fetch_foreign_institution_bulk(codes)
            for s in top_list:
                s["foreign_institution"] = fi_map.get(s.get("code", ""))
            with_fi = sum(1 for s in top_list if s.get("foreign_institution"))
            st.write(f"✅ 외국인/기관 데이터 {with_fi}/{len(top_list)}종목")

            # 3단계: 등락률 절댓값 ≥ 10% 종목 전부 → 뉴스 + AI 분석
            ai_targets = sorted(
                [s for s in top_list if abs(s["change_ratio"]) >= _AI_THRESHOLD_PCT],
                key=lambda x: abs(x["change_ratio"]),
                reverse=True,
            )

            if ai_targets:
                status.update(
                    label=f"③ AI 분석 대상 {len(ai_targets)}종목 뉴스 수집 (특징주 키워드 우선)...",
                    state="running",
                )
                # display=10으로 더 많이 받고 '특징주' 키워드 들어간 뉴스를 상단으로 정렬
                news_map = fetch_news_per_stock(
                    ai_targets, display=10, priority_keyword="특징주"
                )
                st.write(f"✅ 뉴스 수집 {len(news_map)}/{len(ai_targets)}종목")

                status.update(label="④ Gemini AI 상승/하락 사유 분석 중...", state="running")
                stocks_tuple = tuple(
                    (s["name"], s["code"], s["change_ratio"], s["amount"] / 1e8)
                    for s in ai_targets
                )
                news_tuple = tuple(
                    (name, tuple((n["title"], n["description"]) for n in items))
                    for name, items in news_map.items()
                )
                reasons_map, used_model = analyze_movers_reasons(stocks_tuple, news_tuple)

                for s in top_list:
                    s["ai_reason"] = reasons_map.get(s.get("code", ""), "")

                if reasons_map:
                    st.write(f"✅ AI 분석 완료: {len(reasons_map)}종목 코멘트 ({used_model})")
                else:
                    st.write(f"⚠️ AI 응답 파싱 실패 ({used_model})")
            else:
                st.write("ℹ️ 등락률 ±3% 이상 종목 없음 → AI 분석 생략")

            status.update(label="조회 완료!", state="complete")
            st.session_state["vol_data"] = top_list

    if not st.session_state.get("vol_data"):
        return

    top_list = st.session_state["vol_data"]

    # 정렬
    selected_sort = st.selectbox(
        "정렬 기준",
        options=list(_SORT_OPTIONS.keys()),
        key="vol_sort",
    )
    sort_key, sort_desc = _SORT_OPTIONS[selected_sort]
    sorted_list = sorted(top_list, key=lambda x: x[sort_key], reverse=sort_desc)
    for i, item in enumerate(sorted_list, 1):
        item["display_rank"] = i

    # AI 분석 통계
    ai_count = sum(1 for s in sorted_list if s.get("ai_reason"))
    st.markdown(
        f'<div class="section-title">&#127942; {selected_sort} TOP {len(sorted_list)}'
        + (f' <span class="news-count-badge">🤖 AI 코멘트 {ai_count}개</span>' if ai_count else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    for item in sorted_list:
        _render_row(item)
