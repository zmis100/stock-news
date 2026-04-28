"""TAB 5: 관심종목 가격 알림 + 일일 요약 (Discord 전용)"""
import streamlit as st

from notifications import discord
from notifications.checker import run_check, send_daily_summary
from storage import alerts, favorites


def render():
    st.markdown(
        '<div class="section-title">&#128276; Discord 알림</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:-0.5rem;">'
        '관심종목 임계값 알림 + 매일 15:20 / 19:50 일일 요약 자동 발송<br>'
        '<span style="color:rgba(255,255,255,0.4); font-size:0.75rem;">'
        '⚠️ 알림 룰은 서버 공용 저장 (즐겨찾기/히스토리만 브라우저별 분리). '
        '자동 발송은 사용자 본인 PC에서만 동작 (Cloud 배포 시 X)'
        '</span></p>',
        unsafe_allow_html=True,
    )

    cfg = alerts.load()

    # ── 채널 ON/OFF ────────────────────────────────────
    st.markdown('<div class="section-title">&#9881; 알림 채널</div>',
                unsafe_allow_html=True)

    enabled = cfg.get("channel", "off") == "discord"
    new_enabled = st.toggle(
        "💬 Discord 알림 활성화",
        value=enabled,
        key="alert_discord_toggle",
    )
    if new_enabled != enabled:
        alerts.set_channel("discord" if new_enabled else "off")
        st.toast(f"Discord 알림 {'ON' if new_enabled else 'OFF'}")
        st.rerun()

    if not new_enabled:
        st.caption("알림이 꺼져있습니다. 토글을 켜면 발송이 활성화됩니다.")
        return

    st.caption(
        "💡 Webhook URL은 `.streamlit/secrets.toml` 의 `DISCORD_WEBHOOK_URL` 에 저장됩니다."
    )

    # ── 테스트 발송 ─────────────────────────────────────
    if st.button("📤 테스트 메시지 발송", key="alert_test_btn"):
        ok, info = discord.send("🧪 Stock News AI Analyzer — 테스트 메시지")
        if ok:
            st.success(f"✅ {info}")
        else:
            st.error(f"❌ {info}")

    # ── 일일 요약 (즉시 발송) ──────────────────────────
    st.markdown('<div class="section-title">&#128202; 일일 요약 (수동 발송)</div>',
                unsafe_allow_html=True)
    st.caption(
        "거래대금 TOP100 중 등락 ±10% 이상 종목 + 외국인/기관 + AI 코멘트를 Discord로 보냅니다. "
        "**자동 발송은 작업 스케줄러 등록이 필요합니다 (아래 안내).**"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎯 정규장 종가 요약 보내기", use_container_width=True,
                     key="send_summary_regular"):
            with st.spinner("거래대금 + 외인/기관 + AI 분석 중 (1~2분 소요)..."):
                ok, info = send_daily_summary("regular")
            (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {info}")
    with c2:
        if st.button("🌙 NXT 종가 요약 보내기", use_container_width=True,
                     key="send_summary_nxt"):
            with st.spinner("거래대금 + 외인/기관 + AI 분석 중 (1~2분 소요)..."):
                ok, info = send_daily_summary("nxt")
            (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {info}")

    # ── 자동 스케줄 안내 ────────────────────────────────
    with st.expander("⏰ 자동 발송 설정하기 (Windows 작업 스케줄러)", expanded=False):
        st.markdown("""
**1. Windows 작업 스케줄러 열기** (`Win + S` → "작업 스케줄러")

**2. "기본 작업 만들기"** 클릭 → 두 개의 작업 등록:

| 작업 이름 | 트리거 | 동작 |
|---|---|---|
| `주식 정규장 종가 알림` | 매주 평일 **15:20** | python.exe + 인수 `-m notifications.runner --daily-summary regular` |
| `주식 NXT 종가 알림` | 매주 평일 **19:50** | python.exe + 인수 `-m notifications.runner --daily-summary nxt` |

**3. 동작 설정 예시:**
- **프로그램/스크립트**: `C:\\Users\\DKSYSTEMS\\Desktop\\stock-news-summarizer\\venv\\Scripts\\python.exe`
- **인수 추가**: `-m notifications.runner --daily-summary regular`
- **시작 위치**: `C:\\Users\\DKSYSTEMS\\Desktop\\stock-news-summarizer`

**4. 추가 옵션 (속성):**
- "사용자가 로그온할 때만 실행" ← 기본
- "AC 전원 사용 중일 때만 시작" 체크 해제하면 노트북 배터리 모드에서도 동작
- "예약된 시간에 작업을 시작 못 한 경우 가능한 한 빨리 시작" 체크
        """)

    # ── 관심종목 임계값 룰 ─────────────────────────────
    st.markdown('<div class="section-title">&#128276; 관심종목 임계값 알림 (룰)</div>',
                unsafe_allow_html=True)
    st.caption("등록한 종목이 임계값(±%) 이상 움직이면 즉시 알림. "
               "수동 점검 또는 별도 cron 필요.")

    fav_list = favorites.load()
    candidates = ["직접 입력"] + fav_list

    with st.form("alert_add_form", border=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            choice = st.selectbox("종목", options=candidates, key="alert_pick")
            if choice == "직접 입력":
                name = st.text_input(
                    "종목명",
                    placeholder="삼성전자",
                    label_visibility="collapsed",
                    key="alert_name_input",
                )
            else:
                name = choice
        with c2:
            threshold = st.number_input(
                "임계값 (±%)",
                min_value=1.0, max_value=30.0, value=5.0, step=0.5,
                key="alert_threshold",
            )
        with c3:
            if st.form_submit_button("등록", use_container_width=True):
                if name and name.strip():
                    alerts.add_rule(name.strip(), threshold)
                    st.toast(f"✅ '{name}' 등록 (±{threshold:.1f}%)")
                    st.rerun()

    rules = cfg.get("rules", [])
    if not rules:
        st.caption("등록된 룰이 없습니다.")
    else:
        for rule in rules:
            c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
            with c1:
                st.markdown(f"**{rule['name']}**")
            with c2:
                st.markdown(f"±{rule['threshold_pct']:.1f}%")
            with c3:
                last = rule.get("last_alerted_at")
                last_str = last[:16].replace("T", " ") if last else "—"
                st.caption(f"마지막: {last_str}")
            with c4:
                if st.button("🗑", key=f"alert_del_{rule['name']}"):
                    alerts.remove_rule(rule["name"])
                    st.rerun()

    if st.button("🚀 룰 즉시 점검", use_container_width=True, key="alert_check_btn"):
        with st.spinner("점검 중..."):
            sent, skipped, lines = run_check()
        st.success(f"전송 {sent}건 / 스킵 {skipped}건")
        for line in lines[:20]:
            st.caption(line)
