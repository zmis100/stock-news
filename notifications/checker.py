"""관심종목 가격 변동 체크 + 일일 요약 발송 (Discord 전용)."""
import logging

from api.naver import (
    fetch_trading_volume_top,
    fetch_foreign_institution_bulk,
    fetch_news_per_stock,
)
from api.gemini import analyze_movers_reasons
from notifications import discord
from storage import alerts

log = logging.getLogger(__name__)

_AI_THRESHOLD_PCT = 10.0
_MAX_DISCORD_LEN = 1900


# ── 룰 기반 알림 (관심종목 임계값) ─────────────────────────

def _format_rule_message(rule: dict, stock: dict) -> str:
    chg = stock["change_ratio"]
    sign = "+" if chg > 0 else ""
    return (
        f"📢 **[{rule['name']}]** {sign}{chg:.2f}% (임계값 ±{rule['threshold_pct']:.1f}%)\n"
        f"종가 {stock['close']:,}원 · 거래대금 {stock['amount']/1e8:,.0f}억"
    )


def run_check() -> tuple[int, int, list[str]]:
    """등록 룰 점검. (전송수, 스킵수, 메시지로그) 반환."""
    cfg = alerts.load()
    if cfg.get("channel", "off") != "discord":
        return 0, 0, ["Discord 채널 꺼짐"]

    rules = cfg.get("rules", [])
    if not rules:
        return 0, 0, []

    top_list, _ = fetch_trading_volume_top(100)
    name_to_stock = {s["name"]: s for s in top_list}

    sent, skipped = 0, 0
    log_lines: list[str] = []

    for rule in rules:
        s = name_to_stock.get(rule["name"])
        if not s:
            skipped += 1
            log_lines.append(f"⚪ {rule['name']}: TOP100에 없음")
            continue

        if abs(s["change_ratio"]) < rule["threshold_pct"]:
            skipped += 1
            continue

        if s["close"] == rule.get("last_close", 0):
            skipped += 1
            log_lines.append(f"⚪ {rule['name']}: 동일 종가 중복")
            continue

        ok, info = discord.send(_format_rule_message(rule, s))
        if ok:
            alerts.mark_alerted(rule["name"], s["close"])
            sent += 1
            log_lines.append(f"✅ {rule['name']}: 전송")
        else:
            log_lines.append(f"❌ {rule['name']}: {info}")

    return sent, skipped, log_lines


# ── 일일 요약 (15:20 / 19:50 자동 발송) ────────────────────

_SESSION_LABEL = {
    "regular": "🎯 정규장 종가 임박 (15:20)",
    "nxt":     "🎯 NXT 종가 임박 (19:50)",
}


def _format_stock_block(s: dict) -> str:
    chg = s["change_ratio"]
    sign = "+" if chg > 0 else ""
    amt = s["amount"] / 1e8

    fi = s.get("foreign_institution") or {}
    fi_close = fi.get("close", s["close"])
    foreign_eok = fi.get("foreign_qty", 0) * fi_close / 1e8
    inst_eok = fi.get("inst_qty", 0) * fi_close / 1e8

    fi_line = ""
    if fi:
        f_sign = "+" if foreign_eok > 0 else ""
        i_sign = "+" if inst_eok > 0 else ""
        fi_line = f"외인 {f_sign}{foreign_eok:,.0f}억 · 기관 {i_sign}{inst_eok:,.0f}억\n"

    block = (
        f"**{s['name']}** ({s['code']}) `{sign}{chg:.2f}%`\n"
        f"거래대금 {amt:,.0f}억 · 종가 {s['close']:,}원\n"
        f"{fi_line}"
    )
    if s.get("ai_reason"):
        block += f"🤖 {s['ai_reason']}\n"
    return block + "\n"


def _split_into_chunks(header: str, blocks: list[str]) -> list[str]:
    """블록들을 Discord 1900자 한도에 맞게 청크로 분할."""
    chunks: list[str] = []
    current = header
    for b in blocks:
        if len(current) + len(b) > _MAX_DISCORD_LEN:
            chunks.append(current.rstrip())
            current = ""  # 두 번째 청크부터는 헤더 없음
        current += b
    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def send_daily_summary(session_kind: str = "regular") -> tuple[bool, str]:
    """거래대금 TOP100 + 외국인/기관 + AI 코멘트(±10%)를 Discord 일일 요약으로 발송."""
    cfg = alerts.load()
    if cfg.get("channel", "off") != "discord":
        return False, "Discord 채널이 꺼져있습니다."

    label = _SESSION_LABEL.get(session_kind, "거래대금 자동 요약")

    # 1. 거래대금
    top_list, err = fetch_trading_volume_top(100)
    if not top_list:
        return False, f"거래대금 조회 실패: {err}"

    # 2. 외국인/기관 (병렬)
    codes = [s["code"] for s in top_list if s.get("code")]
    fi_map = fetch_foreign_institution_bulk(codes)
    for s in top_list:
        s["foreign_institution"] = fi_map.get(s.get("code", ""))

    # 3. ±10% 종목만 골라 AI 분석
    targets = sorted(
        [s for s in top_list if abs(s["change_ratio"]) >= _AI_THRESHOLD_PCT],
        key=lambda x: abs(x["change_ratio"]),
        reverse=True,
    )

    if not targets:
        msg = (
            f"{label}\n\n"
            f"오늘은 거래대금 TOP100 중 등락 ±{_AI_THRESHOLD_PCT:.0f}% 이상 종목이 없습니다."
        )
        ok, info = discord.send(msg)
        return ok, info

    # 4. 종목별 뉴스 + AI 사유
    news_map = fetch_news_per_stock(targets, display=10, priority_keyword="특징주")
    stocks_tuple = tuple(
        (s["name"], s["code"], s["change_ratio"], s["amount"] / 1e8)
        for s in targets
    )
    news_tuple = tuple(
        (name, tuple((n["title"], n["description"]) for n in items))
        for name, items in news_map.items()
    )
    reasons_map, model = analyze_movers_reasons(stocks_tuple, news_tuple)
    for s in targets:
        s["ai_reason"] = reasons_map.get(s.get("code", ""), "")

    # 5. 청크로 나눠 발송
    header = (
        f"{label}\n"
        f"거래대금 TOP100 중 등락 ±{_AI_THRESHOLD_PCT:.0f}% 이상 "
        f"**{len(targets)}종목** · AI 모델: `{model}`\n\n"
    )
    blocks = [_format_stock_block(s) for s in targets]
    chunks = _split_into_chunks(header, blocks)

    sent = 0
    fail_info = ""
    for chunk in chunks:
        ok, info = discord.send(chunk)
        if ok:
            sent += 1
        else:
            fail_info = info

    if sent > 0:
        return True, f"청크 {sent}/{len(chunks)} 전송 완료"
    return False, f"전송 실패: {fail_info}"
