"""한국 시간(KST) 유틸"""
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def now():
    return datetime.now(KST)


def today():
    return now().date()


def today_str():
    return today().strftime("%Y년 %m월 %d일")


def is_market_open() -> bool:
    """KRX 장 운영 시간 (평일 09:00~15:30)"""
    n = now()
    if n.weekday() >= 5:
        return False
    open_t = n.replace(hour=9, minute=0, second=0, microsecond=0)
    close_t = n.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= n <= close_t


def closing_bet_session() -> tuple[str, str] | None:
    """현재 종가베팅 시간대 반환.

    - 정규장 종가 임박: 평일 15:00~15:20 → ('regular', '정규장 종가 임박 (15:20 마감)')
    - NXT 종가 임박  : 평일 19:30~19:50 → ('nxt', 'NXT 종가 임박 (19:50 마감)')
    - 그 외           : None
    """
    n = now()
    if n.weekday() >= 5:
        return None
    today_at = lambda h, m: n.replace(hour=h, minute=m, second=0, microsecond=0)

    if today_at(15, 0) <= n <= today_at(15, 20):
        return ("regular", "정규장 종가 임박 (15:20 마감)")
    if today_at(19, 30) <= n <= today_at(19, 50):
        return ("nxt", "NXT 종가 임박 (19:50 마감)")
    return None
