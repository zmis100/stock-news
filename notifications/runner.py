"""백그라운드 알림 러너.

사용법:
    # 등록 룰 점검 (즐겨찾기 임계값 알림) — 1회 또는 루프
    python -m notifications.runner
    python -m notifications.runner --loop 300

    # 일일 요약 발송 (정규장 종가 임박 / NXT 종가 임박)
    python -m notifications.runner --daily-summary regular   # 15:20에 실행
    python -m notifications.runner --daily-summary nxt       # 19:50에 실행

Windows 작업 스케줄러 등록 예시:
    [매일 15:20]
        프로그램: <venv>\\Scripts\\python.exe
        인수:     -m notifications.runner --daily-summary regular
        시작 위치: <project root>
        조건:     평일만, 사용자 로그인 시 동작

    [매일 19:50]
        같음, 인수: -m notifications.runner --daily-summary nxt

macOS/Linux crontab:
    20 15 * * 1-5 cd /path/to/project && venv/bin/python -m notifications.runner --daily-summary regular
    50 19 * * 1-5 cd /path/to/project && venv/bin/python -m notifications.runner --daily-summary nxt
"""
import argparse
import sys
import time
from pathlib import Path

# venv import path 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifications.checker import run_check, send_daily_summary  # noqa: E402
from utils.logging import setup_logging  # noqa: E402


def _run_rule_check_loop(interval: int):
    print(f"[runner] 룰 점검 루프 시작 (주기 {interval}초)")
    while True:
        try:
            sent, skipped, _ = run_check()
            print(f"[tick] sent={sent} skipped={skipped}")
        except Exception as e:
            print(f"[error] {e}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--daily-summary",
        choices=["regular", "nxt"],
        help="일일 요약 발송 (regular=정규장 15:20용, nxt=NXT 19:50용)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="룰 점검 반복 주기(초). 0이면 1회만 실행",
    )
    args = parser.parse_args()
    setup_logging()

    # 일일 요약 모드
    if args.daily_summary:
        ok, info = send_daily_summary(args.daily_summary)
        status = "✅" if ok else "❌"
        print(f"{status} [daily-summary {args.daily_summary}] {info}")
        sys.exit(0 if ok else 1)

    # 룰 점검 모드
    if args.loop > 0:
        _run_rule_check_loop(args.loop)
    else:
        sent, skipped, lines = run_check()
        print(f"[done] sent={sent} skipped={skipped}")
        for line in lines:
            print(f"  {line}")


if __name__ == "__main__":
    main()
