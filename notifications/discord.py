"""Discord webhook 알림 채널."""
import logging

import requests

from config import get_secret

log = logging.getLogger(__name__)


def send(message: str, webhook_url: str | None = None) -> tuple[bool, str]:
    """Discord 채널에 메시지 전송. webhook_url 미지정 시 secrets/환경변수 사용."""
    url = webhook_url or get_secret("DISCORD_WEBHOOK_URL")
    if not url:
        return False, "DISCORD_WEBHOOK_URL 미설정"

    try:
        r = requests.post(url, json={"content": message[:1900]}, timeout=10)
        if 200 <= r.status_code < 300:
            return True, "전송 완료"
        log.warning("Discord webhook %s: %s", r.status_code, r.text[:200])
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        log.exception("Discord webhook failed")
        return False, str(e)
