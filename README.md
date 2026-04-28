# Stock News AI Analyzer

AI 기반 한국 주식 뉴스 분석 + 시장 동향 + 거래대금/종가베팅 대시보드.

## 주요 기능

- **종목 검색**: 종목명 입력 → 네이버 뉴스 + Gemini AI 요약 + 캔들/주봉/월봉/년봉 차트 + 즐겨찾기
- **시장 동향**: 국내/미국 증시 오늘의 핵심 이슈 AI 요약
- **거래대금 / 종가베팅**:
  - KRX 거래대금 TOP 100 + 외국인/기관 순매수
  - 등락률 ±10% 이상 종목 자동 AI 코멘트 (특징주 키워드 우선)
  - **정규장 종가 임박 (15:00~15:20) / NXT 종가 임박 (19:30~19:50)** 자동 종합 리포트
- **세계 증시**: 미국·한국·아시아 주요 지수
- **사이드바**: 관심종목 ⭐ + 최근 분석 히스토리 🕐

## 설치

```bash
git clone https://github.com/zmis100/stock-news.git
cd stock-news

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## API 키 발급

`.env.example`을 `.env`로 복사하고 값을 채워주세요.

| 키 | 발급처 | 필수 |
|---|---|---|
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | [네이버 개발자센터](https://developers.naver.com) → 애플리케이션 등록 → 검색 API | ✅ |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) | ✅ |
| `DISCORD_WEBHOOK_URL` | Discord 채널 → 연동 → 웹후크 → URL 복사 (알림 기능 사용 시) | ⬜ |

## 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 http://localhost:8501 열립니다.

## 디렉토리 구조

```
.
├── app.py                  # Streamlit 진입점
├── config/                 # 환경변수/시크릿 로딩
├── utils/                  # KST·텍스트·로깅 유틸
├── api/                    # 외부 API (네이버/Gemini/yfinance/테마)
├── ui/
│   ├── components.py       # 공통 컴포넌트
│   ├── styles.css          # 다크 테마 CSS
│   ├── session_keys.py     # session_state 키 상수
│   └── tabs/               # 탭별 렌더 함수
├── storage/                # 즐겨찾기/히스토리 JSON
├── prompts/                # AI 프롬프트 템플릿 (.md)
└── data/                   # 캐시/로그 (gitignore)
```

## 캐시 정책

| 데이터 | TTL |
|---|---|
| 거래대금 TOP100 | 5분 |
| 외국인/기관 순매수 | 1시간 |
| 종목 뉴스 | 1시간 |
| AI 코멘트 (movers) | 30분 |
| AI 종가베팅 분석 | 10분 |
| 세계 증시 | 5분 |
| 차트 OHLC | 5분 |
| 테마 매핑 | 24시간 |

## 면책

본 서비스는 정보 제공 목적이며, 투자 권유가 아닙니다. 모든 투자 결정의 책임은 사용자에게 있습니다.
