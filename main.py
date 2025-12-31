import streamlit as st
import yfinance as yf
import pandas as pd

# 1. 페이지 기본 설정 (반드시 맨 처음에 와야 함)
st.set_page_config(
    page_title="ETF 포트폴리오 연구소",
    page_icon="🧪",
    layout="wide"
)

# 2. 제목 및 헤더
st.title("🧪 ETF 포트폴리오 연구소")
st.markdown("### 안전하고 체계적인 자산 배분을 위한 연구 공간")
st.divider()

# 3. 앱 사용 가이드 (사이드바 메뉴 설명)
col1, col2 = st.columns(2)

with col1:
    st.info("### 📘 1. 포트폴리오 백테스팅")
    st.write("""
    - **목적:** 내가 구성한 자산 배분 전략이 과거에 어떤 성과를 냈는지 검증합니다.
    - **주요 기능:**
        - 거치식 / 적립식 투자 시뮬레이션
        - 최대 낙폭(MDD) 및 원금 회복 기간 분석
        - 벤치마크(KOSPI, S&P500) 비교
    """)

with col2:
    st.success("### 🇺🇸 2. S&P 500 심층 분석")
    st.write("""
    - **목적:** 나에게 맞는 S&P 500 ETF(환헤지 vs 환노출)를 선택합니다.
    - **주요 기능:**
        - 환율 변동에 따른 수익률 차이 비교
        - 배당 재투자(TR) 효과 분석
    """)

st.divider()

# 4. (보너스) 주요 시장 지표 대시보드
# 연구원님이 좋아하실 만한 '오늘의 시장 분위기'를 간단히 보여줍니다.
st.subheader("📊 오늘의 주요 시장 지표")

# 데이터를 캐싱하여 속도 최적화
@st.cache_data(ttl=3600) # 1시간마다 갱신
def get_market_metrics():
    tickers = {
        "S&P 500": "^GSPC",
        "KOSPI": "^KS11",
        "원/달러 환율": "KRW=X",
        "미국 10년물 국채": "^TNX"
    }
    data = yf.download(list(tickers.values()), period="5d", auto_adjust=False)
    
    metrics = {}
    for name, code in tickers.items():
        try:
            # 최신 종가와 전일 종가 가져오기
            # yfinance 업데이트로 인한 컬럼 접근 방식 수정 반영 ('Close' 사용)
            if 'Adj Close' in data.columns:
                series = data['Adj Close'][code].dropna()
            else:
                series = data['Close'][code].dropna()
            
            latest = series.iloc[-1]
            prev = series.iloc[-2]
            change = (latest - prev) / prev * 100
            metrics[name] = (latest, change)
        except:
            metrics[name] = (0.0, 0.0)
    return metrics

metrics = get_market_metrics()

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
cols = [m_col1, m_col2, m_col3, m_col4]
names = ["S&P 500", "KOSPI", "원/달러 환율", "미국 10년물 국채"]

for col, name in zip(cols, names):
    val, change = metrics.get(name, (0, 0))
    col.metric(
        label=name, 
        value=f"{val:,.2f}", 
        delta=f"{change:+.2f}%"
    )

st.caption("※ 데이터 출처: Yahoo Finance (지연 시세일 수 있습니다)")