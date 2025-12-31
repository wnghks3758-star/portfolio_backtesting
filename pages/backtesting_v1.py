import streamlit as st
from pykrx import stock
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime
from utils.util import *



# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ETF 포트폴리오 백테스트", layout="wide")

st.title("📈 ETF 포트폴리오 백테스트")

st.markdown(
    """
원하는 **ETF / 비중 / 기간**을 선택해서  
간단한 백테스트 결과(수익곡선, 수익률, 변동성 등)를 확인할 수 있는 데모입니다.

또한 포트폴리오를 **KOSPI / KOSDAQ / S&P 500**과 비교할 수 있습니다.
"""
)

# ---- 사이드바: 기본 설정 ----
# ---- 사이드바: 기본 설정 ----
st.sidebar.header("설정")

# 1. 투자 방식 선택 UI
st.sidebar.subheader("💰 투자 방식")
invest_type = st.sidebar.radio(
    "투자 방식을 선택하세요",
    ["거치식 (한 번에 투자)", "적립식 (매월 적립)"],
    index=0
)

# 2. 자금 설정
initial_capital = st.sidebar.number_input(
    "초기 투자금 (원)", 
    min_value=0.0, 
    value=10_000_000.0, 
    step=1_000_000.0, 
    format="%.0f" # 소수점 없이 정수로 표시
)

# 적립식일 경우에만 '월 적립금' 입력창이 나타나게 함 (조건부 렌더링)
monthly_payment = 0.0 
if invest_type == "적립식 (매월 적립)":
    monthly_payment = st.sidebar.number_input(
        "매월 추가 납입금 (원)", 
        min_value=0.0, 
        value=1_000_000.0, 
        step=100_000.0, 
        format="%.0f"
    )
    st.sidebar.caption("💡 매월 25일(월급날)에 비율대로 추가 매수합니다.")

# (이후 포트폴리오 유형 선택 코드는 그대로 두시면 됩니다)

# 🔹 포트폴리오 유형 선택 (새로 추가된 부분)
st.sidebar.subheader("포트폴리오 유형")
portfolio_type = st.sidebar.selectbox(
    "포트폴리오 유형을 선택하세요",
    ["직접 설정", "안정형", "중립형", "공격형"],
    index=1,  # 기본값: 중립형
)
if portfolio_type != "직접 설정":
    st.sidebar.caption(
        f"선택된 유형({portfolio_type})에 맞춰 ETF 구성과 비중이 기본값으로 설정됩니다. 이후 수동 조정도 가능합니다."
    )

# 기간 설정
today = date.today()
default_start = today - timedelta(days=365)

# 1. 날짜 범위 제한 설정 (예: 2000년 1월 1일부터 가능하게)
min_date_allowed = datetime.date(2000, 1, 1)

# 2. date_input에 min_value 파라미터 추가
start_date = st.sidebar.date_input(
    "시작일", 
    value=default_start, 
    min_value=min_date_allowed,  # 이 부분이 핵심입니다!
    max_value=today
)

end_date = st.sidebar.date_input(
    "종료일", 
    value=today, 
    min_value=min_date_allowed,
    max_value=today
)

if start_date >= end_date:
    st.sidebar.error("시작일이 종료일보다 같거나 늦을 수 없습니다.")
    st.stop()

# 기준 날짜 기준 ETF 리스트 가져오기
st.sidebar.subheader("ETF 선택")

ref_date_str = yyyymmdd(end_date)
ticker_list = stock.get_etf_ticker_list(ref_date_str)

# {라벨: 티커} 매핑 생성
label_to_ticker = {}
for t in ticker_list:
    name = stock.get_etf_ticker_name(t)
    label = f"{name} ({t})"
    label_to_ticker[label] = t

labels_sorted = sorted(label_to_ticker.keys())

# 🔹 포트폴리오 유형에 따라 기본 선택 ETF / 기본 비중 결정
model_weights = {}
if portfolio_type == "직접 설정":
    default_selection = labels_sorted[:5]  # 기존처럼 상위 5개
else:
    # 모델 포트폴리오에 정의된 ETF들만 기본 선택
    base_weights = MODEL_PORTFOLIOS.get(portfolio_type, {})
    default_selection = [
        label for label, ticker in label_to_ticker.items()
        if ticker in base_weights.keys()
    ]
    # 혹시 해당 날짜에 상장 전이라 하나도 못 찾으면 fallback
    if not default_selection:
        default_selection = labels_sorted[:5]
        base_weights = {}
    model_weights = base_weights

selected_labels = st.sidebar.multiselect(
    "포트폴리오에 포함할 ETF를 선택하세요",
    labels_sorted,
    default=default_selection,
)

if not selected_labels:
    st.warning("좌측에서 ETF를 하나 이상 선택해주세요.")
    st.stop()

selected_tickers = [label_to_ticker[l] for l in selected_labels]

# 비중 설정
st.sidebar.subheader("비중 설정(%)")

weights_raw = {}
for label, ticker in zip(selected_labels, selected_tickers):
    # 🔹 모델 포트폴리오가 있으면 그 비중을 기본값으로, 아니면 균등비중
    if ticker in model_weights:
        default_w = round(model_weights[ticker] * 100, 2)
    else:
        default_w = round(100 / len(selected_labels), 2)

    w = st.sidebar.number_input(
        f"{label} 비중(%)",
        min_value=0.0,
        max_value=100.0,
        value=default_w,
        step=1.0,
    )
    weights_raw[ticker] = w

sum_w = sum(weights_raw.values())
if sum_w == 0:
    st.sidebar.error("비중의 합이 0%입니다. 하나 이상은 0보다 큰 값으로 설정해주세요.")
    st.stop()

st.sidebar.write(f"비중 합계: **{sum_w:.2f}%**")
st.sidebar.caption("비중은 자동으로 100% 기준으로 정규화되어 계산됩니다.")

# ==== 리벨런싱 설정 ====
st.sidebar.subheader("⚖️ 리밸런싱 설정")
rebal_freq = st.sidebar.selectbox(
    "리밸런싱 주기",
    ["없음 (Buy & Hold)", "매월", "분기별 (3, 6, 9, 12월)", "매년"],
    index=2 # 기본값: 분기별 추천
)

# ==== 벤치마크 선택 ====
st.sidebar.subheader("비교할 벤치마크 지수")
benchmark_label_to_name = {
    "KOSPI (코스피 지수)": "KOSPI",
    "KOSDAQ (코스닥 지수)": "KOSDAQ",
    "S&P 500 (미국)": "S&P 500",
}
benchmark_labels = list(benchmark_label_to_name.keys())

selected_benchmark_labels = st.sidebar.multiselect(
    "비교 지수 선택 (복수 선택 가능)",
    benchmark_labels,
    default=["KOSPI (코스피 지수)"],  # 기본은 KOSPI 비교
)

selected_benchmarks = [
    benchmark_label_to_name[l] for l in selected_benchmark_labels
]

# ==== 추가: 전략 포트폴리오 선택 ====
st.sidebar.subheader("모델 포트폴리오 (옵션)")

strategy_options = ["선택 안 함"] + list(MODEL_STRATEGIES.keys())
selected_strategy = st.sidebar.selectbox(
    "유명 자산배분 전략 선택",
    strategy_options,
    index=0,
)

run_button = st.sidebar.button("백테스트 실행하기")


# =========================
# 메인 로직
# =========================
if not run_button:
    st.info("좌측에서 설정을 마친 뒤 **'백테스트 실행하기'** 버튼을 눌러주세요.")
    st.stop()

with st.spinner("데이터 불러오는 중..."):
    prices = get_etf_price_df(selected_tickers, start_date, end_date)

if prices.empty:
    st.error("해당 기간에 대한 가격 데이터가 없습니다. 기간을 다시 설정해보세요.")
    st.stop()

st.success(f"가격 데이터 로딩 완료! (거래일 수: {len(prices):,}일)")

# 비중 정규화 (합 1이 되도록)
weights = {t: w / sum_w for t, w in weights_raw.items()}

# 백테스트 계산
result = calc_advanced_portfolio(
    prices, 
    weights, 
    initial_capital=initial_capital,
    monthly_payment=monthly_payment,  # <--- 이 부분 추가
    rebal_freq=rebal_freq
)

equity = result["equity"]
cum_return = result["cum_return"]
port_ret = result["portfolio_return"]
drawdown = result["drawdown"]

# =========================
# 결과 표시 - 기본 포트폴리오 지표
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("총 수익률", f"{result['total_return']*100:,.2f} %")
if not np.isnan(result["ann_return"]):
    col2.metric("연환산 수익률", f"{result['ann_return']*100:,.2f} %")
else:
    col2.metric("연환산 수익률", "N/A")

if not np.isnan(result["ann_vol"]):
    col3.metric("연환산 변동성", f"{result['ann_vol']*100:,.2f} %")
else:
    col3.metric("연환산 변동성", "N/A")

col4.metric("최대 낙폭(MDD)", f"{result['max_dd']*100:,.2f} %")

st.subheader("📊 포트폴리오 자산 곡선")
st.line_chart(equity.rename("Portfolio Equity"))

st.subheader("📈 개별 ETF 가격 (정규화, 시작일 = 1)")
norm_prices = prices[selected_tickers] / prices[selected_tickers].iloc[0]
st.line_chart(norm_prices)

st.subheader("📉 포트폴리오 일별 수익률")
st.bar_chart(port_ret)

with st.expander("📃 원본 데이터 (종가) 보기"):
    st.dataframe(prices)

with st.expander("📉 드로다운(낙폭) 시계열 보기"):
    st.area_chart(drawdown)

# =========================
# Sharpe 최대화 최적화 포트폴리오 추천
# =========================
st.subheader("🤖 Sharpe 최대화 포트폴리오 추천")

opt_flag = st.checkbox("선택한 ETF들로 Sharpe 최대화 포트폴리오 계산하기")

if opt_flag:
    with st.spinner("Sharpe 최대화 포트폴리오 최적화 중..."):
        # get_etf_price_df에서 NaN row를 이미 drop 했으므로 그대로 사용
        opt_weights = optimize_sharpe(prices)
        opt_result = calc_advanced_portfolio(
                prices, 
                weights, 
                initial_capital=initial_capital,
                monthly_payment=monthly_payment,  # <--- 이 부분 추가
                rebal_freq=rebal_freq
            )

    # 추천 비중 테이블
    st.markdown("**추천 포트폴리오 비중 (Sharpe 최대화 기준)**")
    # ticker -> label(이름 포함) 매핑 역으로 만들기
    ticker_to_label = {v: k for k, v in label_to_ticker.items()}

    rows = []
    for t, w in opt_weights.items():
        label = ticker_to_label.get(t, t)
        rows.append(
            {
                "Ticker": t,
                "ETF": label,
                "Weight (%)": w * 100,
            }
        )
    opt_weight_df = pd.DataFrame(rows).sort_values("Weight (%)", ascending=False)
    st.dataframe(opt_weight_df.reset_index(drop=True))

    # 추천 포트폴리오 성과 지표
    st.markdown("**추천 포트폴리오 성과 지표 (Sharpe 최대화)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 수익률", f"{opt_result['total_return']*100:,.2f} %")
    if not np.isnan(opt_result["ann_return"]):
        c2.metric("연환산 수익률", f"{opt_result['ann_return']*100:,.2f} %")
    else:
        c2.metric("연환산 수익률", "N/A")

    if not np.isnan(opt_result["ann_vol"]):
        c3.metric("연환산 변동성", f"{opt_result['ann_vol']*100:,.2f} %")
    else:
        c3.metric("연환산 변동성", "N/A")

    c4.metric("최대 낙폭(MDD)", f"{opt_result['max_dd']*100:,.2f} %")

    # 기존 포트폴리오 vs 최적화 포트폴리오 자산 곡선 비교
    st.markdown("**기존 포트폴리오 vs Sharpe 최대화 포트폴리오 자산 곡선**")
    compare_equity = pd.DataFrame(
        {
            "User Portfolio": equity,
            "Sharpe-Opt Portfolio": opt_result["equity"],
        }
    )
    st.line_chart(compare_equity)

    # 원하면 수익률 비교도
    with st.expander("📊 일별 수익률 비교 보기"):
        compare_ret = pd.DataFrame(
            {
                "User Portfolio": port_ret,
                "Sharpe-Opt Portfolio": opt_result["portfolio_return"],
            }
        )
        st.line_chart(compare_ret)



# =========================
# 벤치마크 비교
# =========================
if selected_benchmarks:
    st.subheader("📊 포트폴리오 vs 벤치마크 자산 성장 비교")

    # 모든 비교 대상의 '자산액(Equity)'을 담을 데이터프레임
    comp_equity_df = pd.DataFrame()
    
    # 1. 내 포트폴리오 결과 담기
    comp_equity_df["My Portfolio"] = result["equity"]

    # 2. 벤치마크들 루프 돌면서 똑같은 조건으로 시뮬레이션
    bench_stats_rows = []

    for name in selected_benchmarks:
        # A. 벤치마크 지수 가격 데이터 가져오기
        s = get_index_price_series(name, start_date, end_date)
        
        if s.empty:
            st.warning(f"{name} 지수 데이터를 가져오지 못했습니다.")
            continue
            
        # B. 데이터 전처리 (함수에 넣기 위해 DataFrame으로 변환)
        # 내 포트폴리오와 날짜 인덱스를 맞춤 (교집합)
        common_index = result["equity"].index.intersection(s.index)
        s = s.loc[common_index]
        
        if s.empty:
            continue

        # DataFrame 형태로 변환 (함수 입력 규격 맞춤)
        # 예: col name = "S&P 500"
        bench_price_df = s.to_frame(name=name)
        
        # C. 벤치마크 시뮬레이션 실행 (중요!)
        # 내 포트폴리오와 '완전히 동일한' 자금/적립 설정을 적용합니다.
        # 단, 벤치마크는 단일 종목이므로 비중은 {name: 1.0} 입니다.
        bench_res = calc_advanced_portfolio(
            prices=bench_price_df,
            weights={name: 1.0},             # 100% 몰빵
            initial_capital=initial_capital, # 내 설정과 동일
            monthly_payment=monthly_payment, # 내 설정과 동일 (적립식 적용)
            payment_day=25,                  # 내 설정과 동일
            rebal_freq="없음 (Buy & Hold)",   # 단일 종목이라 리밸런싱 의미 없음
            fee_rate=0.0                     # 지수 자체 비교이므로 보통 수수료는 0으로 둠 (원하면 fee_rate 적용 가능)
        )
        
        # D. 결과 저장
        # 자산 곡선 추가
        comp_equity_df[name] = bench_res["equity"]
        
        # 통계 지표 저장
        bench_stats_rows.append({
            "Name": name,
            "Total Return (%)": bench_res["total_return"] * 100,
            "Ann Return (%)": bench_res["ann_return"] * 100 if not np.isnan(bench_res["ann_return"]) else 0.0,
            "Ann Vol (%)": bench_res["ann_vol"] * 100 if not np.isnan(bench_res["ann_vol"]) else 0.0,
            "Max DD (%)": bench_res["max_dd"] * 100,
        })

    # 3. 차트 그리기
    # NaN 제거 (날짜 안 맞는 부분)
    comp_equity_df = comp_equity_df.dropna()
    
    if not comp_equity_df.empty:
        st.line_chart(comp_equity_df)
    else:
        st.info("표시할 데이터가 없습니다.")

    # 4. 성과 지표 비교 테이블
    st.markdown("#### 📋 성과 지표 상세 비교")
    
    # 내 포트폴리오 요약
    my_summary = {
        "Name": "My Portfolio",
        "Total Return (%)": result["total_return"] * 100,
        "Ann Return (%)": result["ann_return"] * 100 if not np.isnan(result["ann_return"]) else 0.0,
        "Ann Vol (%)": result["ann_vol"] * 100 if not np.isnan(result["ann_vol"]) else 0.0,
        "Max DD (%)": result["max_dd"] * 100,
    }
    
    summary_rows = [my_summary] + bench_stats_rows
    summary_df = pd.DataFrame(summary_rows).set_index("Name")
    
    # 보기 좋게 포맷팅
    st.dataframe(summary_df.style.format("{:.2f}"))

else:
    st.info("비교할 벤치마크가 선택되지 않았습니다.")

# =========================
# ==== 전략 포트폴리오 섹션 ====
# =========================
if selected_strategy != "선택 안 함":
    st.markdown("---")
    st.subheader(f"🎯 전략 포트폴리오 추천: {selected_strategy}")

    st.caption(MODEL_STRATEGIES[selected_strategy]["description"])

    with st.spinner("전략 포트폴리오 최적화 중..."):
        strat_weights, strat_result, strat_prices = build_strategy_portfolio(
            selected_strategy,
            start_date,
            end_date,
            initial_capital,
        )

    if strat_weights is None:
        st.warning("전략 포트폴리오를 구성할 수 있는 데이터가 부족합니다.")
    else:
        # 1) 비중 표
        st.markdown("#### 📋 전략 포트폴리오 구성 (Sharpe 최대화 기반)")

        # 티커 → 라벨 매핑 (앞에서 만든 label_to_ticker를 뒤집기)
        ticker_to_label = {v: k for k, v in label_to_ticker.items()}

        rows = []
        for t, w in strat_weights.items():
            label = ticker_to_label.get(t, t)
            rows.append(
                {
                    "Ticker": t,
                    "Name": label,
                    "Weight (%)": w * 100,
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("Ticker").sort_values("Weight (%)", ascending=False)
        )

        # 2) 지표 비교: 내 포트폴리오 vs 전략 포트폴리오
        st.markdown("#### 📊 성과 지표 비교 (내 포트 vs 전략 포트)")

        comp_rows = [
            {
                "Portfolio": "내 포트폴리오",
                "Total Return (%)": result["total_return"] * 100,
                "Ann Return (%)": result["ann_return"] * 100
                if not np.isnan(result["ann_return"]) else np.nan,
                "Ann Vol (%)": result["ann_vol"] * 100
                if not np.isnan(result["ann_vol"]) else np.nan,
                "Sharpe": result["sharpe"],
                "Max DD (%)": result["max_dd"] * 100,
            },
            {
                "Portfolio": selected_strategy,
                "Total Return (%)": strat_result["total_return"] * 100,
                "Ann Return (%)": strat_result["ann_return"] * 100
                if not np.isnan(strat_result["ann_return"]) else np.nan,
                "Ann Vol (%)": strat_result["ann_vol"] * 100
                if not np.isnan(strat_result["ann_vol"]) else np.nan,
                "Sharpe": strat_result["sharpe"],
                "Max DD (%)": strat_result["max_dd"] * 100,
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows).set_index("Portfolio"))

        # 3) 자산곡선 비교
        st.markdown("#### 📈 자산 곡선 비교 (내 포트 vs 전략 포트)")

        equity_user = result["equity"].rename("My Portfolio")
        equity_strat = strat_result["equity"].rename(selected_strategy)

        equity_comp = pd.concat([equity_user, equity_strat], axis=1).dropna(how="any")
        st.line_chart(equity_comp)
