import streamlit as st
from pykrx import stock
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize
from typing import Dict

# ==== 추가: S&P500용 yfinance (선택) ====
try:
    import yfinance as yf
except ImportError:
    yf = None


# =========================
# 유틸 함수들
# =========================

# =========================
# 유명 자산배분 전략 정의
# =========================
# ※ 종목 코드는 예시이며, 한국 상장 ETF 기준
#   - 069500: KODEX 200 (코스피200) :contentReference[oaicite:0]{index=0}
#   - 360750: TIGER 미국S&P500 :contentReference[oaicite:1]{index=1}
#   - 114260: KODEX 국고채3년 :contentReference[oaicite:2]{index=2}
#   - 148070: KIWOOM 국고채10년(장기 국고채) :contentReference[oaicite:3]{index=3}
#   - 130730: KOSEF 단기자금(머니마켓/단기채) :contentReference[oaicite:4]{index=4}
#   - 132030: KODEX 골드선물(H) :contentReference[oaicite:5]{index=5}

MODEL_STRATEGIES = {
    "60:40 (주식:채권)": {
        "description": "고전적인 60:40 자산배분 (주식 60%, 채권 40%)",
        "sleeves": {
            "stock": {
                "target_weight": 0.60,
                "tickers": ["069500", "360750"],  # KOSPI200 + S&P500
            },
            "bond": {
                "target_weight": 0.40,
                "tickers": ["114260", "148070"],  # 중기 + 장기 국고채
            },
        },
    },
    "Permanent (영구 포트폴리오)": {
        "description": "Harry Browne 영구 포트폴리오: 주식/장기채/현금/금 25%씩",
        "sleeves": {
            "stock": {
                "target_weight": 0.25,
                "tickers": ["069500", "360750"],
            },
            "long_bond": {
                "target_weight": 0.25,
                "tickers": ["148070"],  # 10년 국고채 ETF
            },
            "cash": {
                "target_weight": 0.25,
                "tickers": ["130730"],  # 단기자금 ETF
            },
            "gold": {
                "target_weight": 0.25,
                "tickers": ["132030"],  # 골드 선물 ETF
            },
        },
    },
}


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def get_etf_price_df(tickers, start_date: date, end_date: date) -> pd.DataFrame:
    """
    여러 ETF의 종가를 하나의 DataFrame으로 합치기
    index: 날짜
    columns: ticker
    """
    start = yyyymmdd(start_date)
    end = yyyymmdd(end_date)

    close_list = []
    for t in tickers:
        ohlcv = stock.get_etf_ohlcv_by_date(start, end, t)
        if ohlcv.empty:
            continue
        close = ohlcv["종가"].rename(t)
        close_list.append(close)

    if not close_list:
        return pd.DataFrame()

    prices = pd.concat(close_list, axis=1)
    prices.index = pd.to_datetime(prices.index)
    # 모든 ETF가 상장된 이후 구간만 사용 (NaN 제거)
    prices = prices.dropna(how="any")
    return prices


def calc_advanced_portfolio(
    prices: pd.DataFrame,
    weights: dict,
    initial_capital: float = 1_000_000.0,
    monthly_payment: float = 0.0,
    payment_day: int = 25,
    rebal_freq: str = "분기별",
    fee_rate: float = 0.0015  # [추가] 기본 수수료 0.15% (매매수수료 + 슬리피지 가정)
):
    """
    [업그레이드] 현금 흐름(적립)이 있어도 성과 지표(CAGR, Sharpe)를 
    정확히 발라내서 계산하는 포트폴리오 시뮬레이터
    """
    prices = prices.sort_index()
    tickers = list(weights.keys())
    
    # 목표 비중
    target_w = np.array([weights[t] for t in tickers])
    
    # 일별 수익률 데이터
    rets = prices[tickers].pct_change().fillna(0.0)
    dates = rets.index
    
    # 시뮬레이션 상태 변수
    current_asset_values = np.array(initial_capital * target_w)
    total_invested = initial_capital
    
    # 결과 저장용 리스트
    equity_curve = []       # 평가액 (내 돈 + 수익)
    invested_curve = []     # 투입 원금 (내 돈)
    daily_rets_curve = []   # 포트폴리오의 '순수' 일별 수익률 (입금 효과 제외)
    
    prev_date = dates[0]
    
    # 리밸런싱 및 적립 로직
    for i, date in enumerate(dates):
        # 1. 자산 변동 (Drift)
        # -----------------------------------------------
        # 어제 장 마감 후 자산(Pre-return) -> 오늘 장 마감 자산(Post-return)
        start_val = np.sum(current_asset_values) # 수익률 반영 전 총액
        
        daily_ret_vector = rets.iloc[i].values
        current_asset_values = current_asset_values * (1 + daily_ret_vector)
        
        end_val = np.sum(current_asset_values)   # 수익률 반영 후 총액
        
        # [핵심] 입금 효과를 제외한 '순수 일별 수익률' 계산
        # 오늘 자산 / (어제 자산 + 어제 입금했으면 그 돈) - 1 인데,
        # 여기 로직상 입금은 '수익률 계산 후'에 이루어진다고 가정하거나,
        # 전날 총액 대비 오늘의 변동분을 기록함.
        if start_val > 0:
            pure_daily_ret = (end_val / start_val) - 1.0
        else:
            pure_daily_ret = 0.0
        
        daily_rets_curve.append(pure_daily_ret)
        
        # 2. 적립식 투자 (Cash Inflow) - 장 마감 후 입금 가정
        # -----------------------------------------------
        # "이번 달에 입금해야 하고, 오늘이 그 날짜(이후)라면"
        # (로직 단순화를 위해 date.day == payment_day 사용, 실제론 영업일 체크 필요)

        is_payday = (date.day >= payment_day) 

        # [추가] 마지막으로 적립금을 넣은 '월'을 기억하는 변수
    # 초기값은 불가능한 달(-1)이나, 시작일 이전 달로 설정
        last_paid_month = -1
        
        # 월이 바뀌었는지 체크 (중복 입금 방지용 간단 로직)
        if monthly_payment > 0:
            if date.day >= payment_day and date.month != last_paid_month:
                # [수정] 돈이 들어올 때도 매수 수수료가 발생합니다!
                # 투입금: 100만원 -> 실제 자산 증가분: 100만원 * (1 - 수수료율)
                
                real_input_value = monthly_payment * (1 - fee_rate)
                
                current_asset_values += (real_input_value * target_w)
                total_invested += monthly_payment # 내 원금은 수수료 떼기 전 금액으로 기록
                last_paid_month = date.month

        # 3. 리밸런싱 (Rebalancing)
        # -----------------------------------------------
        do_rebalance = False
        # ... (리밸런싱 주기 체크 로직 동일: 매월, 분기별, 매년 ...) ...
            
        if do_rebalance and rebal_freq != "없음 (Buy & Hold)":
            # 리밸런싱 전 총 평가액
            current_total_val = np.sum(current_asset_values)
            
            # 목표로 하는 각 자산별 금액
            target_asset_values = current_total_val * target_w
            
            # [핵심] 얼마나 사고 팔아야 하는가? (거래 회전율)
            # 현재 보유액과 목표 보유액의 차이의 절댓값 합 = 총 거래 대금
            trade_volume = np.sum(np.abs(current_asset_values - target_asset_values))
            
            # 비용 계산
            cost = trade_volume * fee_rate
            
            # 비용 차감 후 재분배
            final_total_val = current_total_val - cost
            current_asset_values = final_total_val * target_w

        # 기록 및 갱신
        equity_curve.append(np.sum(current_asset_values))
        invested_curve.append(total_invested)
        prev_date = date

    # ==================================================
    # 4. 결과 정리 및 지표 계산 (누락된 부분 복구)
    # ==================================================
    equity = pd.Series(equity_curve, index=dates, name="Equity")
    invested = pd.Series(invested_curve, index=dates, name="Invested")
    port_ret = pd.Series(daily_rets_curve, index=dates, name="Portfolio Return")
    
    # (1) 누적 수익률 (Cumulative Return) - TWRR 방식
    # 입출금 효과를 제거하고 "전략 자체가 얼마나 벌었나"를 보여줌
    cum_ret = (1 + port_ret).cumprod() - 1
    
    # (2) 총 수익률 (ROI) - 내 돈 대비 얼마나 불어났나 (사용자 체감 수익률)
    net_profit = equity.iloc[-1] - invested.iloc[-1]
    total_return_rate = net_profit / invested.iloc[-1] if invested.iloc[-1] > 0 else 0.0

    # (3) 연환산 수익률 (CAGR)
    # 적립식에서는 ROI를 기준으로 연환산하는 것은 부정확하지만, 
    # TWRR(cum_ret)을 기준으로 하면 "전략의 연평균 성과"는 구할 수 있음
    total_strategy_ret = cum_ret.iloc[-1]
    n_days = len(dates)
    if n_days > 1:
        ann_return = (1 + total_strategy_ret) ** (252 / n_days) - 1
        ann_vol = port_ret.std() * np.sqrt(252)
    else:
        ann_return, ann_vol = 0.0, 0.0
        
    # (4) 샤프 지수
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # (5) MDD
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "portfolio_return": port_ret, # 일별 수익률 (Series)
        "cum_return": cum_ret,        # 누적 수익률 곡선 (Series, TWRR 기준)
        "equity": equity,             # 자산 곡선 (Series)
        "invested": invested,         # 원금 곡선 (Series)
        "drawdown": drawdown,         # 낙폭 곡선 (Series)
        
        "total_return": total_return_rate, # [중요] 사용자가 번 돈 (ROI)
        "ann_return": ann_return,          # 전략의 연평균 성장률 (CAGR)
        "ann_vol": ann_vol,                # 변동성
        "sharpe": sharpe,                  # 샤프 지수
        "max_dd": max_dd,                  # MDD
        "final_equity": equity.iloc[-1]
    }


# ==== 추가: 벤치마크 지수 가격 가져오기 ====
INDEX_CODE_MAP = {
    "KOSPI": "1001",   # 코스피
    "KOSDAQ": "2001",  # 코스닥
}


def get_index_price_series(name: str, start_date: date, end_date: date) -> pd.Series:
    """
    name: 'KOSPI', 'KOSDAQ', 'S&P 500'
    return: 종가 시계열 (pd.Series, index=DatetimeIndex)
    """
    start = yyyymmdd(start_date)
    end = yyyymmdd(end_date)

    if name in ("KOSPI", "KOSDAQ"):
        code = INDEX_CODE_MAP[name]
        df = stock.get_index_ohlcv_by_date(start, end, code)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["종가"].copy()
        s.index = pd.to_datetime(s.index)
        s.name = name
        return s

    if name == "S&P 500":
        if yf is None:
            st.error("S&P 500을 사용하려면 'yfinance' 패키지를 설치해야 합니다. (pip install yfinance)")
            return pd.Series(dtype=float)
        df = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            return pd.Series(dtype=float)
        try:
            s = df["Adj Close"].copy()
        except KeyError:
            s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        s.name = name
        return s

    return pd.Series(dtype=float)




def calc_simple_stats_from_price(price: pd.Series):
    """
    종가 시계열로부터 수익률/변동성/샤프 계산 (벤치마크용)
    """
    price = price.sort_index()
    if len(price) < 2:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
        }

    rets = price.pct_change().dropna()
    total_return = price.iloc[-1] / price.iloc[0] - 1
    n_days = len(rets)

    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol not in [0, np.nan] else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }

def optimize_sharpe(prices: pd.DataFrame) -> dict:
    """
    prices: 각 ETF 종가 DataFrame (col: ticker)
    return: {ticker: weight} Sharpe 최대화 가중치 (합 1)
    """
    tickers = list(prices.columns)
    if len(tickers) < 2:
        # ETF가 1개뿐이면 최적화 의미가 거의 없으므로 전부 1
        return {tickers[0]: 1.0}

    # 일별 수익률
    rets = prices.pct_change().dropna()
    if rets.empty:
        # 데이터가 너무 짧은 경우
        return {t: 1.0 / len(tickers) for t in tickers}

    mu = rets.mean().values         # 각 ETF의 평균 일별 수익률
    cov = rets.cov().values         # 공분산 행렬

    def neg_sharpe(w):
        # w: 가중치 벡터
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        if port_vol == 0:
            return 1e6
        # Sharpe = 평균수익 / 변동성 (일 단위 기준, 연환산 상수는 최적화에선 의미 없음)
        sharpe = port_ret / port_vol
        return -sharpe  # 최소화 문제로 바꾸기 위해 음수

    n = len(tickers)
    w0 = np.array([1.0 / n] * n)  # 초기값: 균등 비중
    bounds = [(0.0, 1.0)] * n     # 각 비중은 0~1 사이
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 비중 합 = 1
    )

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)

    if (not res.success) or np.any(np.isnan(res.x)):
        # 최적화 실패 시 균등 비중으로 fallback
        w_opt = w0
    else:
        w_opt = res.x

    return {t: float(w_opt[i]) for i, t in enumerate(tickers)}


def optimize_sharpe(prices: pd.DataFrame) -> Dict[str, float]:
    """
    주어진 ETF 가격 시계열로부터
    '합이 1이고, 단순 롱 온리(0~1)' 제약하에서 Sharpe 비율을 최대화하는 비중 계산
    """
    prices = prices.dropna(how="any").sort_index()
    tickers = list(prices.columns)

    # 종목 1개면 그냥 100%
    if len(tickers) == 1:
        return {tickers[0]: 1.0}

    rets = prices.pct_change().dropna()
    if rets.empty:
        # 데이터가 거의 없으면 균등 비중
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}

    mu = rets.mean().values  # 일별 기대수익률
    cov = rets.cov().values  # 공분산 행렬
    n = len(tickers)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu)
        port_vol = float((w @ cov @ w) ** 0.5)
        if port_vol == 0:
            return 1e6
        return -port_ret / port_vol

    # 제약조건: 합 = 1, 각 비중 0~1
    w0 = np.array([1.0 / n] * n)
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=[cons])

    if not res.success:
        # 실패하면 균등 비중으로 fallback
        return {t: 1.0 / n for t in tickers}

    w_opt = np.clip(res.x, 0, None)
    w_opt = w_opt / w_opt.sum()

    return {t: float(w) for t, w in zip(tickers, w_opt)}


def build_strategy_portfolio(
    strategy_name: str,
    start_date: date,
    end_date: date,
    initial_capital: float,
):
    """
    전략 이름(MODEL_STRATEGIES key)을 받아
    - Sharpe 기준으로 ETF 비중 결정
    - 백테스트 결과까지 한 번에 계산
    """
    if strategy_name not in MODEL_STRATEGIES:
        return None, None, None

    strat = MODEL_STRATEGIES[strategy_name]
    sleeves = strat["sleeves"]

    # 1) 전체 티커 집합
    all_tickers = sorted({t for s in sleeves.values() for t in s["tickers"]})

    # 2) 전체 가격 데이터 한 번에 로딩
    prices_all = get_etf_price_df(all_tickers, start_date, end_date)
    if prices_all.empty:
        return None, None, None

    raw_weights: Dict[str, float] = {}

    # 3) 자산군(sleeve)별로 Sharpe 최적화
    for sleeve_name, sleeve in sleeves.items():
        target_w = sleeve["target_weight"]
        tickers = [t for t in sleeve["tickers"] if t in prices_all.columns]

        if not tickers:
            continue

        sub_prices = prices_all[tickers]

        if len(tickers) == 1:
            intra_w = {tickers[0]: 1.0}
        else:
            intra_w = optimize_sharpe(sub_prices)

        # 자산군 타깃 비중을 곱해서 전체 포트폴리오 비중으로 합산
        for t, w in intra_w.items():
            raw_weights[t] = raw_weights.get(t, 0.0) + target_w * w

    if not raw_weights:
        return None, None, None

    # 4) 혹시 모를 누락 대비 정규화 (합 = 1)
    total = sum(raw_weights.values())
    final_weights = {t: w / total for t, w in raw_weights.items()}

    # 5) 전략 포트폴리오 백테스트
    used_tickers = list(final_weights.keys())
    strat_prices = prices_all[used_tickers]

    strat_result = calc_advanced_portfolio(
        strat_prices,
        final_weights,
        initial_capital=initial_capital,
    )

    return final_weights, strat_result, strat_prices


# ==== 추가: 모델 포트폴리오 정의 ====
#  - 069500: KODEX 200 (국내 주식)
#  - 360750: TIGER 미국S&P500 (미국 주식)
#  - 114260: KODEX 국고채3년 (국내 채권)
MODEL_PORTFOLIOS = {
    "안정형": {
        "114260": 0.60,
        "069500": 0.20,
        "360750": 0.20,
    },
    "중립형": {
        "114260": 0.30,
        "069500": 0.35,
        "360750": 0.35,
    },
    "공격형": {
        "114260": 0.10,
        "069500": 0.45,
        "360750": 0.45,
    },
}
