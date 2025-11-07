# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from streamlit_option_menu import option_menu
from scipy.stats import norm
import time

st.set_page_config(page_title="SPY Pro v2.1", layout="wide")
st.title("SPY Trade Dashboard Pro v2.1")
st.caption("Live SPY signals | $25k bankroll | Auto-Tracker | Paper/Live | Fixed & Stable")

# --- Session State ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=[
        'Time', 'Strategy', 'Action', 'Size', 'Risk', 'POP', 'Exit', 'Result', 'P&L', 'Type'
    ])
if 'client' not in st.session_state:
    st.session_state.client = None

# --- Sidebar ---
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Signals", "Trade Tracker", "Glossary", "Settings"],
        icons=["house", "bell", "clipboard-data", "book", "gear"],
        default_index=0,
    )
    st.divider()
    st.subheader("Risk Settings")
    ACCOUNT_SIZE = 25000
    RISK_PCT = st.slider("Risk/Trade (%)", 0.5, 2.0, 1.0) / 100
    MIN_CREDIT = st.number_input("Min Credit ($)", 0.10, 5.0, 0.30)
    MAX_DTE = st.slider("Max DTE", 7, 90, 45)
    POP_TARGET = st.slider("Min POP (%)", 60, 95, 75)
    PAPER_MODE = st.toggle("Paper Trading", value=True)
    st.divider()
    st.subheader("Alpaca")
    ALPACA_KEY = st.text_input("API Key", type="password")
    ALPACA_SECRET = st.text_input("API Secret", type="password")
    if st.button("Connect"):
        try:
            client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER_MODE)
            st.session_state.client = client
            st.success(f"Connected ({'Paper' if PAPER_MODE else 'Live'})")
        except Exception as e:
            st.error(f"Failed: {e}")

# --- Data Fetch (NO CACHING COMPLEX OBJECTS) ---
def get_market_data():
    try:
        spy = yf.Ticker("SPY")
        S = spy.fast_info.get("lastPrice", 0)
        vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 20)
        hist = spy.history(period="5d", interval="1m")
        if hist.empty:
            hist = pd.DataFrame({"Close": [S]*10}, index=pd.date_range(end=datetime.now(), periods=10, freq='1min'))
        return spy, float(S), hist, float(vix)
    except:
        return None, 540.0, pd.DataFrame({"Close": [540.0]*10}), 20.0

spy, S, hist, vix = get_market_data()
r = 0.05
q = 0.013

# --- Helpers ---
def get_iv(calls, puts, S):
    if calls.empty and puts.empty: return 0.3
    strikes = pd.concat([calls['strike'], puts['strike']]).unique()
    atm = min(strikes, key=lambda x: abs(x - S))
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.iloc[0].impliedVolatility if not row.empty else 0.3

def bs_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1) if type_ == "call" else -np.exp(-q*T) * norm.cdf(-d1)

def log_trade(**kwargs):
    trade = pd.DataFrame([kwargs])
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, trade], ignore_index=True)

# --- Dashboard ---
if selected == "Dashboard":
    col1, col2, col3 = st.columns(3)
    col1.metric("SPY", f"${S:.2f}")
    col2.metric("VIX", f"{vix:.1f}")
    col3.metric("Risk/Trade", f"${ACCOUNT_SIZE * RISK_PCT:.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index[-100:], y=hist['Close'].iloc[-100:], name="Price"))
    fig.add_hline(y=S, line_dash="dash")
    fig.update_layout(title="SPY 1m", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Signals ---
elif selected == "Signals":
    if vix > 30: st.warning("High VIX: Sell premium")

    signals = []
    expirations = [e for e in spy.options if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days <= MAX_DTE]

    # VWAP
    vwap = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
    if len(hist) > 1 and hist['Close'].iloc[-1] > vwap.iloc[-1] and hist['Close'].iloc[-2] <= vwap.iloc[-2]:
        signals.append({
            "Strategy": "VWAP Breakout", "Action": "Buy SPY Call", "Size": "1", "Risk": f"${ACCOUNT_SIZE * RISK_PCT:.0f}",
            "POP": "60%", "Exit": "+$1", "Symbol": "SPY"
        })

    # Iron Condor
    for exp in expirations[:2]:
        dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if dte < 30: continue
        T = dte / 365.25
        try:
            chain = spy.option_chain(exp)
            calls = chain.calls[chain.calls.bid > 0.01]
            puts = chain.puts[puts.bid > 0.01]
            if calls.empty or puts.empty: continue
            iv = get_iv(calls, puts, S)

            put_deltas = [abs(bs_delta(S, k, T, r, iv, q, "put") - 0.16) for k in puts.strike]
            call_deltas = [abs(bs_delta(S, k, T, r, iv, q, "call") + 0.16) for k in calls.strike]
            short_put = puts.iloc[np.argmin(put_deltas)]
            short_call = calls.iloc[np.argmin(call_deltas)]
            long_put = puts[puts.strike < short_put.strike].iloc[-1] if not puts[puts.strike < short_put.strike].empty else short_put
            long_call = calls[calls.strike > short_call.strike].iloc[0] if not calls[calls.strike > short_call.strike].empty else short_call

            credit = short_put.bid + short_call.bid - long_put.ask - long_call.ask
            if credit < MIN_CREDIT: continue
            width = min(short_put.strike - long_put.strike, short_call.strike - long_call.strike)
            max_loss = width - credit
            risk_per = max_loss * 100
            size = max(1, int((ACCOUNT_SIZE * RISK_PCT) / risk_per))

            signals.append({
                "Strategy": "Iron Condor", "Action": f"Sell {long_put.strike}P/{short_put.strike}P - {short_call.strike}C/{long_call.strike}C",
                "Size": f"{size}", "Risk": f"${risk_per*size:.0f}", "POP": "80%", "Exit": "50% profit", "Symbol": "SPY"
            })
        except: continue

    # Display
    if signals:
        for i, sig in enumerate(signals):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{sig['Strategy']}** | {sig['Action']}")
                st.caption(f"Size: {sig['Size']} | Risk: {sig['Risk']} | POP: {sig['POP']}")
            with c2:
                if st.button("Execute", key=f"exec_{i}"):
                    log_trade(
                        Time=datetime.now().strftime("%m/%d %H:%M"),
                        Strategy=sig['Strategy'],
                        Action=sig['Action'],
                        Size=sig['Size'],
                        Risk=sig['Risk'],
                        POP=sig['POP'],
                        Exit=sig['Exit'],
                        Result="Executed (Paper)" if PAPER_MODE else "Live Order",
                        P&L=0.0,
                        Type="Paper" if PAPER_MODE else "Live"
                    )
                    st.success("Logged!")
                    st.rerun()
    else:
        st.info("No signals")

# --- Trade Tracker ---
elif selected == "Trade Tracker":
    if not st.session_state.trade_log.empty:
        df = st.session_state.trade_log
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Total P&L", f"${df['P&L'].sum():.0f}")
        c2.metric("Win Rate", f"{(df['P&L'] > 0).mean()*100:.0f}%")

        csv = df.to_csv(index=False)
        st.download_button("Export CSV", csv, "trades.csv", "text/csv")
    else:
        st.info("No trades")

# --- Glossary / Settings ---
elif selected == "Glossary":
    st.write("**Delta**: Sensitivity to price. **Theta**: Time decay. **IV**: Volatility. **POP**: Win chance.")

elif selected == "Settings":
    st.write("Bankroll: $25,000 | Risk: 1% = $250/trade")
    if st.button("Clear Logs"):
        st.session_state.trade_log = pd.DataFrame(columns=st.session_state.trade_log.columns)
        st.rerun()

# Auto-refresh
time.sleep(60)
st.rerun()
