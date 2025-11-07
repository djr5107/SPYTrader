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
import time
import os

st.set_page_config(page_title="SPY Pro v2.0", layout="wide")
st.title("SPY Trade Dashboard Pro v2.0")
st.caption("Live signals | $25k account | Built-in Trade Tracker | Paper/Live Toggle")

# --- Initialize Session State ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []  # All trades
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'live_trades' not in st.session_state:
    st.session_state.live_trades = []
if 'client' not in st.session_state:
    st.session_state.client = None

# --- Sidebar: Navigation + Settings ---
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Signals", "Trade Tracker", "Glossary", "Settings"],
        icons=["house", "bell", "table", "book", "gear"],
        menu_icon="cast",
        default_index=0,
    )

    st.divider()
    st.subheader("Account")
    ACCOUNT_SIZE = 25000
    RISK_PCT = st.slider("Risk per Trade (%)", 0.5, 2.0, 1.0) / 100
    MIN_CREDIT = st.number_input("Min Credit ($)", 0.10, 5.0, 0.30)
    MAX_DTE = st.slider("Max DTE", 7, 90, 45)
    POP_TARGET = st.slider("Min POP (%)", 60, 95, 75)

    PAPER_MODE = st.toggle("Paper Trading", value=True)
    st.divider()

    st.subheader("Alpaca API")
    ALPACA_KEY = st.text_input("API Key", type="password")
    ALPACA_SECRET = st.text_input("API Secret", type="password")
    if st.button("Connect"):
        try:
            client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER_MODE)
            st.session_state.client = client
            st.success(f"Connected ({'Paper' if PAPER_MODE else 'Live'})")
        except:
            st.error("Invalid keys")

# --- Helpers ---
@st.cache_data(ttl=60)
def get_spy_data():
    spy = yf.Ticker("SPY")
    price = spy.fast_info["lastPrice"]
    hist = spy.history(period="5d", interval="1m")
    vix = yf.Ticker("^VIX").fast_info["lastPrice"]
    return spy, price, hist, vix

def get_iv(calls, puts, S):
    strikes = pd.concat([calls.strike, puts.strike]).unique()
    atm = min(strikes, key=lambda x: abs(x - S))
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.impliedVolatility.iloc[0] if not row.empty else 0.3

def bs_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 0
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1) if type_ == "call" else -np.exp(-q*T) * norm.cdf(-d1)

def log_trade(strategy, action, size, risk, pop, exit_rule, result="Pending"):
    trade = {
        "Time": datetime.now().strftime("%m/%d %H:%M"),
        "Strategy": strategy,
        "Action": action,
        "Size": size,
        "Risk": risk,
        "POP": pop,
        "Exit": exit_rule,
        "Result": result
    }
    st.session_state.trade_log.append(trade)
    if st.session_state.get('client') and not PAPER_MODE:
        st.session_state.live_trades.append(trade)
    else:
        st.session_state.paper_trades.append(trade)

def execute_and_log(client, symbol, qty, side, strategy, action, size, risk, pop, exit_rule):
    if not client:
        log_trade(strategy, action, size, risk, pop, exit_rule, "No API")
        return
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if "buy" in side.lower() else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )
        result = f"Executed (ID: {order.id})"
    except Exception as e:
        result = f"Failed: {str(e)}"
    log_trade(strategy, action, size, risk, pop, exit_rule, result)

# --- Dashboard ---
if selected == "Dashboard":
    spy, S, hist, vix = get_spy_data()
    r = 0.05
    q = 0.013

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("SPY Price", f"${S:.2f}")
    with col2:
        st.metric("VIX", f"{vix:.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index[-100:], y=hist['Close'][-100:], name="Price"))
    fig.add_hline(y=S, line_dash="dash")
    fig.update_layout(title="SPY 1-Minute", height=350)
    st.plotly_chart(fig, use_container_width=True)

# --- Signals ---
elif selected == "Signals":
    spy, S, hist, vix = get_spy_data()
    if vix > 30: st.warning("High VIX: Favor premium selling")

    signals = []
    expirations = [e for e in spy.options if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days <= MAX_DTE]

    # VWAP Breakout
    vwap = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
    if hist['Close'].iloc[-1] > vwap.iloc[-1] and hist['Close'].iloc[-2] <= vwap.iloc[-2]:
        signals.append({
            "Strategy": "VWAP Breakout", "Action": "Buy SPY Call (ATM)", "Size": "1", "Risk": "$250", "POP": "60%", "Exit": "+$1 or stop", "Symbol": "SPY"
        })

    # Iron Condor
    for exp in expirations[:2]:
        dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if dte < 30: continue
        T = dte / 365.25
        chain = spy.option_chain(exp)
        calls = chain.calls[(chain.calls.bid > 0.01)]
        puts = chain.puts[(chain.puts.bid > 0.01)]
        if calls.empty or puts.empty: continue
        iv = get_iv(calls, puts, S)

        short_put_idx = np.argmin(np.abs([bs_delta(S, k, T, r, iv, q, "put") - 0.16 for k in puts.strike]))
        short_call_idx = np.argmin(np.abs([bs_delta(S, k, T, r, iv, q, "call") + 0.16 for k in calls.strike]))
        short_put = puts.iloc[short_put_idx]
        short_call = calls.iloc[short_call_idx]
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

    # Display
    if signals:
        for sig in signals:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{sig['Strategy']}** | {sig['Action']}")
                st.caption(f"Size: {sig['Size']} | Risk: {sig['Risk']} | POP: {sig['POP']} | Exit: {sig['Exit']}")
            with col2:
                if st.button("Execute", key=sig['Strategy']):
                    execute_and_log(
                        st.session_state.client, sig['Symbol'], sig['Size'], "sell",
                        sig['Strategy'], sig['Action'], sig['Size'], sig['Risk'], sig['POP'], sig['Exit']
                    )
                    st.rerun()
    else:
        st.info("No signals. Adjust filters.")

# --- Trade Tracker ---
elif selected == "Trade Tracker":
    st.subheader("All Trades")
    if st.session_state.trade_log:
        df = pd.DataFrame(st.session_state.trade_log)
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", len(df))
            st.metric("Paper Trades", len(st.session_state.paper_trades))
        with col2:
            st.metric("Live Trades", len(st.session_state.live_trades))
            if st.button("Export CSV"):
                csv = df.to_csv(index=False)
                st.download_button("Download", csv, "trades.csv", "text/csv")
    else:
        st.info("No trades yet.")

    st.divider()
    tab1, tab2 = st.tabs(["Paper Trades", "Live Trades"])
    with tab1:
        if st.session_state.paper_trades:
            st.dataframe(pd.DataFrame(st.session_state.paper_trades), use_container_width=True)
        else:
            st.info("No paper trades.")
    with tab2:
        if st.session_state.live_trades:
            st.dataframe(pd.DataFrame(st.session_state.live_trades), use_container_width=True)
        else:
            st.info("No live trades.")

# --- Glossary ---
elif selected == "Glossary":
    st.write("**Key Terms:**")
    terms = {
        "Delta": "Option price sensitivity to underlying.",
        "Theta": "Time decay — profit for sellers.",
        "IV": "Expected volatility — high = rich premium.",
        "POP": "Probability of Profit.",
        "Wheel": "Sell puts → assign → sell calls → repeat."
    }
    for k, v in terms.items():
        st.write(f"**{k}:** {v}")

# --- Settings ---
elif selected == "Settings":
    st.write("Risk: 1% per trade = $250 max")
    st.write("Bankroll: $25,000")
    if st.button("Clear All Logs"):
        st.session_state.trade_log = []
        st.session_state.paper_trades = []
        st.session_state.live_trades = []
        st.rerun()

# Auto-refresh
time.sleep(1)
st.rerun()
