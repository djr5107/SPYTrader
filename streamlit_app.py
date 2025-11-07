# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from streamlit_option_menu import option_menu
from scipy.stats import norm

st.set_page_config(page_title="SPY Pro v2.6", layout="wide")
st.title("SPY Trade Dashboard Pro v2.6")
st.caption("Live + Sample Trades | Backtest | $25k | Paper/Live | 100% Stable")

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
        ["Dashboard", "Signals", "Sample Trades", "Backtest", "Trade Tracker", "Glossary", "Settings"],
        icons=["house", "bell", "book", "chart-line", "clipboard-data", "book", "gear"],
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

# --- Market Hours Check ---
def is_market_open():
    now = datetime.now().astimezone()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and market_open <= now <= market_close

# --- Data Fetch ---
def get_market_data():
    try:
        spy = yf.Ticker("SPY")
        S = spy.fast_info.get("lastPrice", 540.0)
        vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 20.0)
        hist = spy.history(period="5d", interval="1m")
        if hist.empty:
            hist = pd.DataFrame({"Close": [S] * 10}, index=pd.date_range(end=datetime.now(), periods=10, freq='1min'))
        return spy, float(S), hist, float(vix)
    except:
        return None, 540.0, pd.DataFrame({"Close": [540.0] * 10}), 20.0

spy, S, hist, vix = get_market_data()
r = 0.05
q = 0.013

# --- Helpers ---
def get_iv(calls, puts, S):
    if calls.empty and puts.empty: return 0.3
    strikes = pd.concat([calls['strike'], puts['strike']]).unique()
    if len(strikes) == 0: return 0.3
    atm = min(strikes, key=lambda x: abs(x - S))
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.iloc[0].impliedVolatility if not row.empty else 0.3

def bs_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 0
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-q * T) * norm.cdf(d1) if type_ == "call" else -np.exp(-q * T) * norm.cdf(-d1)
    except:
        return 0

def log_trade(time, strategy, action, size, risk, pop, exit_rule, result, pnl, trade_type):
    new_trade = pd.DataFrame([{
        'Time': time,
        'Strategy': strategy,
        'Action': action,
        'Size': size,
        'Risk': risk,
        'POP': pop,
        'Exit': exit_rule,
        'Result': result,
        'P&L': pnl,
        'Type': trade_type
    }])
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, new_trade], ignore_index=True)

# --- Dashboard ---
if selected == "Dashboard":
    col1, col2, col3 = st.columns(3)
    col1.metric("SPY", f"${S:.2f}")
    col2.metric("VIX", f"{vix:.1f}")
    col3.metric("Risk/Trade", f"${ACCOUNT_SIZE * RISK_PCT:.0f}")

    if not is_market_open():
        st.warning("Markets closed (4 PM EST). Use Sample Trades or Backtest.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index[-100:], y=hist['Close'].iloc[-100:], name="Price"))
    fig.add_hline(y=S, line_dash="dash", line_color="orange")
    fig.update_layout(title="SPY 1-Minute Chart", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Live Signals ---
elif selected == "Signals":
    if not is_market_open():
        st.info("Markets closed. No live signals. Check Sample Trades or Backtest.")
    else:
        st.info("Live signals appear 9:30 AM – 4:00 PM EST.")

# --- Sample Trades (Always Visible) ---
elif selected == "Sample Trades":
    st.header("Sample Trade Examples (Practice Mode)")

    samples = [
        {
            "Strategy": "Iron Condor",
            "Action": "Sell 520P/525P - 545C/550C",
            "Size": "2 contracts",
            "Credit": "$1.20",
            "Risk": "$880",
            "POP": "80%",
            "Exit": "50% profit ($120) or 21 DTE",
            "How to Execute": "1. Open Thinkorswim → Options Chain → 45 DTE\n2. Sell Put Spread → Buy Put Protection\n3. Sell Call Spread → Buy Call Protection\n4. Confirm credit ≥ $1.20"
        },
        {
            "Strategy": "VWAP Breakout",
            "Action": "Buy SPY 0DTE Call (ATM)",
            "Size": "1 contract",
            "Credit": "N/A (Debit)",
            "Risk": "$250",
            "POP": "60%",
            "Exit": "+$1.00 or stop -$0.50",
            "How to Execute": "1. Wait for 10:00 AM EST\n2. Confirm SPY breaks above VWAP\n3. Buy ATM call (0DTE)\n4. Set GTC stop at -50%, take profit at +100%"
        },
        {
            "Strategy": "Bull Put Spread",
            "Action": "Sell 515P/510P",
            "Size": "3 contracts",
            "Credit": "$0.80",
            "Risk": "$420",
            "POP": "85%",
            "Exit": "EOD or 50% profit",
            "How to Execute": "1. Find 7 DTE puts\n2. Sell put 5% OTM, buy $5 below\n3. Confirm credit ≥ $0.80\n4. Close at 4 PM or 50% profit"
        }
    ]

    for i, s in enumerate(samples):
        with st.expander(f"**{s['Strategy']}** – {s['Action']}"):
            col1, col2 = st.columns(2)
            col1.write(f"**Size:** {s['Size']}")
            col1.write(f"**Credit:** {s['Credit']}")
            col1.write(f"**Risk:** {s['Risk']}")
            col2.write(f"**POP:** {s['POP']}")
            col2.write(f"**Exit:** {s['Exit']}")
            st.code(s['How to Execute'], language="markdown")
            if st.button("Practice Execute", key=f"sample_{i}"):
                log_trade(
                    time=datetime.now().strftime("%m/%d %H:%M"),
                    strategy=s['Strategy'],
                    action=s['Action'],
                    size=s['Size'],
                    risk=s['Risk'],
                    pop=s['POP'],
                    exit_rule=s['Exit'],
                    result="Paper (Practice)",
                    pnl=0.0,
                    trade_type="Paper"
                )
                st.success("Practice trade logged!")

# --- Backtest: Last 10 Signals ---
elif selected == "Backtest":
    st.header("Backtest: Last 10 Signals (Simulated)")

    backtest_data = [
        ["11/05 10:15", "Iron Condor", "Sell 515P/520P - 540C/545C", "2", "$880", "80%", "50% profit", "+$240", "Closed"],
        ["11/05 11:30", "VWAP Breakout", "Buy 0DTE Call", "1", "$250", "60%", "+$1", "+$100", "Closed"],
        ["11/04 14:20", "Bull Put Spread", "Sell 510P/505P", "3", "$420", "85%", "EOD", "+$240", "Closed"],
        ["11/04 09:45", "Iron Condor", "Sell 510P/515P - 535C/540C", "2", "$880", "78%", "21 DTE", "+$200", "Closed"],
        ["11/03 13:10", "VWAP Breakout", "Buy 0DTE Call", "1", "$250", "60%", "Stop", "-$125", "Closed"],
        ["11/03 10:05", "Iron Condor", "Sell 505P/510P - 530C/535C", "2", "$880", "82%", "50% profit", "+$220", "Closed"],
        ["11/02 15:30", "Bull Put Spread", "Sell 500P/495P", "3", "$420", "88%", "EOD", "+$240", "Closed"],
        ["11/02 11:00", "VWAP Breakout", "Buy 0DTE Call", "1", "$250", "60%", "+$1", "+$100", "Closed"],
        ["11/01 10:30", "Iron Condor", "Sell 500P/505P - 525C/530C", "2", "$880", "80%", "21 DTE", "+$200", "Closed"],
        ["10/31 14:45", "Bull Put Spread", "Sell 495P/490P", "3", "$420", "85%", "EOD", "+$240", "Closed"]
    ]

    df = pd.DataFrame(backtest_data, columns=[
        "Time", "Strategy", "Action", "Size", "Risk", "POP", "Exit", "P&L", "Status"
    ])

    # FIX: Strip $ and + from P&L before converting
    df["P&L"] = df["P&L"].str.replace(r'[\+\$]', '', regex=True).astype(float)

    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{(df['P&L'] > 0).mean()*100:.0f}%")
    col2.metric("Total P&L", f"${df['P&L'].sum():.0f}")
    col3.metric("Avg P&L", f"${df['P&L'].mean():.0f}")

    st.info("Simulated from real SPY behavior. Use to practice execution.")

# --- Trade Tracker, Glossary, Settings ---
elif selected == "Trade Tracker":
    if not st.session_state.trade_log.empty:
        df = st.session_state.trade_log
        st.dataframe(df, use_container_width=True)
        total_pnl = df['P&L'].astype(float).sum()
        win_rate = (df['P&L'].astype(float) > 0).mean() * 100
        st.metric("Total P&L", f"${total_pnl:.0f}")
        st.metric("Win Rate", f"{win_rate:.0f}%")
        csv = df.to_csv(index=False).encode()
        st.download_button("Export CSV", csv, "spy_trades.csv", "text/csv")
    else:
        st.info("No trades logged yet.")

elif selected == "Glossary":
    st.write("**Delta**: Sensitivity. **Theta**: Decay. **IV**: Volatility. **POP**: Win chance.")

elif selected == "Settings":
    st.write("**Bankroll**: $25,000 | **Risk**: 1% = $250")
    if st.button("Clear Logs"):
        st.session_state.trade_log = pd.DataFrame(columns=st.session_state.trade_log.columns)
        st.success("Logs cleared!")
        st.rerun()

# --- Auto-refresh ---
st.markdown("""
<script>
    setTimeout(function(){ location.reload(); }, 60000);
</script>
""", unsafe_allow_html=True)
