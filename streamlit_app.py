# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_option_menu import option_menu

st.set_page_config(page_title="SPY Pro v2.20", layout="wide")
st.title("SPY Pro v2.20 – Pro Options Chain + Greeks + Charts")
st.caption("DTE ≤ 30 | Full Greeks | Click-to-Chart | Watchlist | Princeton Meadows")

# --- Session State ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=[
        'Timestamp', 'Type', 'Symbol', 'Action', 'Size', 'Entry Price', 'Exit Price', 'P&L', 'Status', 'Signal ID'
    ])
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = []
if 'signal_queue' not in st.session_state:
    st.session_state.signal_queue = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# --- Sidebar ---
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Trading Hub", "Options Chain", "Backtest", "Sample Trades", "Trade Tracker", "Glossary", "Settings"],
        icons=["house", "table", "chart-line", "book", "clipboard-data", "book", "gear"],
        default_index=1,  # Open on Options Chain
    )

# --- Market Hours ---
def is_market_open():
    now = datetime.now(ZoneInfo("US/Eastern"))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and market_open <= now <= market_close

# --- Live Data + Options Chain (DTE ≤ 30) ---
@st.cache_data(ttl=30)
def get_spy_data():
    try:
        spy = yf.Ticker("SPY")
        S = spy.fast_info.get("lastPrice", 671.50)
        vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 17.38)
        hist = spy.history(period="1d", interval="1m")

        # Get all expirations
        expirations = spy.options
        today = datetime.now(ZoneInfo("US/Eastern")).date()
        near_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if 0 < dte <= 30:
                near_exps.append((exp, dte))

        chains = []
        for exp, dte in near_exps:
            opt = spy.option_chain(exp)
            calls = opt.calls.copy()
            puts = opt.puts.copy()

            # Add metadata
            for df, typ in [(calls, 'Call'), (puts, 'Put')]:
                df['type'] = typ
                df['expiration'] = exp
                df['dte'] = dte
                df['mid'] = (df['bid'] + df['ask']) / 2
                df['symbol'] = f"SPY {exp.replace('-', '')} {typ[0]} {df['strike'].astype(int)}"
                df = df[['symbol', 'type', 'strike', 'dte', 'lastPrice', 'bid', 'ask', 'mid',
                         'impliedVolatility', 'delta', 'gamma', 'theta', 'vega',
                         'volume', 'openInterest']]
                chains.append(df)

        full_chain = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()
        full_chain = full_chain.round({'mid': 2, 'impliedVolatility': 3, 'delta': 3, 'gamma': 3, 'theta': 3, 'vega': 3})
        return S, vix, hist, [e[0] for e in near_exps], full_chain
    except Exception as e:
        st.warning(f"Data error: {e}")
        return 671.50, 17.38, pd.DataFrame(), [], pd.DataFrame()

S, vix, hist, expirations, option_chain = get_spy_data()

# --- Log Trade ---
def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id):
    new = pd.DataFrame([{
        'Timestamp': ts,
        'Type': typ,
        'Symbol': sym,
        'Action': action,
        'Size': size,
        'Entry Price': entry,
        'Exit Price': exit,
        'P&L': pnl,
        'Status': status,
        'Signal ID': sig_id
    }])
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, new], ignore_index=True)

# --- OPTIONS CHAIN PAGE ---
if selected == "Options Chain":
    st.header("SPY Options Chain – DTE ≤ 30 | Full Greeks | Click for Chart")

    if option_chain.empty:
        st.warning("No options within 30 DTE. Market may be closed.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            exp_filter = st.multiselect("Expiration", expirations, default=expirations)
        with col2:
            type_filter = st.multiselect("Type", ["Call", "Put"], default=["Call", "Put"])
        with col3:
            dte_filter = st.slider("Max DTE", 0, 30, 30)

        # Apply filters
        df = option_chain.copy()
        df = df[df['expiration'].isin(exp_filter)]
        df = df[df['type'].isin(type_filter)]
        df = df[df['dte'] <= dte_filter]

        # Watchlist
        if st.session_state.watchlist:
            st.subheader("Watchlist")
            watch_df = pd.DataFrame(st.session_state.watchlist)
            st.dataframe(watch_df, use_container_width=True)

        # Main Table
        st.subheader(f"Options: {len(df)}")
        for idx, row in df.iterrows():
            key = row['symbol']
            with st.expander(f"**{row['type']} {row['strike']} | DTE {row['dte']} | Mid ${row['mid']:.2f}**", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**Last:** ${row['lastPrice']:.2f}")
                col2.write(f"**IV:** {row['impliedVolatility']:.1%}")
                col3.write(f"**Delta:** {row['delta']:.3f}")
                col4.write(f"**Theta:** {row['theta']:.3f}")

                col1.write(f"**Bid/Ask:** ${row['bid']:.2f} / ${row['ask']:.2f}")
                col2.write(f"**Gamma:** {row['gamma']:.3f}")
                col3.write(f"**Vega:** {row['vega']:.3f}")
                col4.write(f"**OI:** {row['openInterest']:,} | Vol: {row['volume']:,}")

                # Buttons
                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    if st.button("Paper Trade", key=f"pt_{key}"):
                        log_trade(
                            ts=datetime.now(ZoneInfo("US/Eastern")).strftime("%m/%d %H:%M"),
                            typ="Open",
                            sym=key,
                            action="Buy",
                            size=1,
                            entry=f"${row['mid']:.2f}",
                            exit="Pending",
                            pnl="Open",
                            status="Open",
                            sig_id=f"MAN-{idx}"
                        )
                        st.success(f"Paper trade: {key}")
                with bcol2:
                    if st.button("Add to Watchlist", key=f"wl_{key}"):
                        st.session_state.watchlist.append({
                            'Type': row['type'], 'Strike': row['strike'],
                            'Exp': row['expiration'], 'DTE': row['dte'], 'Mid': row['mid']
                        })
                        st.rerun()
                with bcol3:
                    if st.button("View Chart", key=f"ch_{key}"):
                        with st.spinner("Loading price history..."):
                            try:
                                opt_ticker = yf.Ticker(key.replace(" ", ""))
                                opt_hist = opt_ticker.history(period="30d")
                                if not opt_hist.empty:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=opt_hist.index, y=opt_hist['Close'],
                                                             mode='lines+markers', name='Price'))
                                    fig.update_layout(title=f"{key} – 30-Day Price", height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No price history.")
                            except:
                                st.error("Chart unavailable.")

# --- Trading Hub (unchanged) ---
elif selected == "Trading Hub":
    st.header("Trading Hub")
    col1, col2 = st.columns(2)
    col1.metric("SPY", f"${S:.2f}")
    col2.metric("VIX", f"{vix:.2f}")
    # [Signal logic unchanged]

# --- Backtest, Tracker, etc. (unchanged) ---
elif selected == "Backtest":
    st.header("Backtest: 25 Trades")
    # [Same]

elif selected == "Trade Tracker":
    st.header("Trade Tracker")
    if not st.session_state.trade_log.empty:
        df = st.session_state.trade_log.sort_values("Timestamp", ascending=False)
        st.dataframe(df)
        st.download_button("Export", df.to_csv(index=False), "log.csv")
    else:
        st.info("No trades.")

# --- Auto-refresh ---
st.markdown("""
<script>
    setTimeout(() => location.reload(), 30000);
</script>
""", unsafe_allow_html=True)
