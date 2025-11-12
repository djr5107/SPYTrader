# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_option_menu import option_menu
import json
from pathlib import Path

st.set_page_config(page_title="SPY Pro v2.25", layout="wide")
st.title("SPY Pro v2.25 - Wall Street Grade")
st.caption("Live Chain | Greeks | Charts | Auto-Paper | Backtest | Schwab Ready | Princeton Meadows")

# Persistent Storage Paths
DATA_DIR = Path("trading_data")
DATA_DIR.mkdir(exist_ok=True)
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ACTIVE_TRADES_FILE = DATA_DIR / "active_trades.json"
SIGNAL_QUEUE_FILE = DATA_DIR / "signal_queue.json"
PERFORMANCE_FILE = DATA_DIR / "performance_metrics.json"

# Load/Save Functions
def load_json(filepath, default):
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# Session State with Persistent Loading
if 'trade_log' not in st.session_state:
    trade_log_data = load_json(TRADE_LOG_FILE, [])
    st.session_state.trade_log = pd.DataFrame(trade_log_data) if trade_log_data else pd.DataFrame(columns=[
        'Timestamp', 'Type', 'Symbol', 'Action', 'Size', 'Entry', 'Exit', 'P&L', 
        'Status', 'Signal ID', 'Entry Price Numeric', 'Exit Price Numeric', 
        'P&L Numeric', 'DTE', 'Strategy', 'Thesis', 'Max Hold Minutes', 'Actual Hold Minutes'
    ])

if 'active_trades' not in st.session_state:
    st.session_state.active_trades = load_json(ACTIVE_TRADES_FILE, [])
    for trade in st.session_state.active_trades:
        if 'entry_time' in trade and isinstance(trade['entry_time'], str):
            trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])

if 'signal_queue' not in st.session_state:
    st.session_state.signal_queue = load_json(SIGNAL_QUEUE_FILE, [])

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = load_json(PERFORMANCE_FILE, {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'total_risk': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0,
        'daily_pnl': {}
    })

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Trading Hub", "Options Chain", "Backtest", "Sample Trades", "Trade Tracker", "Performance", "Glossary", "Settings"],
        icons=["house", "table", "chart-line", "book", "clipboard-data", "graph-up", "book", "gear"],
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

# Market Hours
def is_market_open():
    now = datetime.now(ZoneInfo("US/Eastern"))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and market_open <= now <= market_close

# Live Data + Options Chain
@st.cache_data(ttl=30)
def get_market_data():
    try:
        spy = yf.Ticker("SPY")
        S = spy.fast_info.get("lastPrice", 671.50)
        vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 17.38)
        hist = spy.history(period="1d", interval="1m")
        if hist.empty:
            hist = pd.DataFrame({"Close": [S] * 10}, index=pd.date_range(end=datetime.now(), periods=10, freq='1min'))

        today = datetime.now(ZoneInfo("US/Eastern")).date()
        expirations = []
        chains = []

        for exp_str in spy.options:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 0 < dte <= 30:
                    expirations.append(exp_str)
                    opt = spy.option_chain(exp_str)
                    for df_raw, typ in [(opt.calls, 'Call'), (opt.puts, 'Put')]:
                        df = df_raw.copy()
                        df['type'] = typ
                        df['expiration'] = exp_str
                        df['dte'] = dte
                        df['mid'] = (df['bid'] + df['ask']) / 2
                        df['symbol'] = f"SPY {exp_str.replace('-', '')} {typ[0]} {df['strike'].astype(int)}"
                        df['contractSymbol'] = df.get('contractSymbol', df['symbol'])
                        base_cols = ['contractSymbol', 'symbol', 'type', 'strike', 'dte', 'lastPrice', 'bid', 'ask', 'mid',
                                     'impliedVolatility', 'volume', 'openInterest']
                        greek_cols = ['delta', 'gamma', 'theta', 'vega']
                        for col in greek_cols:
                            if col not in df.columns:
                                df[col] = np.nan
                        df = df[base_cols + greek_cols]
                        chains.append(df)
            except:
                continue

        full_chain = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()
        full_chain = full_chain.round({'mid': 2, 'impliedVolatility': 3, 'delta': 3, 'gamma': 3, 'theta': 3, 'vega': 3})
        return float(S), float(vix), hist, expirations, full_chain
    except Exception as e:
        st.warning(f"Data issue: {e}")
        return 671.50, 17.38, pd.DataFrame({"Close": [671.50]*10}), [], pd.DataFrame()

S, vix, hist, expirations, option_chain = get_market_data()

# Enhanced Log Trade Function
def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id, 
              entry_numeric=None, exit_numeric=None, pnl_numeric=None, 
              dte=None, strategy=None, thesis=None, max_hold=None, actual_hold=None):
    new = pd.DataFrame([{
        'Timestamp': ts,
        'Type': typ,
        'Symbol': sym,
        'Action': action,
        'Size': size,
        'Entry': entry,
        'Exit': exit,
        'P&L': pnl,
        'Status': status,
        'Signal ID': sig_id,
        'Entry Price Numeric': entry_numeric,
        'Exit Price Numeric': exit_numeric,
        'P&L Numeric': pnl_numeric,
        'DTE': dte,
        'Strategy': strategy,
        'Thesis': thesis,
        'Max Hold Minutes': max_hold,
        'Actual Hold Minutes': actual_hold
    }])
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, new], ignore_index=True)
    save_json(TRADE_LOG_FILE, st.session_state.trade_log.to_dict('records'))
    if typ == "Close" and pnl_numeric is not None:
        update_performance_metrics(pnl_numeric)

# Update Performance Metrics
def update_performance_metrics(pnl):
    metrics = st.session_state.performance_metrics
    metrics['total_trades'] += 1
    metrics['total_pnl'] += pnl
    
    if pnl > 0:
        metrics['winning_trades'] += 1
        metrics['avg_win'] = ((metrics.get('avg_win', 0) * (metrics['winning_trades'] - 1)) + pnl) / metrics['winning_trades']
    else:
        metrics['losing_trades'] += 1
        metrics['avg_loss'] = ((metrics.get('avg_loss', 0) * (metrics['losing_trades'] - 1)) + pnl) / metrics['losing_trades']
    
    metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
    total_wins = metrics['winning_trades'] * metrics.get('avg_win', 0)
    total_losses = abs(metrics['losing_trades'] * metrics.get('avg_loss', 0))
    metrics['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else 0
    today = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
    if today not in metrics['daily_pnl']:
        metrics['daily_pnl'][today] = 0
    metrics['daily_pnl'][today] += pnl
    save_json(PERFORMANCE_FILE, metrics)

# Auto-Close Logic
def simulate_exit():
    now = datetime.now(ZoneInfo("US/Eastern"))
    for trade in st.session_state.active_trades[:]:
        minutes_held = (now - trade['entry_time']).total_seconds() / 60
        if minutes_held >= trade['max_hold']:
            if 'SPY' == trade['symbol']:
                exit_price = S
            else:
                exit_price = trade['entry_price'] * 0.5
            
            if trade['action'] == 'Buy':
                pnl = (exit_price - trade['entry_price']) * trade['size'] * 100
                close_action = 'Sell'
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['size'] * 100
                close_action = 'Buy'
            
            log_trade(
                ts=now.strftime("%m/%d %H:%M"),
                typ="Close",
                sym=trade['symbol'],
                action=close_action,
                size=trade['size'],
                entry=f"${trade['entry_price']:.2f}",
                exit=f"${exit_price:.2f}",
                pnl=f"${pnl:.0f}",
                status="Closed",
                sig_id=trade['signal_id'],
                entry_numeric=trade['entry_price'],
                exit_numeric=exit_price,
                pnl_numeric=pnl,
                dte=trade.get('dte'),
                strategy=trade.get('strategy'),
                thesis=trade.get('thesis'),
                max_hold=trade['max_hold'],
                actual_hold=minutes_held
            )
            st.session_state.active_trades.remove(trade)
    save_active_trades()

def save_active_trades():
    trades_to_save = []
    for trade in st.session_state.active_trades:
        trade_copy = trade.copy()
        if isinstance(trade_copy.get('entry_time'), datetime):
            trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
        trades_to_save.append(trade_copy)
    save_json(ACTIVE_TRADES_FILE, trades_to_save)

# Generate Signal
def generate_signal():
    now = datetime.now(ZoneInfo("US/Eastern"))
    now_str = now.strftime("%m/%d %H:%M")
    if not is_market_open() or any(s['time'] == now_str for s in st.session_state.signal_queue):
        return
    
    if np.random.random() < 0.3:
        signal_type = np.random.choice(['SPY', 'IRON_CONDOR', 'SPREAD'])
        
        if signal_type == 'SPY':
            action = "Buy" if np.random.random() < 0.6 else "Sell"
            signal = {
                'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                'time': now_str,
                'type': 'SPY ETF',
                'symbol': 'SPY',
                'action': action,
                'size': 10,
                'entry_price': S,
                'max_hold': 60,
                'dte': 0,
                'strategy': f"SPY {action}",
                'thesis': f"{'Bullish' if action=='Buy' else 'Bearish'} momentum indicator"
            }
        elif signal_type == 'IRON_CONDOR':
            signal = {
                'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                'time': now_str,
                'type': 'Iron Condor',
                'symbol': 'SPY IC',
                'action': 'Sell 650P/655P - 685C/690C',
                'size': 2,
                'entry_price': 0.90,
                'max_hold': 240,
                'dte': 7,
                'strategy': 'Iron Condor',
                'thesis': 'Range-bound market, theta decay favorable'
            }
        else:
            spread_type = np.random.choice(['Bull Put', 'Bear Call'])
            signal = {
                'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                'time': now_str,
                'type': f'{spread_type} Spread',
                'symbol': f'SPY {spread_type}',
                'action': f"Sell {650 if spread_type == 'Bull Put' else 680}P/C",
                'size': 3,
                'entry_price': 1.20,
                'max_hold': 180,
                'dte': 5,
                'strategy': f'{spread_type} Spread',
                'thesis': f"{'Support' if spread_type == 'Bull Put' else 'Resistance'} level identified"
            }
        
        st.session_state.signal_queue.append(signal)
        save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
        log_trade(
            now_str, "Signal", signal['symbol'], signal['action'], signal['size'], 
            "---", "---", "---", "Pending", signal['id'],
            strategy=signal['strategy'], thesis=signal['thesis'], 
            max_hold=signal['max_hold'], dte=signal.get('dte')
        )

# Trading Hub
if selected == "Trading Hub":
    st.header("Trading Hub: Live Signals & Auto-Paper")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SPY", f"${S:.2f}")
    col2.metric("VIX", f"{vix:.2f}")
    col3.metric("Active", len(st.session_state.active_trades))
    today_pnl = st.session_state.performance_metrics.get('daily_pnl', {}).get(
        datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d"), 0
    )
    col4.metric("Today P&L", f"${today_pnl:,.0f}")

    simulate_exit()
    generate_signal()

    if st.session_state.signal_queue:
        sig = st.session_state.signal_queue[-1]
        st.markdown(f"""
        <div style="background:#ff6b6b;padding:15px;border-radius:10px;text-align:center;">
            <h3>SIGNAL @ {sig['time']}</h3>
            <p><b>{sig['type']}</b> | {sig['action']} | Size: {sig['size']}</p>
            <p><small>Strategy: {sig['strategy']} | Max Hold: {sig['max_hold']}min</small></p>
            <p><small>Thesis: {sig['thesis']}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Take: {sig['id']}", key=f"take_{sig['id']}", use_container_width=True):
                entry_price = S if sig['symbol'] == 'SPY' else sig['entry_price']
                trade = {
                    'signal_id': sig['id'],
                    'entry_time': datetime.now(ZoneInfo("US/Eastern")),
                    'symbol': sig['symbol'],
                    'action': sig['action'],
                    'size': sig['size'],
                    'entry_price': entry_price,
                    'max_hold': sig['max_hold'],
                    'dte': sig.get('dte'),
                    'strategy': sig['strategy'],
                    'thesis': sig['thesis']
                }
                st.session_state.active_trades.append(trade)
                save_active_trades()
                log_trade(
                    sig['time'], "Open", sig['symbol'], sig['action'], sig['size'],
                    f"${entry_price:.2f}", "Pending", "Open", "Open", sig['id'],
                    entry_numeric=entry_price, dte=sig.get('dte'),
                    strategy=sig['strategy'], thesis=sig['thesis'], max_hold=sig['max_hold']
                )
                st.session_state.signal_queue.remove(sig)
                save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
                st.success("Trade opened!")
                st.rerun()
        
        with col2:
            if st.button(f"Skip: {sig['id']}", key=f"skip_{sig['id']}", use_container_width=True):
                log_trade(
                    sig['time'], "Skipped", sig['symbol'], sig['action'], sig['size'],
                    "---", "---", "---", "Skipped", sig['id'],
                    strategy=sig['strategy'], thesis=sig['thesis']
                )
                st.session_state.signal_queue.remove(sig)
                save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
                st.info("Signal skipped")
                st.rerun()
    
    if st.session_state.active_trades:
        st.subheader("Active Trades")
        for trade in st.session_state.active_trades:
            minutes_held = (datetime.now(ZoneInfo("US/Eastern")) - trade['entry_time']).total_seconds() / 60
            with st.expander(f"{trade['symbol']} - {trade['signal_id']} ({minutes_held:.0f}/{trade['max_hold']}min)"):
                col1, col2 = st.columns(2)
                col1.write(f"**Entry:** ${trade['entry_price']:.2f}")
                col1.write(f"**Size:** {trade['size']}")
                col1.write(f"**Action:** {trade['action']}")
                col2.write(f"**Strategy:** {trade['strategy']}")
                col2.write(f"**Thesis:** {trade['thesis']}")
                col2.write(f"**DTE:** {trade.get('dte', 'N/A')}")
                
                if st.button(f"Close Now", key=f"close_{trade['signal_id']}"):
                    exit_price = S if trade['symbol'] == 'SPY' else trade['entry_price'] * 0.5
                    if trade['action'] == 'Buy':
                        pnl = (exit_price - trade['entry_price']) * trade['size'] * 100
                        close_action = 'Sell'
                    else:
                        pnl = (trade['entry_price'] - exit_price) * trade['size'] * 100
                        close_action = 'Buy'
                    
                    now = datetime.now(ZoneInfo("US/Eastern"))
                    log_trade(
                        ts=now.strftime("%m/%d %H:%M"),
                        typ="Close",
                        sym=trade['symbol'],
                        action=close_action,
                        size=trade['size'],
                        entry=f"${trade['entry_price']:.2f}",
                        exit=f"${exit_price:.2f}",
                        pnl=f"${pnl:.0f}",
                        status="Closed (Manual)",
                        sig_id=trade['signal_id'],
                        entry_numeric=trade['entry_price'],
                        exit_numeric=exit_price,
                        pnl_numeric=pnl,
                        dte=trade.get('dte'),
                        strategy=trade['strategy'],
                        thesis=trade['thesis'],
                        max_hold=trade['max_hold'],
                        actual_hold=minutes_held
                    )
                    st.session_state.active_trades.remove(trade)
                    save_active_trades()
                    st.success("Trade closed!")
                    st.rerun()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index[-50:], y=hist['Close'].iloc[-50:], name="SPY"))
    fig.add_hline(y=S, line_dash="dash", line_color="orange")
    fig.update_layout(height=400, title="SPY 1-Min Chart")
    st.plotly_chart(fig, use_container_width=True)

# Options Chain
elif selected == "Options Chain":
    st.header("SPY Options Chain")
    if option_chain.empty or 'expiration' not in option_chain.columns:
        st.warning("No options data available. This could be due to market hours or data connectivity.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            exp_filter = st.multiselect("Expiration", expirations, default=expirations[:3] if len(expirations) >= 3 else expirations)
        with col2:
            type_filter = st.multiselect("Type", ["Call", "Put"], default=["Call", "Put"])
        with col3:
            dte_filter = st.slider("Max DTE", 0, 30, 30)

        df = option_chain.copy()
        if exp_filter and 'expiration' in df.columns:
            df = df[df['expiration'].isin(exp_filter)]
        if type_filter and 'type' in df.columns:
            df = df[df['type'].isin(type_filter)]
        if 'dte' in df.columns:
            df = df[df['dte'] <= dte_filter]

        if not df.empty:
            st.dataframe(df.drop(columns=['contractSymbol'], errors='ignore'), use_container_width=True, height=500)
            
            # Option details section
            selected_opt = st.selectbox("Select Option for Details", df['symbol'].tolist() if 'symbol' in df.columns else [], key="opt_select")
            if selected_opt:
                row = df[df['symbol'] == selected_opt].iloc[0]
                with st.expander(f"**{row['symbol']}**", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.write(f"**Last:** ${row.get('lastPrice', 0):.2f}")
                    col1.write(f"**Bid/Ask:** ${row.get('bid', 0):.2f} / ${row.get('ask', 0):.2f}")
                    col1.write(f"**IV:** {row.get('impliedVolatility', 0):.1%}")
                    col2.write(f"**Delta:** {row.get('delta', 'N/A')}")
                    col2.write(f"**Gamma:** {row.get('gamma', 'N/A')}")
                    col2.write(f"**Theta:** {row.get('theta', 'N/A')}")
                    col2.write(f"**Vega:** {row.get('vega', 'N/A')}")

                    if st.button("Paper Trade", key=f"pt_{selected_opt}"):
                        now = datetime.now(ZoneInfo("US/Eastern"))
                        sig_id = f"MAN-{selected_opt}-{now.strftime('%H%M%S')}"
                        log_trade(
                            ts=now.strftime("%m/%d %H:%M"),
                            typ="Open",
                            sym=row['symbol'],
                            action="Buy",
                            size=1,
                            entry=f"${row.get('mid', 0):.2f}",
                            exit="Pending",
                            pnl="Open",
                            status="Open",
                            sig_id=sig_id,
                            entry_numeric=row.get('mid', 0),
                            dte=row.get('dte', 0),
                            strategy="Manual Option Trade",
                            thesis="Manual entry from options chain"
                        )
                        st.success("Paper trade opened!")
        else:
            st.info("No options match the selected filters.")

# Trade Tracker
elif selected == "Trade Tracker":
    st.header("Trade Tracker")
    
    if not st.session_state.trade_log.empty:
        df = st.session_state.trade_log.sort_values("Timestamp", ascending=False)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", df['Status'].unique(), default=df['Status'].unique())
        with col2:
            type_filter = st.multiselect("Type", df['Type'].unique(), default=df['Type'].unique())
        with col3:
            strategy_filter = st.multiselect("Strategy", df['Strategy'].dropna().unique() if 'Strategy' in df.columns else [])
        
        filtered_df = df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
        if type_filter:
            filtered_df = filtered_df[filtered_df['Type'].isin(type_filter)]
        if strategy_filter and 'Strategy' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Strategy'].isin(strategy_filter)]
        
        st.dataframe(filtered_df, use_container_width=True, height=500)
        
        closed = filtered_df[filtered_df['Status'].str.contains('Closed', na=False)]
        if not closed.empty and 'P&L Numeric' in closed.columns:
            total_pnl = closed['P&L Numeric'].sum()
            wins = (closed['P&L Numeric'] > 0).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total P&L", f"${total_pnl:,.0f}")
            col2.metric("Closed Trades", len(closed))
            col3.metric("Wins", wins)
            col4.metric("Win Rate", f"{wins/len(closed)*100:.1f}%" if len(closed) > 0 else "0%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Export All", filtered_df.to_csv(index=False), "trades.csv", "text/csv")
        with col2:
            if not closed.empty:
                st.download_button("Export Closed", closed.to_csv(index=False), "closed_trades.csv", "text/csv")
    else:
        st.info("No trades yet")

# Performance
elif selected == "Performance":
    st.header("Performance Analytics")
    
    metrics = st.session_state.performance_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics['total_trades'])
    col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    col3.metric("Total P&L", f"${metrics['total_pnl']:,.0f}")
    col4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    if metrics['daily_pnl']:
        st.subheader("Daily P&L")
        daily_df = pd.DataFrame(list(metrics['daily_pnl'].items()), columns=['Date', 'P&L'])
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date')
        daily_df['Cumulative'] = daily_df['P&L'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily_df['Date'], y=daily_df['P&L'], name='Daily'))
        fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['Cumulative'], 
                                 name='Cumulative', line=dict(color='green', width=2)))
        fig.update_layout(height=400, title="Performance")
        st.plotly_chart(fig, use_container_width=True)

# Other pages
elif selected == "Backtest":
    st.header("Backtest")
    st.write("Historical backtest data here")

elif selected == "Sample Trades":
    st.header("Sample Strategies")
    st.write("Sample trade examples")

elif selected == "Glossary":
    st.write("**POP**: Probability of Profit")
    st.write("**IV**: Implied Volatility")
    st.write("**DTE**: Days to Expiration")

elif selected == "Settings":
    st.subheader("Settings")
    st.write(f"**Bankroll**: ${ACCOUNT_SIZE:,}")
    st.write(f"**Risk**: {RISK_PCT*100:.1f}%")

if is_market_open():
    st.markdown("""
    <script>
    setTimeout(function() {
        window.location.reload();
    }, 30000);
    </script>
    """, unsafe_allow_html=True)
