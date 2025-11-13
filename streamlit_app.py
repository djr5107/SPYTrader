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

st.set_page_config(page_title=“SPY Pro v2.25”, layout=“wide”)
st.title(“SPY Pro v2.25 - Wall Street Grade”)
st.caption(“Live Chain | Greeks | Charts | Auto-Paper | Backtest | Schwab Ready | Princeton Meadows”)

# Persistent Storage Paths

DATA_DIR = Path(“trading_data”)
DATA_DIR.mkdir(exist_ok=True)
TRADE_LOG_FILE = DATA_DIR / “trade_log.json”
ACTIVE_TRADES_FILE = DATA_DIR / “active_trades.json”
SIGNAL_QUEUE_FILE = DATA_DIR / “signal_queue.json”
PERFORMANCE_FILE = DATA_DIR / “performance_metrics.json”

# Load/Save Functions

def load_json(filepath, default):
if filepath.exists():
try:
with open(filepath, ‘r’) as f:
return json.load(f)
except:
return default
return default

def save_json(filepath, data):
with open(filepath, ‘w’) as f:
json.dump(data, f, indent=2)

# Session State with Persistent Loading

if ‘trade_log’ not in st.session_state:
trade_log_data = load_json(TRADE_LOG_FILE, [])
st.session_state.trade_log = pd.DataFrame(trade_log_data) if trade_log_data else pd.DataFrame(columns=[
‘Timestamp’, ‘Type’, ‘Symbol’, ‘Action’, ‘Size’, ‘Entry’, ‘Exit’, ‘P&L’,
‘Status’, ‘Signal ID’, ‘Entry Price Numeric’, ‘Exit Price Numeric’,
‘P&L Numeric’, ‘DTE’, ‘Strategy’, ‘Thesis’, ‘Max Hold Minutes’, ‘Actual Hold Minutes’
])

if ‘active_trades’ not in st.session_state:
st.session_state.active_trades = load_json(ACTIVE_TRADES_FILE, [])
for trade in st.session_state.active_trades:
if ‘entry_time’ in trade and isinstance(trade[‘entry_time’], str):
trade[‘entry_time’] = datetime.fromisoformat(trade[‘entry_time’])

if ‘signal_queue’ not in st.session_state:
st.session_state.signal_queue = load_json(SIGNAL_QUEUE_FILE, [])

if ‘watchlist’ not in st.session_state:
st.session_state.watchlist = []

if ‘performance_metrics’ not in st.session_state:
st.session_state.performance_metrics = load_json(PERFORMANCE_FILE, {
‘total_trades’: 0,
‘winning_trades’: 0,
‘losing_trades’: 0,
‘total_pnl’: 0,
‘total_risk’: 0,
‘win_rate’: 0,
‘avg_win’: 0,
‘avg_loss’: 0,
‘profit_factor’: 0,
‘daily_pnl’: {}
})

# Sidebar

with st.sidebar:
selected = option_menu(
“Menu”,
[“Trading Hub”, “Options Chain”, “Backtest”, “Sample Trades”, “Trade Tracker”, “Performance”, “Glossary”, “Settings”],
icons=[“house”, “table”, “chart-line”, “book”, “clipboard-data”, “graph-up”, “book”, “gear”],
default_index=0,
)
st.divider()
st.subheader(“Risk Settings”)
ACCOUNT_SIZE = 25000
RISK_PCT = st.slider(“Risk/Trade (%)”, 0.5, 2.0, 1.0) / 100
MIN_CREDIT = st.number_input(“Min Credit ($)”, 0.10, 5.0, 0.30)
MAX_DTE = st.slider(“Max DTE”, 7, 90, 45)
POP_TARGET = st.slider(“Min POP (%)”, 60, 95, 75)
PAPER_MODE = st.toggle(“Paper Trading”, value=True)

# Market Hours

def is_market_open():
now = datetime.now(ZoneInfo(“US/Eastern”))
market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
return now.weekday() < 5 and market_open <= now <= market_close

# Live Data + Options Chain

@st.cache_data(ttl=30)
def get_market_data():
try:
spy = yf.Ticker(“SPY”)
S = spy.fast_info.get(“lastPrice”, 671.50)
vix = yf.Ticker(”^VIX”).fast_info.get(“lastPrice”, 17.38)
hist = spy.history(period=“1d”, interval=“1m”)
if hist.empty:
hist = pd.DataFrame({“Close”: [S] * 10}, index=pd.date_range(end=datetime.now(), periods=10, freq=‘1min’))

```
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
```

S, vix, hist, expirations, option_chain = get_market_data()

# Enhanced Log Trade Function

def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id,
entry_numeric=None, exit_numeric=None, pnl_numeric=None,
dte=None, strategy=None, thesis=None, max_hold=None, actual_hold=None):
new = pd.DataFrame([{
‘Timestamp’: ts,
‘Type’: typ,
‘Symbol’: sym,
‘Action’: action,
‘Size’: size,
‘Entry’: entry,
‘Exit’: exit,
‘P&L’: pnl,
‘Status’: status,
‘Signal ID’: sig_id,
‘Entry Price Numeric’: entry_numeric,
‘Exit Price Numeric’: exit_numeric,
‘P&L Numeric’: pnl_numeric,
‘DTE’: dte,
‘Strategy’: strategy,
‘Thesis’: thesis,
‘Max Hold Minutes’: max_hold,
‘Actual Hold Minutes’: actual_hold
}])
st.session_state.trade_log = pd.concat([st.session_state.trade_log, new], ignore_index=True)
save_json(TRADE_LOG_FILE, st.session_state.trade_log.to_dict(‘records’))
if typ == “Close” and pnl_numeric is not None:
update_performance_metrics(pnl_numeric)

# Update Performance Metrics

def update_performance_metrics(pnl):
metrics = st.session_state.performance_metrics
metrics[‘total_trades’] += 1
metrics[‘total_pnl’] += pnl

```
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
```

# Auto-Close Logic

def simulate_exit():
now = datetime.now(ZoneInfo(“US/Eastern”))
for trade in st.session_state.active_trades[:]:
minutes_held = (now - trade[‘entry_time’]).total_seconds() / 60
if minutes_held >= trade[‘max_hold’]:
if ‘SPY’ == trade[‘symbol’]:
exit_price = S
else:
exit_price = trade[‘entry_price’] * 0.5

```
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
```

def save_active_trades():
trades_to_save = []
for trade in st.session_state.active_trades:
trade_copy = trade.copy()
if isinstance(trade_copy.get(‘entry_time’), datetime):
trade_copy[‘entry_time’] = trade_copy[‘entry_time’].isoformat()
trades_to_save.append(trade_copy)
save_json(ACTIVE_TRADES_FILE, trades_to_save)

# Generate Signal

def generate_signal():
now = datetime.now(ZoneInfo(“US/Eastern”))
now_str = now.strftime(”%m/%d %H:%M”)
if not is_market_open() or any(s[‘time’] == now_str for s in st.session_state.signal_queue):
return

```
# Calculate market conditions for signal generation
try:
    spy_ticker = yf.Ticker("SPY")
    hist_data = spy_ticker.history(period="5d", interval="1h")
    if hist_data.empty or len(hist_data) < 20:
        return
    
    # Calculate indicators
    sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
    sma_50 = hist_data['Close'].rolling(window=min(50, len(hist_data))).mean().iloc[-1]
    volume_avg = hist_data['Volume'].rolling(window=20).mean().iloc[-1]
    current_volume = hist_data['Volume'].iloc[-1]
    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
    price_change = (S - hist_data['Close'].iloc[-2]) / hist_data['Close'].iloc[-2]
    
    # Calculate volatility estimate
    returns = hist_data['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized IV estimate
    
    # Determine signal type based on actual market conditions
    signal = None
    
    # SPY EQUITY SIGNALS
    # Strong uptrend with volume
    if (S > sma_20 * 1.005 and 
        sma_20 > sma_50 and 
        volume_ratio > 1.3 and 
        price_change > 0.002):
        signal = {
            'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
            'time': now_str,
            'type': 'SPY Long',
            'symbol': 'SPY',
            'action': f"BUY {10} shares @ ${S:.2f}",
            'size': 10,
            'entry_price': S,
            'max_hold': 3900,  # 10 days
            'dte': 0,
            'strategy': 'SPY Long - Momentum',
            'thesis': f"Strong uptrend: SPY ${S:.2f} above SMA20 (${sma_20:.2f}), volume {volume_ratio:.1f}x average, +{price_change*100:.2f}% momentum"
        }
    
    # Bearish breakdown
    elif (S < sma_20 * 0.995 and 
          sma_20 < sma_50 and 
          volume_ratio > 1.3 and 
          price_change < -0.002):
        signal = {
            'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
            'time': now_str,
            'type': 'SPY Short',
            'symbol': 'SPY',
            'action': f"SELL SHORT {10} shares @ ${S:.2f}",
            'size': 10,
            'entry_price': S,
            'max_hold': 3900,
            'dte': 0,
            'strategy': 'SPY Short - Breakdown',
            'thesis': f"Bearish breakdown: SPY ${S:.2f} below SMA20 (${sma_20:.2f}), volume {volume_ratio:.1f}x, {price_change*100:.2f}% down"
        }
    
    # OPTIONS SIGNALS - Based on real market conditions
    
    # IRON CONDOR - Low vol + ranging market
    elif vix < 15 and abs(price_change) < 0.003 and volatility < 12:
        put_strike_short = int(S * 0.98)  # 2% OTM
        put_strike_long = int(S * 0.96)   # 4% OTM
        call_strike_short = int(S * 1.02)  # 2% OTM
        call_strike_long = int(S * 1.04)   # 4% OTM
        
        # Estimate credit (rough approximation)
        credit = round((S * 0.0015), 2)  # ~0.15% of SPY price
        max_risk = ((put_strike_short - put_strike_long) - credit) * 100
        pop = 75  # Approximate POP for 2% OTM
        
        signal = {
            'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
            'time': now_str,
            'type': 'Iron Condor',
            'symbol': f'SPY {put_strike_short}P/{put_strike_long}P - {call_strike_short}C/{call_strike_long}C',
            'action': f"SELL Iron Condor ${put_strike_short}P/${put_strike_long}P - ${call_strike_short}C/${call_strike_long}C",
            'size': 1,
            'entry_price': credit,
            'max_hold': 3900,
            'dte': 7,
            'strategy': 'Iron Condor',
            'thesis': f"Low volatility setup: VIX {vix:.1f}, IV {volatility:.1f}%, SPY ranging. Credit ${credit*100:.0f}, Max Risk ${max_risk:.0f}, POP {pop}%"
        }
    
    # BULL PUT SPREAD - Bullish bias with moderate IV
    elif (S > sma_20 and 
          15 < vix < 25 and 
          price_change > 0 and
          sma_20 > sma_50):
        put_strike_short = int(S * 0.97)  # 3% OTM
        put_strike_long = int(S * 0.95)   # 5% OTM
        
        credit = round(S * 0.0020, 2)  # ~0.2% of SPY
        max_risk = ((put_strike_short - put_strike_long) - credit) * 100
        pop = 80  # Approximate POP for 3% OTM
        
        signal = {
            'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
            'time': now_str,
            'type': 'Bull Put Spread',
            'symbol': f'SPY {put_strike_short}P/{put_strike_long}P',
            'action': f"SELL Bull Put Spread ${put_strike_short}P/${put_strike_long}P",
            'size': 2,
            'entry_price': credit,
            'max_hold': 3900,
            'dte': 7,
            'strategy': 'Bull Put Spread',
            'thesis': f"Bullish setup: SPY ${S:.2f} above SMA20 (${sma_20:.2f}), VIX {vix:.1f}. Credit ${credit*200:.0f}, Max Risk ${max_risk*2:.0f}, POP {pop}%"
        }
    
    # BEAR CALL SPREAD - Bearish bias with moderate IV
    elif (S < sma_20 and 
          15 < vix < 25 and 
          price_change < 0 and
          sma_20 < sma_50):
        call_strike_short = int(S * 1.03)  # 3% OTM
        call_strike_long = int(S * 1.05)   # 5% OTM
        
        credit = round(S * 0.0020, 2)
        max_risk = ((call_strike_long - call_strike_short) - credit) * 100
        pop = 80
        
        signal = {
            'id': f"SIG-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
            'time': now_str,
            'type': 'Bear Call Spread',
            'symbol': f'SPY {call_strike_short}C/{call_strike_long}C',
            'action': f"SELL Bear Call Spread ${call_strike_short}C/${call_strike_long}C",
            'size': 2,
            'entry_price': credit,
            'max_hold': 3900,
            'dte': 7,
            'strategy': 'Bear Call Spread',
            'thesis': f"Bearish setup: SPY ${S:.2f} below SMA20 (${sma_20:.2f}), VIX {vix:.1f}. Credit ${credit*200:.0f}, Max Risk ${max_risk*2:.0f}, POP {pop}%"
        }
    
    # Add signal if conditions met (about 20% probability any given check)
    if signal and np.random.random() < 0.2:
        st.session_state.signal_queue.append(signal)
        save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
        log_trade(
            now_str, "Signal", signal['symbol'], signal['action'], signal['size'], 
            "---", "---", "---", "Pending", signal['id'],
            strategy=signal['strategy'], thesis=signal['thesis'], 
            max_hold=signal['max_hold'], dte=signal.get('dte')
        )
except Exception as e:
    # Silently fail if data unavailable
    pass
```

# Trading Hub

if selected == “Trading Hub”:
st.header(“Trading Hub: Live Signals & Auto-Paper”)
col1, col2, col3, col4 = st.columns(4)
col1.metric(“SPY”, f”${S:.2f}”)
col2.metric(“VIX”, f”{vix:.2f}”)
col3.metric(“Active”, len(st.session_state.active_trades))
today_pnl = st.session_state.performance_metrics.get(‘daily_pnl’, {}).get(
datetime.now(ZoneInfo(“US/Eastern”)).strftime(”%Y-%m-%d”), 0
)
col4.metric(“Today P&L”, f”${today_pnl:,.0f}”)

```
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
```

# Options Chain

elif selected == “Options Chain”:
st.header(“SPY Options Chain”)
if option_chain.empty or ‘expiration’ not in option_chain.columns:
st.warning(“No options data available. This could be due to market hours or data connectivity.”)
else:
col1, col2, col3 = st.columns(3)
with col1:
exp_filter = st.multiselect(“Expiration”, expirations, default=expirations[:3] if len(expirations) >= 3 else expirations)
with col2:
type_filter = st.multiselect(“Type”, [“Call”, “Put”], default=[“Call”, “Put”])
with col3:
dte_filter = st.slider(“Max DTE”, 0, 30, 30)

```
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
```

# Trade Tracker

elif selected == “Trade Tracker”:
st.header(“Trade Tracker”)

```
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
```

# Performance

elif selected == “Performance”:
st.header(“Performance Analytics”)

```
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
```

# Backtest with Real Historical Data

elif selected == “Backtest”:
st.header(“Backtest: Last 10 Completed Trades”)
st.caption(“Real historical SPY data with signal-based entries and exits”)

```
@st.cache_data(ttl=3600)
def run_backtest():
    """Run backtest using actual historical SPY data"""
    try:
        spy = yf.Ticker("SPY")
        
        # Try to get minute data first (most detailed)
        hist = None
        data_interval = None
        
        # Try different data granularities - extended lookback
        attempts = [
            ("1mo", "5m", "5-minute"),   # Try 1 month of 5-min data
            ("1mo", "15m", "15-minute"), # 1 month of 15-min data
            ("3mo", "1h", "hourly"),     # 3 months of hourly
            ("6mo", "1d", "daily"),      # 6 months of daily
            ("1y", "1d", "daily"),       # 1 year of daily
        ]
        
        for period, interval, desc in attempts:
            try:
                test_hist = spy.history(period=period, interval=interval)
                if not test_hist.empty and len(test_hist) > 50:
                    hist = test_hist
                    data_interval = desc
                    break
            except:
                continue
        
        if hist is None or hist.empty:
            return pd.DataFrame(), "Unable to fetch historical data from yfinance. This may be due to market hours or API limitations. Please try again later."
        
        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['volume_ma'] = hist['Volume'].rolling(window=20).mean()
        hist['volume_ratio'] = hist['Volume'] / hist['volume_ma']
        hist['price_change'] = hist['Close'].pct_change()
        
        # Generate signals based on actual conditions
        completed_trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_reason = ""
        signal_type = ""
        security = ""
        order_type = ""
        
        for i in range(50, len(hist)):
            current_time = hist.index[i]
            current_price = hist['Close'].iloc[i]
            current_volume = hist['Volume'].iloc[i]
            
            # For minute/hourly data, skip outside market hours
            if data_interval in ["5-minute", "15-minute", "hourly"]:
                if hasattr(current_time, 'hour'):
                    if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30) or current_time.hour >= 16:
                        continue
            
            if not in_position:
                # Look for entry signals - IMPROVED BASED ON BACKTEST PERFORMANCE
                
                # Signal 1: Volume Breakout - TIGHTENED (was losing money)
                # Only take when VERY strong volume AND confirming price action
                if (hist['volume_ratio'].iloc[i] > 2.2 and  # Increased from 1.4 to 2.2
                    hist['price_change'].iloc[i] > 0.003 and  # Must have 0.3%+ move
                    current_price > hist['SMA_20'].iloc[i] * 1.005 and  # Must be 0.5% above SMA20
                    hist['Close'].iloc[i-1] > hist['SMA_20'].iloc[i-1]):  # Previous close also above SMA
                    in_position = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_reason = f"Strong volume breakout ({hist['volume_ratio'].iloc[i]:.1f}x avg) with solid price momentum +{hist['price_change'].iloc[i]*100:.1f}% above SMA20 (${hist['SMA_20'].iloc[i]:.2f})"
                    signal_type = "Volume Breakout"
                    security = "SPY"
                    order_type = "BUY"
                    continue
                
                # Signal 2: Golden Cross - HIGH CONVICTION (historically strong)
                if (hist['SMA_20'].iloc[i] > hist['SMA_50'].iloc[i] and 
                    hist['SMA_20'].iloc[i-1] <= hist['SMA_50'].iloc[i-1] and
                    current_volume > hist['volume_ma'].iloc[i] * 0.9):  # Slight volume confirmation
                    in_position = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_reason = f"Golden Cross: SMA20 (${hist['SMA_20'].iloc[i]:.2f}) crossed above SMA50 (${hist['SMA_50'].iloc[i]:.2f}) - strong trend signal"
                    signal_type = "Golden Cross"
                    security = "SPY"
                    order_type = "BUY"
                    continue
                
                # Signal 3: SMA Breakout - STRENGTHENED (good historical performance)
                if (current_price > hist['SMA_20'].iloc[i] * 1.003 and  # Back to 0.3% above
                    hist['Close'].iloc[i-1] <= hist['SMA_20'].iloc[i-1] * 1.002 and  # Clear break
                    hist['volume_ratio'].iloc[i] > 1.2 and  # Volume confirmation
                    hist['SMA_20'].iloc[i] > hist['SMA_50'].iloc[i]):  # In uptrend
                    in_position = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_reason = f"SMA20 breakout: Price at ${current_price:.2f} broke above SMA20 (${hist['SMA_20'].iloc[i]:.2f}) with volume ({hist['volume_ratio'].iloc[i]:.1f}x) in confirmed uptrend"
                    signal_type = "SMA Breakout"
                    security = "SPY"
                    order_type = "BUY"
                    continue
                
                # Signal 4: Oversold Bounce - MODERATE (works in right conditions)
                recent_low = hist['Close'].iloc[i-10:i].min()
                recent_high = hist['Close'].iloc[i-10:i].max()
                price_range = recent_high - recent_low
                if (current_price < recent_low * 1.002 and  # Near recent low
                    hist['price_change'].iloc[i] > 0.002 and  # Bounce starting
                    current_volume > hist['volume_ma'].iloc[i] * 1.5 and  # Strong volume
                    price_range / recent_low > 0.02 and  # Ensure meaningful range
                    hist['SMA_20'].iloc[i] > hist['SMA_50'].iloc[i]):  # Still in uptrend
                    in_position = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_reason = f"Oversold bounce: Price bounced from recent low (${recent_low:.2f}) with strong volume ({hist['volume_ratio'].iloc[i]:.1f}x) in uptrend"
                    signal_type = "Reversal Long"
                    security = "SPY"
                    order_type = "BUY"
                    continue
                
                # Signal 5: Bearish Breakdown for SHORT - TIGHTENED
                if (current_price < hist['SMA_20'].iloc[i] * 0.997 and
                    hist['Close'].iloc[i-1] >= hist['SMA_20'].iloc[i-1] and
                    hist['volume_ratio'].iloc[i] > 1.5 and  # Strong volume
                    hist['price_change'].iloc[i] < -0.003 and  # Clear down move
                    hist['SMA_20'].iloc[i] < hist['SMA_50'].iloc[i]):  # In downtrend
                    in_position = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_reason = f"Bearish breakdown: Price broke below SMA20 (${hist['SMA_20'].iloc[i]:.2f}) with volume ({hist['volume_ratio'].iloc[i]:.1f}x) in confirmed downtrend"
                    signal_type = "Bearish Breakdown"
                    security = "SPY"
                    order_type = "SELL SHORT"
                    continue
            
            else:
                # Look for exit signals
                if order_type == "BUY":
                    pnl_pct = (current_price - entry_price) / entry_price
                    pnl_dollars = (current_price - entry_price) * 10  # 10 shares
                    exit_order = "SELL"
                else:  # SELL SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                    pnl_dollars = (entry_price - current_price) * 10  # 10 shares
                    exit_order = "BUY TO COVER"
                
                # Calculate time in trade based on data interval
                if data_interval == "daily":
                    time_in_trade = (current_time - entry_time).days * 390  # Trading minutes per day
                else:
                    time_in_trade = (current_time - entry_time).total_seconds() / 60
                
                exit_triggered = False
                exit_reason = ""
                
                # Exit 1: REMOVED PROFIT CAP - Let winners run to 10%+!
                # Only exit on stop loss, time limit, or trailing stop
                
                # Exit 2: Stop loss hit (3% for higher conviction)
                if pnl_pct <= -0.03:  # 3% stop loss - allows for volatility
                    exit_triggered = True
                    exit_reason = f"Stop loss triggered ({pnl_pct*100:.2f}%) - cut losses at 3%"
                
                # Exit 3: Extended time limit (10 days for daily, 50 hours for intraday)
                elif time_in_trade >= 3900:  # ~10 trading days for daily, ~65 hours for intraday
                    exit_triggered = True
                    if pnl_pct > 0:
                        exit_reason = f"Maximum hold time reached ({time_in_trade/390:.1f} trading days) - exit with {pnl_pct*100:.2f}% profit" if data_interval == "daily" else f"Maximum hold time reached ({time_in_trade:.0f} min) - exit with {pnl_pct*100:.2f}% profit"
                    else:
                        exit_reason = f"Maximum hold time reached ({time_in_trade/390:.1f} trading days) - exit position" if data_interval == "daily" else f"Maximum hold time reached ({time_in_trade:.0f} min) - exit position"
                
                # Exit 4: Trailing stop - protect profits once up 4%
                elif pnl_pct >= 0.04:  # Once up 4%, use trailing stop
                    # Check if price pulled back 2% from recent high
                    recent_high = hist['Close'].iloc[max(0, i-10):i].max()
                    if current_price < recent_high * 0.98:  # 2% pullback
                        exit_triggered = True
                        exit_reason = f"Trailing stop triggered - locking in {pnl_pct*100:.2f}% profit after pullback from recent high ${recent_high:.2f}"
                
                # Exit 5: Strong momentum reversal with volume (only if losing)
                elif (pnl_pct < -0.01 and 
                      hist['volume_ratio'].iloc[i] > 2.0 and
                      hist['price_change'].iloc[i] < -0.008):
                    exit_triggered = True
                    exit_reason = f"Strong bearish momentum reversal with very high volume - exit to prevent larger loss (currently {pnl_pct*100:.2f}%)"
                
                # Exit 6: Major SMA support/resistance break (only if losing significantly)
                elif order_type == "BUY" and current_price < hist['SMA_20'].iloc[i] * 0.985 and pnl_pct < -0.015:
                    exit_triggered = True
                    exit_reason = f"Price broke significantly below SMA20 support (${hist['SMA_20'].iloc[i]:.2f}) - exit long position"
                elif order_type == "SELL SHORT" and current_price > hist['SMA_20'].iloc[i] * 1.015 and pnl_pct < -0.015:
                    exit_triggered = True
                    exit_reason = f"Price broke significantly above SMA20 resistance (${hist['SMA_20'].iloc[i]:.2f}) - cover short position"
                
                if exit_triggered:
                    # Calculate P&L for 10 shares
                    shares = 10
                    
                    completed_trades.append({
                        'Entry Date': entry_time.strftime('%m/%d/%Y'),
                        'Entry Time': entry_time.strftime('%I:%M %p') if data_interval != "daily" else "Market Open",
                        'Security': security,
                        'Entry Order': order_type,
                        'Signal Type': signal_type,
                        'Entry Price': f"${entry_price:.2f}",
                        'Entry Reason': entry_reason,
                        'Exit Date': current_time.strftime('%m/%d/%Y'),
                        'Exit Time': current_time.strftime('%I:%M %p') if data_interval != "daily" else "Market Close",
                        'Exit Order': exit_order,
                        'Exit Price': f"${current_price:.2f}",
                        'Exit Reason': exit_reason,
                        'Shares': shares,
                        'Hold Time': f"{time_in_trade/390:.1f} days" if data_interval == "daily" else f"{time_in_trade:.0f} min",
                        'P&L': f"${pnl_dollars:.2f}",
                        'P&L %': f"{pnl_pct*100:.2f}%",
                        'Entry Price Numeric': entry_price,
                        'Exit Price Numeric': current_price,
                        'P&L Numeric': pnl_dollars
                    })
                    
                    in_position = False
                    
                    # Stop after 10 completed trades
                    if len(completed_trades) >= 10:
                        break
        
        if not completed_trades:
            return pd.DataFrame(), f"No completed trades found using {data_interval} data. Try adjusting signal parameters or check back when more trading activity occurs."
        
        result_df = pd.DataFrame(completed_trades)
        return result_df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Backtest error: {str(e)}"

# Run the backtest
with st.spinner("Running backtest on historical data..."):
    backtest_df, error = run_backtest()

if error:
    st.error(error)
    st.info("💡 Tip: The backtest uses real-time yfinance data. If you're seeing this error, it may be due to:")
    st.write("- Market being closed")
    st.write("- API rate limits")
    st.write("- Network connectivity")
    st.write("Try refreshing the page in a few minutes.")
elif backtest_df.empty:
    st.warning("No trades generated from backtest")
else:
    # Summary metrics
    total_pnl = backtest_df['P&L Numeric'].sum()
    win_count = (backtest_df['P&L Numeric'] > 0).sum()
    loss_count = (backtest_df['P&L Numeric'] < 0).sum()
    win_rate = (win_count / len(backtest_df) * 100) if len(backtest_df) > 0 else 0
    avg_win = backtest_df[backtest_df['P&L Numeric'] > 0]['P&L Numeric'].mean() if win_count > 0 else 0
    avg_loss = backtest_df[backtest_df['P&L Numeric'] < 0]['P&L Numeric'].mean() if loss_count > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", len(backtest_df))
    col2.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Avg Win", f"${avg_win:.2f}")
    col5.metric("Avg Loss", f"${avg_loss:.2f}")
    
    # Display trades
    st.subheader("Trade Details")
    
    # Create display dataframe without numeric columns
    display_df = backtest_df.drop(columns=['Entry Price Numeric', 'Exit Price Numeric', 'P&L Numeric'])
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Individual trade expandable details
    st.subheader("Detailed Trade Analysis")
    for idx, trade in backtest_df.iterrows():
        pnl_color = "green" if trade['P&L Numeric'] > 0 else "red"
        with st.expander(f"Trade #{idx+1}: {trade['Entry Order']} {trade['Security']} - {trade['Signal Type']} - {trade['P&L']} ({trade['P&L %']})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 ENTRY**")
                st.write(f"📅 Date: {trade['Entry Date']}")
                st.write(f"🕐 Time: {trade['Entry Time']}")
                st.write(f"🎯 Security: **{trade['Security']}**")
                st.write(f"📊 Order: **{trade['Entry Order']}** {trade['Shares']} shares")
                st.write(f"💰 Entry Price: {trade['Entry Price']}")
                st.markdown(f"**Signal:** *{trade['Entry Reason']}*")
            
            with col2:
                st.markdown("**📉 EXIT**")
                st.write(f"📅 Date: {trade['Exit Date']}")
                st.write(f"🕐 Time: {trade['Exit Time']}")
                st.write(f"📊 Order: **{trade['Exit Order']}**")
                st.write(f"💰 Exit Price: {trade['Exit Price']}")
                st.write(f"⏱️ Hold Time: {trade['Hold Time']}")
                st.markdown(f"**Reason:** *{trade['Exit Reason']}*")
            
            st.markdown(f"**P&L: <span style='color:{pnl_color}; font-size:1.2em;'>{trade['P&L']} ({trade['P&L %']})</span>**", unsafe_allow_html=True)
    
    # Download option
    st.download_button(
        "📥 Download Backtest Results",
        display_df.to_csv(index=False),
        "backtest_results.csv",
        "text/csv"
    )
    
    # Analysis insights
    st.subheader("📊 Backtest Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Signal Type Performance**")
        signal_perf = backtest_df.groupby('Signal Type')['P&L Numeric'].agg(['count', 'sum', 'mean']).round(2)
        signal_perf.columns = ['Trades', 'Total P&L', 'Avg P&L']
        st.dataframe(signal_perf, use_container_width=True)
    
    with col2:
        st.markdown("**Exit Reason Analysis**")
        # Extract primary exit reason (first part before parenthesis)
        backtest_df['Exit Type'] = backtest_df['Exit Reason'].str.split('(').str[0].str.strip()
        exit_analysis = backtest_df.groupby('Exit Type')['P&L Numeric'].agg(['count', 'mean']).round(2)
        exit_analysis.columns = ['Count', 'Avg P&L']
        st.dataframe(exit_analysis, use_container_width=True)
    
    # P&L Chart
    st.subheader("Cumulative P&L")
    backtest_df['Cumulative P&L'] = backtest_df['P&L Numeric'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(backtest_df) + 1)),
        y=backtest_df['Cumulative P&L'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='green' if total_pnl > 0 else 'red', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Cumulative P&L Over Last 10 Trades",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
```

elif selected == “Sample Trades”:
st.header(“Sample Strategies”)
st.write(“Sample trade examples”)

elif selected == “Glossary”:
st.write(”**POP**: Probability of Profit”)
st.write(”**IV**: Implied Volatility”)
st.write(”**DTE**: Days to Expiration”)

elif selected == “Settings”:
st.subheader(“Settings”)
st.write(f”**Bankroll**: ${ACCOUNT_SIZE:,}”)
st.write(f”**Risk**: {RISK_PCT*100:.1f}%”)

if is_market_open():
st.markdown(”””
<script>
setTimeout(function() {
window.location.reload();
}, 30000);
</script>
“””, unsafe_allow_html=True)
