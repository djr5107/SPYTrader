# streamlit_app_COMPLETE.py - SPY Pro v3.0 COMPLETE
# Full featured trading system with backtest, live trading, signal history, and persistent storage

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_option_menu import option_menu
import json
from pathlib import Path
from plotly.subplots import make_subplots

# V3.0 COMPLETE FEATURES:
# ✅ Enhanced signal generation (more active)
# ✅ Persistent timestamps (no reset on refresh)
# ✅ Signal history tracking
# ✅ Options signals
# ✅ Complete backtest system
# ✅ Robust data persistence
# ✅ Live trading paper/real mode
# ✅ Performance analytics

from macro_analysis_CORRECTED import (
    MacroAnalyzer,
    calculate_dynamic_stop_loss,
    calculate_trailing_stop,
    check_exit_conditions,
    detect_drawdown_regime,
    calculate_recovery_speed,
    adjust_signal_for_macro,
    should_take_signal,
    calculate_position_size
)

# ============================================================
# PREMIUM SELLER ENGINE
# Black-Scholes-Merton + volatility skew + Student-t fat-tail
# probability, combined IV/RV edge score, EV from live mid prices.
# ============================================================
from scipy.stats import norm, t as student_t
import math as _math

PS_DF_BASE = 4          # Student-t degrees of freedom (fatter tails than normal)

def ps_bsm_put(S, K, T, r, sigma):
    """Black-Scholes-Merton put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (_math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * _math.sqrt(T))
    d2 = d1 - sigma * _math.sqrt(T)
    return K * _math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def ps_bsm_call(S, K, T, r, sigma):
    """Black-Scholes-Merton call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (_math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * _math.sqrt(T))
    d2 = d1 - sigma * _math.sqrt(T)
    return S * norm.cdf(d1) - K * _math.exp(-r * T) * norm.cdf(d2)

def ps_skew_iv(K, S, atm_iv, T):
    """Approximate a volatility skew: OTM puts carry higher IV than ATM.
    Keeps the model from treating cheap-looking far puts as free money."""
    denom = atm_iv * _math.sqrt(max(T, 1 / 365))
    if denom <= 0:
        return atm_iv
    m = _math.log(K / S) / denom  # standardized moneyness
    skew = (-m) * 0.045 if m < 0 else abs(m) * 0.015
    return atm_iv * (1 + skew)

def ps_pop_fattail(S, K, T, r, iv, is_put, df=PS_DF_BASE):
    """Probability the short option expires worthless under a Student-t
    (fat-tailed) terminal distribution. Lower than Black-Scholes by design;
    the gap is the 'steamroller discount'."""
    if T <= 0 or iv <= 0:
        return 1.0 if (is_put and K < S) or (not is_put and K > S) else 0.0
    t_scale = _math.sqrt((df - 2) / df) if df > 2 else 0.6
    drift = (r - iv * iv / 2) * T
    z = (_math.log(K / S) - drift) / (iv * _math.sqrt(T))
    t_stat = z / t_scale
    # put worthless => S_T > K => prob = 1 - F(t)
    return float(1 - student_t.cdf(t_stat, df)) if is_put else float(student_t.cdf(t_stat, df))

def ps_pop_bs(S, K, T, r, iv, is_put):
    """Black-Scholes probability of expiring worthless (normal tails)."""
    if T <= 0 or iv <= 0:
        return 1.0 if (is_put and K < S) or (not is_put and K > S) else 0.0
    d2 = (_math.log(S / K) + (r - iv * iv / 2) * T) / (iv * _math.sqrt(T))
    return float(norm.cdf(d2)) if is_put else float(norm.cdf(-d2))

def ps_event_df(event_flag):
    """Fatten tails for known volatility events inside the window."""
    return {"none": PS_DF_BASE, "macro": 3.0, "earn": 2.5, "both": 2.0}.get(event_flag, PS_DF_BASE)

def ps_realized_vol(symbol, window=20):
    """Annualized historical (realized) volatility from daily returns."""
    try:
        hist = yf.Ticker(symbol).history(period=f"{window + 10}d", interval="1d")
        if len(hist) < window:
            return None
        rets = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        return float(rets.tail(window).std() * _math.sqrt(252))
    except Exception:
        return None

def ps_iv_rank(symbol, current_iv_proxy):
    """Rough IV Rank using realized-vol range as a proxy when a true IV
    history is unavailable on free data. Returns 0-100 or None."""
    try:
        hist = yf.Ticker(symbol).history(period="1y", interval="1d")
        if len(hist) < 60:
            return None
        rets = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        roll = rets.rolling(20).std() * _math.sqrt(252)
        roll = roll.dropna()
        lo, hi = roll.min(), roll.max()
        if hi <= lo:
            return None
        return float(max(0, min(100, (current_iv_proxy - lo) / (hi - lo) * 100)))
    except Exception:
        return None

def ps_payoff_figure(rec, spot, width):
    """Plotly payoff-at-expiration chart for a single candidate."""
    struct = rec['struct']
    credit = rec['credit']
    mult = 100  # one contract
    lo, hi = spot * 0.90, spot * 1.10
    xs = np.linspace(lo, hi, 240)
    Ks = rec.get('Kshort')   # short put strike (or short strike)
    Kc = rec.get('Kcall')    # short call strike for two-sided structures

    pl = []
    for px in xs:
        if struct == "CSP":
            v = (credit - max(Ks - px, 0)) * mult
        elif struct == "Put Spread":
            Klong = Ks - width
            v = (credit - (max(Ks - px, 0) - max(Klong - px, 0))) * mult
            v = max(v, -rec['maxloss'] * mult)
        elif struct == "Iron Condor":
            KpL, KcL = Ks - width, Kc + width
            p_int = max(Ks - px, 0) - max(KpL - px, 0)
            c_int = max(px - Kc, 0) - max(px - KcL, 0)
            v = (credit - p_int - c_int) * mult
            v = max(v, -rec['maxloss'] * mult)
        elif struct == "Strangle":
            v = (credit - (max(Ks - px, 0) + max(px - Kc, 0))) * mult
        else:
            v = credit * mult
        pl.append(v)

    pl = np.array(pl)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=pl, mode='lines',
                             line=dict(color='#2dd4bf', width=2.5),
                             fill='tozeroy', fillcolor='rgba(45,212,191,0.10)'))
    fig.add_hline(y=0, line=dict(color='#6b7785', width=1, dash='dash'))
    fig.add_vline(x=spot, line=dict(color='#ffb03a', width=1, dash='dot'),
                  annotation_text=f"spot {spot:.0f}", annotation_position="top")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d4e0'), showlegend=False,
        xaxis=dict(title="Underlying at expiration", gridcolor='rgba(120,130,140,0.12)'),
        yaxis=dict(title="P/L ($)", gridcolor='rgba(120,130,140,0.12)'),
    )
    return fig


st.set_page_config(page_title="DJR Trading System", layout="wide")
st.title("DJR Trading System 🚀")
st.caption("✨ Full Trading System | Backtest | Live Trading | Signal History | Persistent Storage")

# Persistent Storage Paths
DATA_DIR = Path("trading_data")
DATA_DIR.mkdir(exist_ok=True)
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ACTIVE_TRADES_FILE = DATA_DIR / "active_trades.json"
SIGNAL_QUEUE_FILE = DATA_DIR / "signal_queue.json"
SIGNAL_HISTORY_FILE = DATA_DIR / "signal_history.json"
PERFORMANCE_FILE = DATA_DIR / "performance_metrics.json"

# Multi-Ticker Support
TICKERS = ["SPY", "SVXY", "QQQ", "EFA", "EEM", "AGG", "TLT"]

# Signal expiration time (minutes) - default
DEFAULT_SIGNAL_EXPIRATION = 30

# Load/Save Functions with Error Handling
def load_json(filepath, default):
    """Load JSON file with robust error handling"""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Validate it's not corrupt
                if data is None:
                    return default
                return data
        except json.JSONDecodeError:
            st.warning(f"⚠️ Corrupted file {filepath.name}, resetting to default")
            return default
        except Exception as e:
            st.warning(f"⚠️ Error loading {filepath.name}: {e}")
            return default
    return default

def save_json(filepath, data):
    """Save JSON file with atomic write (prevents corruption)"""
    try:
        # Write to temp file first
        temp_file = filepath.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        # Atomic rename
        temp_file.replace(filepath)
    except Exception as e:
        st.error(f"Error saving {filepath.name}: {e}")

# Initialize Session State with Persistent Loading
def init_session_state():
    """Initialize all session state variables with persistent data"""
    
    # Trade Log
    if 'trade_log' not in st.session_state:
        trade_log_data = load_json(TRADE_LOG_FILE, [])
        st.session_state.trade_log = pd.DataFrame(trade_log_data) if trade_log_data else pd.DataFrame(columns=[
            'Timestamp', 'Type', 'Symbol', 'Action', 'Size', 'Entry', 'Exit', 'P&L', 
            'Status', 'Signal ID', 'Entry Price Numeric', 'Exit Price Numeric', 
            'P&L Numeric', 'DTE', 'Strategy', 'Thesis', 'Max Hold Minutes', 'Actual Hold Minutes',
            'Conviction', 'Signal Type'
        ])
    
    # Active Trades
    if 'active_trades' not in st.session_state:
        trades_data = load_json(ACTIVE_TRADES_FILE, [])
        st.session_state.active_trades = []
        for trade in trades_data:
            # Convert string timestamp back to datetime
            if 'entry_time' in trade and isinstance(trade['entry_time'], str):
                try:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                except:
                    continue
            st.session_state.active_trades.append(trade)
    
    # Signal Queue with timestamp conversion
    if 'signal_queue' not in st.session_state:
        signal_queue_data = load_json(SIGNAL_QUEUE_FILE, [])
        st.session_state.signal_queue = []
        for sig in signal_queue_data:
            if 'timestamp' in sig and isinstance(sig['timestamp'], str):
                try:
                    sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
                except:
                    sig['timestamp'] = datetime.now(ZoneInfo("US/Eastern"))
            elif 'timestamp' not in sig:
                sig['timestamp'] = datetime.now(ZoneInfo("US/Eastern"))
            st.session_state.signal_queue.append(sig)
    
    # Signal History
    if 'signal_history' not in st.session_state:
        history_data = load_json(SIGNAL_HISTORY_FILE, [])
        st.session_state.signal_history = []
        for sig in history_data:
            if 'timestamp' in sig and isinstance(sig['timestamp'], str):
                try:
                    sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
                except:
                    sig['timestamp'] = datetime.now(ZoneInfo("US/Eastern"))
            if 'expiration' in sig and isinstance(sig['expiration'], str):
                try:
                    sig['expiration'] = datetime.fromisoformat(sig['expiration'])
                except:
                    pass
            st.session_state.signal_history.append(sig)
    
    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    # Performance Metrics
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
    
    # Macro Analyzer
    if 'macro_analyzer' not in st.session_state:
        st.session_state.macro_analyzer = MacroAnalyzer()
        try:
            st.session_state.macro_analyzer.fetch_macro_data()
        except:
            pass
    
    # Last save timestamp
    if 'last_save' not in st.session_state:
        st.session_state.last_save = datetime.now()

# Initialize everything
init_session_state()

# Settings (Sidebar)
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Trading Mode
    st.subheader("Trading Mode")
    TRADING_MODE = st.radio("Mode", ["Paper Trading", "Live Trading"], help="Paper = Simulated, Live = Real money")
    
    # Signal Generation Settings
    st.subheader("Signal Generation")
    MIN_CONVICTION = st.slider("Min Conviction", 1, 10, 5, help="Minimum conviction level to show signals")
    ENABLE_MACRO_FILTER = st.checkbox("Macro Signal Filter", value=True, help="Use macro conditions to filter signals")
    USE_DYNAMIC_STOPS = st.checkbox("Dynamic Position Sizing", value=True, help="Adjust position size based on conviction")
    SIGNAL_EXPIRATION_MINUTES = st.slider("Signal Expiration (min)", 10, 120, DEFAULT_SIGNAL_EXPIRATION, help="Minutes before signal expires")
    
    # V3.0: New signal types toggles
    st.subheader("Signal Types")
    ENABLE_CONTINUATION_SIGNALS = st.checkbox("Continuation Signals", value=True, help="Add to existing trends")
    ENABLE_SUPPORT_RESISTANCE = st.checkbox("Support/Resistance", value=True, help="Bounce signals at key levels")
    ENABLE_OPTIONS_SIGNALS = st.checkbox("Options Signals", value=True, help="Options trade opportunities")
    ENABLE_TREND_FOLLOWING = st.checkbox("Trend Following", value=True, help="Follow strong trends")
    ENABLE_MEAN_REVERSION = st.checkbox("Mean Reversion", value=True, help="Bollinger band bounces")
    
    # Stop Loss Settings
    st.subheader("Risk Management")
    STOP_LOSS_PCT = st.slider("Stop Loss %", -5.0, -0.5, -2.0, 0.5)
    TRAILING_STOP_PCT = st.slider("Trailing Stop %", 0.5, 3.0, 1.0, 0.5)
    
    # Data Persistence
    st.subheader("Data Management")
    if st.button("💾 Force Save All Data"):
        save_all_data()
        st.success("✅ All data saved!")
    
    if st.button("🗑️ Clear All History"):
        if st.checkbox("Confirm clear all"):
            st.session_state.signal_history = []
            st.session_state.trade_log = pd.DataFrame(columns=['Timestamp', 'Type', 'Symbol', 'Action', 'Size', 'Entry', 'Exit', 'P&L'])
            save_all_data()
            st.success("History cleared!")
    
    st.divider()
    st.caption(f"DJR Trading System | {TRADING_MODE}")
    st.caption(f"Last Save: {st.session_state.last_save.strftime('%H:%M:%S')}")

# Save all data function
def save_all_data():
    """Save all session data to disk"""
    try:
        # Save trade log
        save_json(TRADE_LOG_FILE, st.session_state.trade_log.to_dict('records'))
        
        # Save active trades with datetime conversion
        trades_to_save = []
        for trade in st.session_state.active_trades:
            trade_copy = trade.copy()
            if isinstance(trade_copy.get('entry_time'), datetime):
                trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
            trades_to_save.append(trade_copy)
        save_json(ACTIVE_TRADES_FILE, trades_to_save)
        
        # Save signal queue with datetime conversion
        signals_to_save = []
        for sig in st.session_state.signal_queue:
            sig_copy = sig.copy()
            if isinstance(sig_copy.get('timestamp'), datetime):
                sig_copy['timestamp'] = sig_copy['timestamp'].isoformat()
            signals_to_save.append(sig_copy)
        save_json(SIGNAL_QUEUE_FILE, signals_to_save)
        
        # Save signal history
        history_to_save = []
        for sig in st.session_state.signal_history:
            sig_copy = sig.copy()
            if isinstance(sig_copy.get('timestamp'), datetime):
                sig_copy['timestamp'] = sig_copy['timestamp'].isoformat()
            if isinstance(sig_copy.get('expiration'), datetime):
                sig_copy['expiration'] = sig_copy['expiration'].isoformat()
            history_to_save.append(sig_copy)
        save_json(SIGNAL_HISTORY_FILE, history_to_save)
        
        # Save performance metrics
        save_json(PERFORMANCE_FILE, st.session_state.performance_metrics)
        
        # Update last save time
        st.session_state.last_save = datetime.now()
        
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Auto-save every 5 minutes
if (datetime.now() - st.session_state.last_save).total_seconds() > 300:
    save_all_data()

# Market hours check
def is_market_open():
    """Check if market is open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

@st.cache_data(ttl=300)
def get_last_trading_day():
    """Return the last completed trading day as a date.

    Uses SPY's most recent data point from yfinance, which only contains
    actual trading days. Automatically skips weekends and market holidays
    with no hardcoded calendar. Returns a python date, or None on failure.
    """
    try:
        hist = yf.Ticker("SPY").history(period="5d", interval="1d")
        if hist.empty:
            return None
        return hist.index[-1].date()
    except Exception:
        return None



@st.cache_data(ttl=300)
def validate_ticker(symbol):
    """Return (is_valid, info_dict). Accepts any stock/ETF symbol.
    Cheap existence check: a 5-day history with at least one row.
    info_dict carries price, name, and day change when available."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return False, {}
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            return False, {}
        price = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else price
        chg = price - prev
        chg_pct = (chg / prev * 100) if prev else 0.0
        # name lookup is best-effort; never fail on it
        name = symbol
        try:
            fi = getattr(t, "fast_info", None)
            info = t.info if not fi else {}
            name = (info.get("shortName") or info.get("longName") or symbol) if info else symbol
        except Exception:
            name = symbol
        return True, {"symbol": symbol, "price": price, "change": chg,
                      "change_pct": chg_pct, "name": name}
    except Exception:
        return False, {}

def ai_portfolio_tickers():
    """Tickers from the live AI portfolio, for one-click quick-pick."""
    try:
        return [h['ticker'] for h in ai_load_portfolio() if h.get('ticker')]
    except Exception:
        return []

def resolve_search_symbol(text_input, quick_pick, fallback):
    """Decide which symbol to use. Free-text wins if non-empty, else the
    quick-pick selection, else the fallback (first default ticker)."""
    text_input = (text_input or "").strip().upper()
    if text_input:
        return text_input
    if quick_pick and quick_pick != "—":
        return quick_pick
    return fallback

AI_ENTRY_TARGETS = {
    "GOOGL": {"optimal": 353.72, "secondary": 368.93, "ceiling": 418.37, "pe_anchor": "~24x", "reval": False},
    "AMZN": {"optimal": 254.4, "secondary": 263.87, "ceiling": 286.88, "pe_anchor": "~32x", "reval": False},
    "META": {"optimal": 594.56, "secondary": 616.7, "ceiling": 670.46, "pe_anchor": "~24x", "reval": False},
    "MSFT": {"optimal": 423.23, "secondary": 438.98, "ceiling": 477.25, "pe_anchor": "~31x", "reval": False},
    "ORCL": {"optimal": 207.72, "secondary": 216.75, "ceiling": 234.81, "pe_anchor": "~26x", "reval": False},
    "NVDA": {"optimal": 196.36, "secondary": 204.81, "ceiling": 232.25, "pe_anchor": "~34x", "reval": False},
    "AVGO": {"optimal": 415.5, "secondary": 433.37, "ceiling": 491.45, "pe_anchor": "~38x", "reval": False},
    "TSM": {"optimal": 389.16, "secondary": 405.9, "ceiling": 460.3, "pe_anchor": "~26x", "reval": False},
    "MU": {"optimal": 893.32, "secondary": 932.16, "ceiling": 1009.84, "pe_anchor": "~18x", "reval": False},
    "MRVL": {"optimal": 232.0, "secondary": 255.0, "ceiling": 300.0, "pe_anchor": "~40x FY28", "reval": True},
    "AMD": {"optimal": 474.81, "secondary": 495.46, "ceiling": 536.74, "pe_anchor": "~32x", "reval": False},
    "ANET": {"optimal": 149.9, "secondary": 155.48, "ceiling": 169.04, "pe_anchor": "~38x", "reval": False},
    "DELL": {"optimal": 378.82, "secondary": 399.86, "ceiling": 433.54, "pe_anchor": "~14x", "reval": False},
    "ARM": {"optimal": 317.96, "secondary": 335.63, "ceiling": 363.89, "pe_anchor": "~95x", "reval": True},
    "VRT": {"optimal": 293.61, "secondary": 306.24, "ceiling": 347.28, "pe_anchor": "~50x", "reval": False},
    "ETN": {"optimal": 376.56, "secondary": 390.59, "ceiling": 424.64, "pe_anchor": "~30x", "reval": False},
    "ASML": {"optimal": 1499.87, "secondary": 1564.38, "ceiling": 1774.04, "pe_anchor": "~32x", "reval": False},
    "APH": {"optimal": 139.83, "secondary": 145.04, "ceiling": 157.69, "pe_anchor": "~38x", "reval": False},
    "AMAT": {"optimal": 423.06, "secondary": 438.81, "ceiling": 477.06, "pe_anchor": "~26x", "reval": False},
    "PANW": {"optimal": 264.84, "secondary": 274.71, "ceiling": 298.66, "pe_anchor": "~48x", "reval": True},
    "TT": {"optimal": 424.22, "secondary": 440.02, "ceiling": 478.38, "pe_anchor": "~30x", "reval": False},
    "CSCO": {"optimal": 111.99, "secondary": 116.81, "ceiling": 124.03, "pe_anchor": "~17x", "reval": False},
    "VST": {"optimal": 149.01, "secondary": 155.42, "ceiling": 176.25, "pe_anchor": "~22x", "reval": False},
    "CEG": {"optimal": 267.61, "secondary": 279.12, "ceiling": 316.53, "pe_anchor": "~28x", "reval": False},
    "GEV": {"optimal": 900.54, "secondary": 939.27, "ceiling": 1065.15, "pe_anchor": "~45x", "reval": False},
    "PWR": {"optimal": 669.03, "secondary": 693.94, "ceiling": 754.43, "pe_anchor": "~38x", "reval": False},
    "CCJ": {"optimal": 105.94, "secondary": 109.88, "ceiling": 119.46, "pe_anchor": "n/a", "reval": False},
    "NEE": {"optimal": 81.79, "secondary": 84.83, "ceiling": 92.23, "pe_anchor": "~20x", "reval": False},
    "CAT": {"optimal": 814.56, "secondary": 849.59, "ceiling": 902.15, "pe_anchor": "~20x", "reval": False},
    "FSLR": {"optimal": 285.31, "secondary": 297.59, "ceiling": 315.99, "pe_anchor": "~12x", "reval": False},
}

AI_BENCH_TARGETS = {
    "LITE": {"optimal": 895.0, "secondary": 931.0, "ceiling": 1067.0},
    "COHR": {"optimal": 108.56, "secondary": 113.28, "ceiling": 127.44},
    "FN": {"optimal": 395.6, "secondary": 413.0, "ceiling": 464.4},
    "ARM": {"optimal": 317.96, "secondary": 335.63, "ceiling": 363.89},
    "QCOM": {"optimal": 235.96, "secondary": 245.0, "ceiling": 271.1},
    "INTC": {"optimal": 104.36, "secondary": 110.1, "ceiling": 120.41},
    "CRWV": {"optimal": 98.58, "secondary": 103.0, "ceiling": 120.48},
    "ON": {"optimal": 111.57, "secondary": 116.0, "ceiling": 130.27},
    "SNPS": {"optimal": 442.33, "secondary": 461.35, "ceiling": 513.67},
    "LRCX": {"optimal": 295.91, "secondary": 308.63, "ceiling": 343.63},
    "GLW": {"optimal": 168.48, "secondary": 175.73, "ceiling": 195.65},
    "SKHHY": {"optimal": 0, "secondary": 0, "ceiling": 0},
    "SSNLF": {"optimal": 0, "secondary": 0, "ceiling": 0},
    "SNDK": {"optimal": 0, "secondary": 0, "ceiling": 0},
    "WDC": {"optimal": 0, "secondary": 0, "ceiling": 0},
    "OKLO": {"optimal": 0, "secondary": 0, "ceiling": 0},
    "SMR": {"optimal": 11.51, "secondary": 12.04, "ceiling": 13.69},
}

def ai_entry_zone(ticker, price_now):
    """Classify where the live price sits vs the entry targets.
    Returns (zone_label, target_dict_or_None).

    Re-rate handling: names flagged reval=True (a recent earnings/guidance event
    obsoleted the static levels) return a 'RE-RATE — refresh' status rather than a
    stale 🔴. Also, if the live price has run more than 12% past the ceiling, the
    targets are treated as stale and flagged for re-derivation regardless of reval."""
    t = AI_ENTRY_TARGETS.get(ticker)
    if not t or price_now is None:
        return ("—", t)
    if t.get("reval"):
        return ("🔄 RE-RATE — refresh", t)
    # stale guard: price gapped well past the ceiling => targets likely obsolete
    if price_now > t["ceiling"] * 1.12:
        return ("🔄 Stale — re-derive", t)
    if price_now <= t["optimal"]:
        return ("🟢 At/below optimal", t)
    if price_now <= t["secondary"]:
        return ("🟢 Buy zone", t)
    if price_now <= t["ceiling"]:
        return ("🟡 Below ceiling", t)
    return ("🔴 Above ceiling", t)


def ai_bench_zone(ticker, price_now):
    """Entry zone for a BENCH name (tracking only). Returns (label, target|None).
    Bench names with no price data (e.g. Korea-listed OTC) return ('—', None)."""
    t = AI_BENCH_TARGETS.get(ticker)
    if not t or price_now is None or t.get("ceiling", 0) <= 0:
        return ("—", t if t and t.get("ceiling", 0) > 0 else None)
    if price_now <= t["optimal"]:
        return ("🟢 At/below optimal", t)
    if price_now <= t["secondary"]:
        return ("🟢 Buy zone", t)
    if price_now <= t["ceiling"]:
        return ("🟡 Below ceiling", t)
    return ("🔴 Above ceiling", t)

def expire_old_signals():
    """Move expired signals from queue to history"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    expired = []
    
    for sig in st.session_state.signal_queue[:]:
        if 'timestamp' not in sig:
            continue
            
        signal_age_minutes = (now - sig['timestamp']).total_seconds() / 60
        
        if signal_age_minutes > SIGNAL_EXPIRATION_MINUTES:
            # Add to history
            sig['expiration'] = now
            sig['status'] = 'Expired'
            st.session_state.signal_history.append(sig)
            expired.append(sig)
            # Remove from queue
            st.session_state.signal_queue.remove(sig)
    
    if expired:
        save_all_data()
    
    return len(expired)

# V3.0: ENHANCED SIGNAL GENERATION (same as before but with better error handling)
def generate_signal():
    """Enhanced signal generation with more liberal conditions"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    now_str = now.strftime("%m/%d %H:%M")
    
    # Expire old signals first
    expire_old_signals()
    
    # Get macro regime
    try:
        regime = st.session_state.macro_analyzer.detect_regime()
        
        try:
            vix = yf.Ticker("^VIX")
            vix_current = vix.history(period="1d")['Close'].iloc[-1]
            vix_elevated = vix_current > 20
            
            if vix_elevated and 'SVXY' in regime.get('avoid_tickers', []):
                regime['avoid_tickers'] = [t for t in regime['avoid_tickers'] if t != 'SVXY']
                regime['conviction_boost'] += 1
        except:
            vix_elevated = False
            vix_current = 15
            
    except:
        regime = {
            'type': 'NORMAL',
            'environment': 'Normal Market',
            'equity_bias': 'BALANCED',
            'preferred_tickers': TICKERS,
            'avoid_tickers': [],
            'conviction_boost': 0
        }
        vix_elevated = False
        vix_current = 15
    
    # Analyze each ticker
    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)
            hist_data = t.history(period="100d", interval="1d")
            
            if hist_data.empty or len(hist_data) < 50:
                continue
            
            df = calculate_technical_indicators(hist_data)
            
            if df.empty or len(df) < 20:
                continue
            
            # Get current values
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Get indicators
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else 0
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 20
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            stoch_k = df['Stoch_%K'].iloc[-1] if 'Stoch_%K' in df.columns and not pd.isna(df['Stoch_%K'].iloc[-1]) else 50
            
            # Get SMAs
            sma_values = {}
            for period in [10, 20, 50, 100, 200]:
                col_name = f'SMA_{period}'
                if col_name in df.columns and len(df) >= period:
                    sma_values[period] = df[col_name].iloc[-1]
            
            signal = None
            
            # SVXY Volatility Signals (same as enhanced version)
            if ticker == "SVXY" and len(df) >= 5:
                five_day_drop = ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
                
                if (five_day_drop < -8 and volume_ratio > 1.1 and price_change_pct > -1.0):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'SVXY Vol Spike Recovery',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Mean Reversion - Vol Spike',
                        'thesis': f"VOL SPIKE RECOVERY: SVXY dropped {five_day_drop:.1f}% in 5 days. Mean reversion at ${current_price:.2f}. Vol {volume_ratio:.1f}x. VIX: {vix_current:.1f}",
                        'conviction': 8,
                        'signal_type': 'SVXY Vol Spike Recovery'
                    }
                
                elif (len(df) >= 2 and df['Close'].pct_change().iloc[-2] < -0.03 and 
                      price_change_pct > 0.5 and volume_ratio > 1.2):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'SVXY Sharp Drop Bounce',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Mean Reversion - Bounce',
                        'thesis': f"SHARP DROP RECOVERY: SVXY bouncing +{price_change_pct:.1f}% after yesterday's {df['Close'].pct_change().iloc[-2]*100:.1f}% drop.",
                        'conviction': 7,
                        'signal_type': 'SVXY Sharp Drop Bounce'
                    }
            
            # SMA Uptrend Signals (already crossed, not just today)
            if not signal and (10 in sma_values and 20 in sma_values):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                
                if (sma_10 > sma_20 and current_price > sma_10 and volume_ratio > 1.1):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'SMA 10/20 Uptrend',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - SMA Uptrend',
                        'thesis': f"UPTREND: {ticker} above SMA10 (${sma_10:.2f}) and SMA20 (${sma_20:.2f}). Price ${current_price:.2f}, Vol {volume_ratio:.1f}x, RSI {rsi:.0f}.",
                        'conviction': 7,
                        'signal_type': 'SMA Uptrend'
                    }
            
            # Golden Cross (already crossed)
            if not signal and (50 in sma_values and 200 in sma_values):
                sma_50 = sma_values[50]
                sma_200 = sma_values[200]
                
                if (sma_50 > sma_200 and current_price > sma_50 and macd > 0 and adx > 15):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Golden Cross Trend',
                        'symbol': ticker,
                        'action': f"BUY 20 shares @ ${current_price:.2f}",
                        'size': 20,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Golden Cross',
                        'thesis': f"GOLDEN CROSS: {ticker} in golden cross (50>200). Price ${current_price:.2f} above SMA50, MACD bullish, ADX {adx:.1f}.",
                        'conviction': 9,
                        'signal_type': 'Golden Cross'
                    }
            
            # Volume Breakout
            if not signal and (20 in sma_values):
                sma_20 = sma_values[20]
                
                if (volume_ratio > 1.5 and price_change_pct > 0.5 and 
                    current_price > sma_20 and rsi > 50 and rsi < 75):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Volume Breakout',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Volume Breakout',
                        'thesis': f"VOLUME BREAKOUT: {ticker} ${current_price:.2f} with strong volume ({volume_ratio:.1f}x), +{price_change_pct:.1f}% move. RSI {rsi:.0f}.",
                        'conviction': 7,
                        'signal_type': 'Volume Breakout'
                    }
            
            # Continuation Signals
            if not signal and ENABLE_CONTINUATION_SIGNALS and (10 in sma_values and 20 in sma_values and 50 in sma_values):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
                
                price_near_10_sma = abs(current_price - sma_10) / sma_10 < 0.02
                
                if (sma_10 > sma_20 > sma_50 and price_near_10_sma and rsi > 40 and rsi < 60):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Trend Continuation',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Add to Winner',
                        'thesis': f"CONTINUATION: {ticker} in uptrend, pullback to 10-SMA (${sma_10:.2f}). Add at ${current_price:.2f}, RSI {rsi:.0f}.",
                        'conviction': 8,
                        'signal_type': 'Continuation'
                    }
            
            # Support/Resistance
            if not signal and ENABLE_SUPPORT_RESISTANCE and 'S1' in df.columns:
                s1 = df['S1'].iloc[-1]
                price_near_support = abs(current_price - s1) / s1 < 0.005
                
                if (price_near_support and price_change_pct > 0.3 and rsi > 30 and rsi < 50):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Support Bounce',
                        'symbol': ticker,
                        'action': f"BUY 12 shares @ ${current_price:.2f}",
                        'size': 12,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Support Bounce',
                        'thesis': f"SUPPORT: {ticker} bouncing at S1 (${s1:.2f}). Price ${current_price:.2f}, RSI {rsi:.0f}, +{price_change_pct:.1f}% reversal.",
                        'conviction': 6,
                        'signal_type': 'Support Bounce'
                    }
            
            # Trend Following
            if not signal and ENABLE_TREND_FOLLOWING and all(p in sma_values for p in [10, 20, 50, 200]):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
                sma_200 = sma_values[200]
                
                all_aligned = (current_price > sma_10 > sma_20 > sma_50 > sma_200)
                
                if (all_aligned and adx > 20 and rsi > 50 and rsi < 70):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Strong Trend',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Trend Following',
                        'thesis': f"STRONG TREND: {ticker} all SMAs aligned. ADX {adx:.1f}, RSI {rsi:.0f}. Momentum continues.",
                        'conviction': 9,
                        'signal_type': 'Strong Trend'
                    }
            
            # Mean Reversion
            if not signal and ENABLE_MEAN_REVERSION and 'BB_Lower' in df.columns and 'RSI' in df.columns:
                bb_lower = df['BB_Lower'].iloc[-1]
                if (current_price <= bb_lower * 1.01 and rsi < 40 and price_change_pct > 0):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Mean Reversion',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Mean Reversion',
                        'thesis': f"MEAN REVERSION: Price ${current_price:.2f} at lower BB (${bb_lower:.2f}), RSI {rsi:.0f}. Statistical bounce.",
                        'conviction': 7,
                        'signal_type': 'Mean Reversion'
                    }
            
            # Oversold Bounce
            if not signal and (rsi < 30 and price_change_pct > 0.2 and 20 in sma_values):
                sma_20 = sma_values[20]
                signal = {
                    'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                    'timestamp': now,
                    'time': now_str,
                    'type': 'Oversold Bounce',
                    'symbol': ticker,
                    'action': f"BUY 12 shares @ ${current_price:.2f}",
                    'size': 12,
                    'entry_price': current_price,
                    'max_hold': None,
                    'dte': 0,
                    'strategy': f'{ticker} Long - Oversold Bounce',
                    'thesis': f"OVERSOLD: {ticker} ${current_price:.2f} reversing (RSI {rsi:.0f}). +{price_change_pct:.1f}% bounce.",
                    'conviction': 6,
                    'signal_type': 'Oversold Bounce'
                }
            
            # Options Signals
            if not signal and ENABLE_OPTIONS_SIGNALS:
                try:
                    options_df = get_options_chain(ticker, dte_min=14, dte_max=45)
                    
                    if not options_df.empty and 20 in sma_values and 'impliedVolatility' in options_df.columns:
                        sma_20 = sma_values[20]
                        avg_iv = options_df[options_df['type'] == 'Put']['impliedVolatility'].mean()
                        
                        if avg_iv > 0.25 and current_price > sma_20 and rsi > 45:
                            puts = options_df[options_df['type'] == 'Put'].copy()
                            puts['strike_diff'] = abs(puts['strike'] - current_price)
                            atm_put = puts.nsmallest(1, 'strike_diff').iloc[0]
                            
                            signal = {
                                'id': f"SIG-{ticker}-OPT-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                                'timestamp': now,
                                'time': now_str,
                                'type': 'Options: Put Credit Spread',
                                'symbol': ticker,
                                'action': f"SELL {ticker} Put Spread ${atm_put['strike']:.0f}/${atm_put['strike']-5:.0f}",
                                'size': 1,
                                'entry_price': atm_put['mid'],
                                'max_hold': None,
                                'dte': int(atm_put['dte']),
                                'strategy': f'{ticker} Options - Put Credit Spread',
                                'thesis': f"HIGH IV: {ticker} IV={avg_iv:.1%}, price ${current_price:.2f} above support. Sell ${atm_put['strike']:.0f}P DTE {int(atm_put['dte'])}.",
                                'conviction': 7,
                                'signal_type': 'Options Put Spread'
                            }
                except:
                    pass
            
            # Apply filters and add signal
            if signal:
                if ENABLE_MACRO_FILTER:
                    signal = adjust_signal_for_macro(signal, regime, df)
                
                if signal and should_take_signal(signal, MIN_CONVICTION):
                    if USE_DYNAMIC_STOPS:
                        signal['size'] = calculate_position_size(signal)
                        if 'shares' in signal['action']:
                            signal['action'] = f"BUY {signal['size']} shares @ ${signal['entry_price']:.2f}"
                    
                    st.session_state.signal_queue.append(signal)
                    
                    # Add to history
                    sig_history = signal.copy()
                    sig_history['status'] = 'Active'
                    st.session_state.signal_history.append(sig_history)
                    
                    save_all_data()
                    break
                    
        except Exception as e:
            continue

# Auto-Exit Logic
def simulate_exit():
    """Exit trades based on stop loss, trailing stop, momentum reversal"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    
    for trade in st.session_state.active_trades[:]:
        current_price = market_data[trade['symbol']]['price']
        entry_price = trade['entry_price']
        
        if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        exit_triggered = False
        exit_reason = ""
        
        # Stop Loss
        if pnl_pct <= STOP_LOSS_PCT:
            exit_triggered = True
            exit_reason = f"Stop Loss ({STOP_LOSS_PCT:.1f}%)"
        
        # Trailing Stop
        elif pnl_pct >= 4.0:
            max_pnl_reached = trade.get('max_pnl_reached', pnl_pct)
            if pnl_pct > max_pnl_reached:
                trade['max_pnl_reached'] = pnl_pct
            elif (max_pnl_reached - pnl_pct) >= TRAILING_STOP_PCT:
                exit_triggered = True
                exit_reason = f"Trailing Stop (from +{max_pnl_reached:.1f}% to +{pnl_pct:.1f}%)"
        
        # Execute exit
        if exit_triggered:
            exit_price = current_price
            minutes_held = (now - trade['entry_time']).total_seconds() / 60
            
            if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
                pnl = (entry_price - exit_price) * trade['size']
                close_action = 'Buy to Cover'
            else:
                pnl = (exit_price - entry_price) * trade['size']
                close_action = 'Sell'
            
            log_trade(
                ts=now.strftime("%m/%d %H:%M"),
                typ="Close",
                sym=trade['symbol'],
                action=close_action,
                size=trade['size'],
                entry=f"${entry_price:.2f}",
                exit=f"${exit_price:.2f}",
                pnl=f"${pnl:.0f}",
                status=f"Closed ({exit_reason})",
                sig_id=trade['signal_id'],
                entry_numeric=entry_price,
                exit_numeric=exit_price,
                pnl_numeric=pnl,
                dte=trade.get('dte'),
                strategy=trade.get('strategy'),
                thesis=trade.get('thesis'),
                max_hold=trade.get('max_hold'),
                actual_hold=minutes_held,
                conviction=trade.get('conviction'),
                signal_type=trade.get('signal_type')
            )
            st.session_state.active_trades.remove(trade)
    
    save_all_data()

# Get options chain
@st.cache_data(ttl=300)
def get_options_chain(symbol="SPY", dte_min=7, dte_max=60):
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            return pd.DataFrame()
        
        all_options = []
        for exp_date in expirations:
            exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
            dte = (exp_datetime - datetime.now()).days
            
            if dte_min <= dte <= dte_max:
                try:
                    chain = ticker.option_chain(exp_date)
                    
                    for opt_type, opts in [('Call', chain.calls), ('Put', chain.puts)]:
                        if not opts.empty:
                            opts = opts.copy()
                            opts['type'] = opt_type
                            opts['dte'] = dte
                            opts['expiration'] = exp_date
                            # keep bid/ask explicitly; mid is informational only.
                            # For selling, the executable price is the BID, not the mid.
                            opts['bid'] = pd.to_numeric(opts['bid'], errors='coerce').fillna(0.0)
                            opts['ask'] = pd.to_numeric(opts['ask'], errors='coerce').fillna(0.0)
                            opts['mid'] = (opts['bid'] + opts['ask']) / 2
                            all_options.append(opts)
                except:
                    continue
        
        if all_options:
            df = pd.concat(all_options, ignore_index=True)
            df['symbol'] = df['contractSymbol']
            return df
        
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ============================================================
# AI STRATEGY DATA (embedded, from ai-strategy-toolkit-v2.html)
# ============================================================

AI_STRATEGY_INCEPTION = "2026-02-10"
AI_STRATEGY_BENCHMARK = "^NDX"

# Tier allocation guardrails (target % of portfolio)
AI_TIER_TARGETS = {
    "HYPER": 30, "TIER 1": 32, "TIER 2": 19, "TIER 3": 17, "CASH": 2,
}

AI_PORTFOLIO = [
    {"ticker": "GOOGL", "name": "Alphabet", "tier": "HYPER", "score": 8.5, "target_weight": 7.0, "conviction_date": "Q1 25", "thesis": "Fastest cloud growth (48% YoY). $175-185B 2026 capex. Strongest balance sheet (D/E 0.14, ROIC 31.6%). Q4 EPS $2.82 vs $2.64 est. TPU custom silicon reduces NVDA dependency. PEG 1.84 reasonable. Risk: DOJ antitrust, search disruption narrative."},
    {"ticker": "META", "name": "Meta Platforms", "tier": "HYPER", "score": 8.2, "target_weight": 6.0, "conviction_date": "Q1 25", "thesis": "Best PEG among hyperscalers (1.09). $201B revenue, 30% ROE, 28% ROIC. AI ad targeting driving $5B+ incremental revenue. Llama ecosystem. $57B+ 2026 capex self-funded. Risk: Reality Labs losses ($16B/yr), regulatory."},
    {"ticker": "AMZN", "name": "Amazon", "tier": "HYPER", "score": 6.8, "target_weight": 5.0, "conviction_date": "Q1 25", "thesis": "AWS $142B run rate, 19% growth. $200B 2026 AI capex. Custom Trainium/Inferentia gaining. $691B revenue. Post-earnings selloff on capex guidance. PEG 1.65 is fair. Benchmark weight; add on pullback below $190."},
    {"ticker": "MSFT", "name": "Microsoft", "tier": "HYPER", "score": 6.2, "target_weight": 5.0, "conviction_date": "Q1 25", "thesis": "$625B remaining obligations. Azure 31% growth (13pts AI). $305B rev, 34% ROE. Richest PEG (1.65) among hyperscalers. Cloud deceleration vs peers. Underweight vs NDX 8.1% until Copilot monetizes."},
    {"ticker": "ORCL", "name": "Oracle", "tier": "HYPER", "score": 7.0, "target_weight": 4.0, "conviction_date": "Q2 25", "thesis": "OCI revenue surging 50%+. $130B+ remaining obligations. NVDA partnership. EPS $5.45, growing 89% YoY. High debt (D/E ~6x) is primary concern. Execution risk on rapid global DC buildout."},
    {"ticker": "BABA", "name": "Alibaba (ADR)", "tier": "HYPER", "score": 5.5, "target_weight": 3.0, "conviction_date": "Q2 25", "thesis": "Largest non-US hyperscaler. China's #1 cloud. $10B+ AI capex ramp. 12x fwd P/E, massive discount to US peers. Risk: China regulatory/geopolitical, VIE structure, US-China decoupling."},
    {"ticker": "NVDA", "name": "NVIDIA", "tier": "TIER 1", "score": 8.7, "target_weight": 8.0, "conviction_date": "Q1 25", "thesis": "85%+ GPU share for AI training. PEG 0.73. Blackwell demand exceeds supply into H2 2026. 73% gross margins, $45B net cash. CUDA lock-in. ER 2/25 is major catalyst. Risk: custom ASIC, China export controls, DeepSeek efficiency."},
    {"ticker": "AVGO", "name": "Broadcom", "tier": "TIER 1", "score": 8.0, "target_weight": 5.0, "conviction_date": "Q1 25", "thesis": "Leading custom ASIC designer (Google TPU, Meta MTIA). AI revenue tripled YoY. VMware adds recurring SW revenue. 2025 return +50.6%. Strong networking portfolio. Risk: customer concentration, premium valuation."},
    {"ticker": "TSM", "name": "TSMC (ADR)", "tier": "TIER 1", "score": 8.5, "target_weight": 4.0, "conviction_date": "Q1 25", "thesis": "Fabs every advanced AI chip (NVDA, AMD, AVGO). 60%+ gross margins on advanced nodes. Only true monopoly in AI supply chain. Arizona fab de-risks geopolitics. Risk: Taiwan/China (primary), cyclicality."},
    {"ticker": "AMD", "name": "AMD", "tier": "TIER 1", "score": 6.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "#2 GPU with MI300X gaining enterprise. Best PEG in semis (0.62). DC revenue +70% YoY. Xilinx for edge AI. Underperformed NVDA in 2025. Add on MI350 evidence. Risk: NVDA dominance."},
    {"ticker": "MU", "name": "Micron", "tier": "TIER 1", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "HBM3E critical bottleneck for AI training. 2026 HBM sold out. Memory pricing favorable. Cyclical history but AI creates structural shift. Risk: oversupply cycles, Samsung/SK competition."},
    {"ticker": "ANET", "name": "Arista Networks", "tier": "TIER 1", "score": 7.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "DC networking leader, 157% ROIC. 800G/1.6T for AI clusters. META/MSFT top customers. ER 2/12 will update outlook. Risk: customer concentration, Cisco competitive threat."},
    {"ticker": "DELL", "name": "Dell Technologies", "tier": "TIER 1", "score": 6.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Cheapest AI infra play at ~10x fwd P/E. $6B+ AI server pipeline. Enterprise refresh cycle. ER 2/26 critical. Risk: low-margin hardware, SMCI competition."},
    {"ticker": "MRVL", "name": "Marvell Tech", "tier": "TIER 1", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Custom ASIC + networking silicon. Revenue +45% from DC segment. Electro-optics for AI clusters. Higher growth potential than AVGO but more volatile. Risk: execution, customer concentration."},
    {"ticker": "VRT", "name": "Vertiv", "tier": "TIER 2", "score": 7.5, "target_weight": 4.0, "conviction_date": "Q1 25", "thesis": "Pure-play AI DC cooling/power. $9.5B+ backlog +30% QoQ. AI racks 3-10x more heat. 2025 return +42.8%. Expanding liquid cooling. Risk: conversion timing, Schneider/ABB competition."},
    {"ticker": "ETN", "name": "Eaton Corp", "tier": "TIER 2", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "DC electrical infra (UPS, switchgear, PDUs). $13B+ backlog. 20%+ ROIC. Div aristocrat 2%+ yield. More diversified than VRT. Risk: industrial cycle, premium valuation."},
    {"ticker": "ASML", "name": "ASML (ADR)", "tier": "TIER 2", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "EUV lithography monopoly. $35B+ backlog. Every advanced AI chip runs through ASML machines. Risk: cyclical equipment spend, China restriction reduces TAM, premium valuation."},
    {"ticker": "APH", "name": "Amphenol", "tier": "TIER 2", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q4 25", "thesis": "High-speed connectors/cables for AI racks. Every GPU cluster needs Amphenol interconnects. 25%+ ROE, 20%+ ROIC. M&A machine. Risk: diversified industrial dampens AI sensitivity."},
    {"ticker": "AMAT", "name": "Applied Materials", "tier": "TIER 2", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q2 25", "thesis": "Semi equipment leader, tools for TSMC/Samsung/Intel. AI chip demand drives fab buildout. Picks-and-shovels play. Risk: cyclical spend, China export restrictions."},
    {"ticker": "TT", "name": "Trane Technologies", "tier": "TIER 2", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q4 25", "thesis": "DC HVAC/cooling. 35% ROE, consistent compounder. AI cooling demand structural. More diversified than VRT. Risk: indirect AI exposure, premium valuation."},
    {"ticker": "CSCO", "name": "Cisco", "tier": "TIER 2", "score": 5.5, "target_weight": 2.0, "conviction_date": "Q1 25", "thesis": "Defensive. 2.8% yield. Splunk adds AI observability. Enterprise networking refresh. Lower beta ballast. Risk: slow innovation vs Arista, legacy decline."},
    {"ticker": "HPE", "name": "HPE", "tier": "TIER 2", "score": 5.0, "target_weight": 2.0, "conviction_date": "Q1 25", "thesis": "Deep value at ~9x fwd P/E. AI server segment growing. Juniper acquisition adds networking. Risk: low margins, execution risk, losing AI server share to DELL."},
    {"ticker": "VST", "name": "Vistra Corp", "tier": "TIER 3", "score": 7.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Largest competitive US power gen (nuclear+gas+solar). ~35% off ATH, entry opportunity. AI DCs consume 1-2 GW each. Nuclear renaissance thesis. Risk: regulation, nat gas exposure."},
    {"ticker": "CEG", "name": "Constellation Energy", "tier": "TIER 3", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Largest US nuclear fleet. TMI restart deal with MSFT (20-yr PPA). Clean energy premium. Nuclear capacity irreplaceable. Risk: TMI execution, regulatory timelines."},
    {"ticker": "GEV", "name": "GE Vernova", "tier": "TIER 3", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Gas turbine orders surging for DC baseload. Grid equipment (transformers) multi-year backlogs. Only near-term bridge for AI power. Risk: turbine execution, offshore wind losses."},
    {"ticker": "PWR", "name": "Quanta Services", "tier": "TIER 3", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Largest US electrical contractor. Builds transmission/substations. $30B+ backlog. Grid investment deferred 20 years. Risk: labor shortages, execution."},
    {"ticker": "CCJ", "name": "Cameco", "tier": "TIER 3", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Premier uranium producer + Westinghouse JV. 10+ yrs underinvestment in fuel supply. Long-term contracts. Risk: uranium volatility, Kazakhstan competition."},
    {"ticker": "NEE", "name": "NextEra Energy", "tier": "TIER 3", "score": 5.5, "target_weight": 2.0, "conviction_date": "Q1 25", "thesis": "Largest US utility, largest wind/solar. Low beta (0.5), portfolio ballast. 3%+ yield. Renewables PPAs with hyperscalers. Risk: rate-sensitive, slower growth vs IPPs."},
    {"ticker": "CAT", "name": "Caterpillar", "tier": "TIER 3", "score": 5.0, "target_weight": 2.0, "conviction_date": "Q4 25", "thesis": "Backup power generators + DC construction equipment. Every DC needs 50-200MW backup from CAT. Blue-chip dividend. Risk: indirect AI exposure, global construction cyclicality."},
    {"ticker": "FSLR", "name": "First Solar", "tier": "TIER 3", "score": 5.0, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Largest US solar manufacturer. DCs co-locating solar. IRA subsidies tailwind. Domestic mfg protects vs tariffs. Risk: IRA policy uncertainty, panel oversupply."},
]

AI_BENCH = [
    {"ticker": "TCEHY", "name": "Tencent (ADR)", "tier": "HYPER", "score": 4.5, "note": "China #2 cloud. Gaming + social fund capex. Not owned: Lower AI visibility than BABA, ADR/VIE risk. Trigger: improved US-China relations or clear AI revenue metrics."},
    {"ticker": "BIDU", "name": "Baidu (ADR)", "tier": "HYPER", "score": 4.0, "note": "Ernie LLM leader in China. Apollo Go autonomous. Not owned: Core search declining, limited AI monetization. Trigger: cloud revenue acceleration."},
    {"ticker": "ARM", "name": "ARM Holdings", "tier": "TIER 1", "score": 4.5, "note": "CPU architecture for inference chips. Royalty model = high margin. Not owned: Extreme valuation (100x+ fwd P/E). Trigger: 30%+ pullback or revenue growth above 40%."},
    {"ticker": "SMCI", "name": "Super Micro", "tier": "TIER 1", "score": 4.0, "note": "Fastest AI server assembler. Revenue +100%. Not owned: Governance (delayed 10-K, auditor switch, DOJ). Trigger: full governance resolution."},
    {"ticker": "QCOM", "name": "Qualcomm", "tier": "TIER 1", "score": 4.5, "note": "Edge AI leader (Snapdragon). On-device inference. Not owned: Mobile cyclicality, ARM dispute. Trigger: edge AI monetization breakout."},
    {"ticker": "INTC", "name": "Intel", "tier": "TIER 1", "score": 3.0, "note": "Foundry 18A could serve AI fab demand. Gaudi accelerators. Not owned: Execution failures, behind TSMC by 2+ nodes, burning cash. Monitor only."},
    {"ticker": "CRWV", "name": "CoreWeave", "tier": "TIER 1", "score": 4.5, "note": "AI-native cloud, GPU-as-a-service. Not owned: Newly public (2025), unproven at scale, heavy debt, MSFT concentration. Trigger: 2-3 Qs of public financials."},
    {"ticker": "ON", "name": "ON Semi", "tier": "TIER 1", "score": 4.0, "note": "Power semis for DCs, SiC for efficiency. Not owned: Primarily automotive. Trigger: DC power revenue reaches 15%+ of total."},
    {"ticker": "SNPS", "name": "Synopsys", "tier": "TIER 1", "score": 4.5, "note": "EDA tools, every AI chip designed here. AI-enhanced workflows. Not owned: Indirect exposure, Ansys integration risk. Solid long-term pick-and-shovel."},
    {"ticker": "LRCX", "name": "Lam Research", "tier": "TIER 1", "score": 4.5, "note": "Semi etch/deposition, HBM capacity buildout. Not owned: Similar thesis to AMAT. Trigger: swap for AMAT if HBM spend accelerates (Lam has higher HBM exposure)."},
    {"ticker": "GLW", "name": "Corning", "tier": "TIER 2", "score": 4.5, "note": "Optical fiber/connectivity for AI DCs. 800G/1.6T transceivers. Not owned: Revenue mix still heavy display/telecom. Trigger: optical segment reaches 30%+ revenue."},
    {"ticker": "KEYS", "name": "Keysight Tech", "tier": "TIER 2", "score": 4.0, "note": "Test/measurement for high-speed networking. Not owned: Cyclical order recovery underway. Trigger: order recovery confirmation."},
    {"ticker": "TEL", "name": "TE Connectivity", "tier": "TIER 2", "score": 4.0, "note": "DC connectors/sensors, competes with APH. Not owned: Prefer APH for higher AI concentration. Trigger: swap if APH valuation stretches."},
    {"ticker": "SNDR", "name": "Schneider (ADR)", "tier": "TIER 2", "score": 4.0, "note": "Global DC power/cooling. Competes with VRT/ETN. Not owned: ADR liquidity concerns. Trigger: European DC buildout acceleration."},
    {"ticker": "NXPI", "name": "NXP Semi", "tier": "TIER 2", "score": 3.5, "note": "Automotive AI semis (ADAS). Not owned: Slower adoption curve than DC. Trigger: autonomous driving investment acceleration."},
    {"ticker": "JBL", "name": "Jabil", "tier": "TIER 2", "score": 3.5, "note": "AI server rack contract mfg. Not owned: Low margin, limited pricing power. Trigger: deep value basis only."},
    {"ticker": "NRG", "name": "NRG Energy", "tier": "TIER 3", "score": 4.5, "note": "Power gen + retail electricity. DC PPA pipeline growing. Not owned: Higher leverage than VST/CEG. Trigger: pullback below $80 or PPA acceleration."},
    {"ticker": "DLR", "name": "Digital Realty", "tier": "TIER 3", "score": 4.5, "note": "Largest DC REIT. Long-term hyperscaler leases. 3%+ yield. Not owned: REIT limits capital appreciation in growth strategy. Trigger: rate-sensitive environment needing defensive yield."},
    {"ticker": "EQIX", "name": "Equinix", "tier": "TIER 3", "score": 4.5, "note": "Global DC colocation REIT. AI increasing density/pricing. Not owned: Same REIT concern, premium valuation. Consider for DC operator exposure."},
    {"ticker": "SO", "name": "Southern Company", "tier": "TIER 3", "score": 4.0, "note": "Regulated utility, SE US (prime DC location). Vogtle nuclear. 3.5%+ yield. Not owned: Regulated growth too slow. Trigger: need lower beta."},
    {"ticker": "EMR", "name": "Emerson Electric", "tier": "TIER 3", "score": 4.5, "note": "Industrial automation + DC mgmt SW (AspenTech). Not owned: AI indirect, transformation underway. Trigger: DC mgmt revenue becomes material."},
    {"ticker": "AES", "name": "AES Corp", "tier": "TIER 3", "score": 4.0, "note": "Renewable developer, battery storage. Not owned: High debt, EM exposure. Trigger: deleveraging + DC PPA wins."},
    {"ticker": "SMR", "name": "NuScale Power", "tier": "TIER 3", "score": 3.0, "note": "Small modular reactor tech. Long-term AI power solution. Not owned: Pre-revenue, speculative. Trigger: SMR regulatory progress only."},
    {"ticker": "PRIM", "name": "Primoris Services", "tier": "TIER 3", "score": 4.0, "note": "Infrastructure contractor, growing DC backlog. Not owned: Prefer PWR. Trigger: specific DC contract wins or PWR overvalued."},
    {"ticker": "UNP", "name": "Union Pacific", "tier": "TIER 3", "score": 3.0, "note": "Transports materials for DC buildout. Not owned: Too indirect. Extreme second-derivative play only."},
    {"ticker": "SKHHY", "name": "SK Hynix (ADR/OTC)", "tier": "TIER 1", "score": 5.0,
     "note": "Best HBM pure-play: ~57% HBM share, pioneered the tech, strong NVDA ties. Not owned: Korea-listed, only thin OTC ADR (SKHHY) in US, hard to hold cleanly in a 401k. Up ~50% YTD 2026. Access via a US memory ETF is the practical route. Trigger: pullback + confirmed 401k tradability."},
    {"ticker": "SSNLF", "name": "Samsung Elec (OTC)", "tier": "TIER 1", "score": 4.0,
     "note": "HBM3/HBM4 maker closing gap on SK Hynix; some analysts call it undervalued vs MU. Not owned: Korea-listed, OTC only, and a diluted conglomerate (phones/displays/foundry). Up ~60% YTD 2026. Trigger: cleaner US access or memory-only spinoff."},
    {"ticker": "SNDK", "name": "SanDisk", "tier": "TIER 1", "score": 4.0,
     "note": "Pure NAND/SSD, spun from WDC in 2025. Not owned: +183% YTD 2026 (run too far, too fast); NAND is more commoditized than HBM/DRAM and the shortage is supply-cut driven. Trigger: meaningful pullback or evidence of durable NAND pricing power."},
    {"ticker": "WDC", "name": "Western Digital", "tier": "TIER 1", "score": 3.5,
     "note": "Post-spin, essentially an HDD pure-play; benefits from AI data-storage volumes. Not owned: +80% YTD 2026; second-derivative, lower-margin, commoditized. Trigger: HDD pricing strength sustained, or as a cheaper memory-adjacent proxy."},
]

AI_MANDATE_MD = r"""## Investment Policy Statement & Mandate

# Investment Policy Statement & Mandate

AI Infrastructure Equity Strategy, v2.0 | February 6, 2026

## Strategy Overview

| Strategy Name | AI Infrastructure Equity Strategy |
|---|---|
| Objective | Long-term capital appreciation by investing across the full AI infrastructure supply chain |
| Style | Thematic / Tactical Growth |
| Primary Benchmark | Nasdaq-100 Index (NDX) |
| Reference Benchmark | Indxx Artificial Intelligence & Big Data Index |
| Universe | US-listed equities (including ADRs) across the AI infrastructure ecosystem |
| Holdings | 30-50 positions |
| SMA Minimum | $100,000 ($250K+ recommended) |
| SMA Manager Fee | 0.55% under $250K | 0.50% $250K-$1M | 0.40% $1M+ | 0.35% $5M+ |
| ETF Expense Ratio | 0.65% |
| Inception | February 10, 2026 |

## Investment Universe: Tier Definitions

| Tier | Definition | Examples |
|---|---|---|
| Hyperscalers | Companies making direct AI infrastructure capex investments at scale | Alphabet, Meta, Oracle |
| Tier 1: Direct | Companies receiving direct revenue from hyperscaler AI capex (GPUs, custom silicon, servers, networking, memory) | NVIDIA, Broadcom, TSMC |
| Tier 2: Secondary | Broader DC infrastructure supply chain (power mgmt, thermal, connectors, semi equipment) | Vertiv, Eaton, ASML |
| Tier 3: Tertiary | Downstream effects (power generation, utilities, grid construction, fuel supply) | Vistra, Constellation Energy, Quanta Services |

## Tier Allocation Guardrails

| Tier | Minimum | Target | Maximum |
|---|---|---|---|
| Hyperscalers | 10% | 25% | 45% |
| Tier 1: Direct | 10% | 25% | 45% |
| Tier 2: Secondary | 10% | 25% | 45% |
| Tier 3: Tertiary | 10% | 25% | 45% |
| Cash | 0% | 0-2% | 5% |

Equal 25% targets across all four tiers provide maximum tactical flexibility. A PM who becomes more constructive on power/grid (Tier 3) over semiconductors (Tier 1) can rotate up to 45% into Tier 3 while maintaining 10% minimum in Tier 1. Max allocations are soft guidelines that can be exceeded with documented rationale and IC awareness. Minimum allocations are hard constraints.

## Position Sizing Rules

| Parameter | Constraint | Rationale |
|---|---|---|
| Max Individual (at cost) | 10% | Prevents single-stock concentration |
| Max Drift Before Rebalance | 12.5% at market value | 2.5% drift tolerance, forces trimming above cap |
| Min Individual (at cost) | 2% | Meaningful contribution; no "toe-in-the-water" positions |
| Max vs. NDX Weight | 1.5x NDX weight | Applies only to NDX constituents; non-NDX names use 10% absolute cap |
| Holdings Count | 30-50 | Diversification with conviction weighting |
| Sizing Driver | Conviction Score | Scores 8-10 → 5-10%; 6-7 → 3-5%; 5-6 → 2-3% |

## Turnover & Trading

| Annual Turnover Target | 50-80% |
|---|---|
| Annual Turnover Cap | 120% (above requires CIO approval) |
| Rebalancing | Quarterly (Mar, Jun, Sep, Dec) + event-driven |
| Tax-Loss Harvesting (SMA) | Active, ongoing, 30-day wash sale compliance |
| New Position Entry | Minimum 2% initial allocation |
| Position Exit | Full exit or reduce to 2% min; no orphan positions |

## Risk Parameters

| Metric | Target/Limit | Monitoring |
|---|---|---|
| Tracking Error vs. NDX | 3-8% annualized | Monthly |
| Beta vs. NDX | 0.85-1.15 | Monthly |
| Max Drawdown Flag | Flag if exceeds benchmark by 500+ bps | Daily |
| Max GICS Technology | 70% | Quarterly |
| Liquidity Floor | $10M+ avg daily volume | At entry |

## Governance

IC Review: Quarterly portfolio review covering positioning, attribution, risk metrics, tier allocations, conviction score updates. Annual mandate review.

Breach Protocol: Hard constraint breach → 5 business days to remediate. Market-driven drift → 10 business days. All trades require brief rationale in trading journal."""

AI_CONVICTION_MD = r"""## Conviction Scoring Framework

# Conviction Scoring Framework

Systematic approach to position sizing that drives portfolio weights and explains benchmark deviations

## Framework Overview

Every holding and watch list name receives a Conviction Score on a 1-10 scale. This score directly drives target portfolio weight and provides documented rationale for overweight/underweight positions vs. benchmark. Updated quarterly or event-driven for material changes (earnings, M&A, regulation).

## Scoring Components

| Component | Weight | What It Measures |
|---|---|---|
| 1. Fundamental Strength | 35% | Valuation (P/E, PEG, EV/EBITDA vs. growth), balance sheet quality, profitability (ROIC, ROE, margins), FCF generation |
| 2. Competitive Positioning | 30% | Market share in AI supply chain role, barriers to entry, technology moat (patents, architecture lock-in), pricing power |
| 3. AI Revenue Visibility | 20% | % of revenue tied to AI capex, order backlog size/duration, customer concentration, forward booking clarity |
| 4. Tier Outlook | 15% | Forward 12-month outlook for company's tier/sub-segment. Captures regime preferences (e.g., favoring energy over hyperscalers). Current ranking: Tier 1 > Tier 3 > Tier 2 > Hyperscalers, reflecting near-term caution on the pace and ROI of hyperscaler capex commitments |

## Score-to-Weight Mapping

| Score | Level | Target Weight | Interpretation |
|---|---|---|---|
| 9-10 | Maximum Conviction | 7-10% | Core position. Best risk/reward. Overweight vs. benchmark. |
| 7-8 | High Conviction | 4-6% | Strong position. Favorable fundamentals + outlook. Likely overweight. |
| 5-6 | Moderate Conviction | 2-4% | Solid thematic exposure but some concerns. Benchmark-ish weight. |
| 3-4 | Watch List | 0% | Interesting but not investable today. Needs catalyst or better entry. |
| 1-2 | Avoid | N/A | Fundamental or structural concerns. Remove from watch list. |

## Scoring Example: NVDA

| Fundamental Strength (35%) | 8/10 | PEG 0.73 (cheapest large-cap semi), 73% gross margins, $45B net cash. P/E optically rich at 43x though growth justifies it. |
|---|---|---|
| Competitive Positioning (30%) | 9/10 | 85%+ GPU market share for AI training. CUDA lock-in. Custom ASIC competition real but not displacing GPUs yet. |
| AI Revenue Visibility (20%) | 10/10 | 95%+ data center revenue is AI. Blackwell demand exceeds supply into H2 2026. Every hyperscaler is a customer. |
| Tier Outlook (15%) | 8/10 | Tier 1 ranked highest in current outlook. Direct beneficiaries of sustained capex regardless of hyperscaler ROI debates. |
| Weighted Score | 8.7 | → Maximum Conviction → Target Weight: 8% |

## Using Scores to Explain Benchmark Deviations

Every overweight/underweight traces to the conviction score. Example client language:"""


# ============================================================
# AI STRATEGY HELPERS: persistence + model-portfolio performance
# ============================================================
import json as _json
import os as _os

AI_PORTFOLIO_FILE = "ai_portfolio_state.json"

def ai_load_portfolio():
    """Load the working portfolio. Starts from the embedded default and
    overlays any saved edits. Saved edits survive reruns and app sleeps."""
    if _os.path.exists(AI_PORTFOLIO_FILE):
        try:
            with open(AI_PORTFOLIO_FILE) as f:
                saved = _json.load(f)
            if isinstance(saved, list) and saved:
                return saved
        except Exception:
            pass
    # deep copy of embedded default
    return [dict(h) for h in AI_PORTFOLIO]

def ai_save_portfolio(holdings):
    """Persist the working portfolio to disk."""
    try:
        with open(AI_PORTFOLIO_FILE, "w") as f:
            _json.dump(holdings, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save: {e}")
        return False

def ai_reset_portfolio():
    """Delete saved edits and revert to the embedded default."""
    try:
        if _os.path.exists(AI_PORTFOLIO_FILE):
            _os.remove(AI_PORTFOLIO_FILE)
        return True
    except Exception:
        return False



AI_BENCH_FILE = "ai_bench_state.json"

def ai_load_bench():
    """Load the working bench list. Starts from embedded AI_BENCH and overlays saved edits."""
    if _os.path.exists(AI_BENCH_FILE):
        try:
            with open(AI_BENCH_FILE) as f:
                saved = _json.load(f)
            if isinstance(saved, list) and saved:
                return saved
        except Exception:
            pass
    return [dict(b) for b in AI_BENCH]

def ai_save_bench(bench_list):
    """Persist the working bench list to disk."""
    try:
        with open(AI_BENCH_FILE, "w") as f:
            _json.dump(bench_list, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save bench: {e}")
        return False

def ai_reset_bench():
    """Delete saved bench edits and revert to the embedded default."""
    try:
        if _os.path.exists(AI_BENCH_FILE):
            _os.remove(AI_BENCH_FILE)
        return True
    except Exception:
        return False

@st.cache_data(ttl=900)
def ai_fetch_prices(tickers, inception):
    """Fetch current price and inception-date price for each ticker.
    Model performance = target-weighted price change since inception.
    Returns dict: ticker -> {price_now, price_incept, ret_pct} (None on failure)."""
    out = {}
    if not tickers:
        return out
    try:
        # batch download from a few days before inception through today
        start = (pd.Timestamp(inception) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        data = yf.download(list(tickers), start=start, progress=False, auto_adjust=True)
        # handle single vs multi ticker shape
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']] if 'Close' in data else data
            if len(tickers) == 1:
                close.columns = list(tickers)
        incept_ts = pd.Timestamp(inception)
        for t in tickers:
            try:
                series = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
                if series.empty:
                    out[t] = {"price_now": None, "price_incept": None, "ret_pct": None}
                    continue
                # inception price = first available on/after inception date
                on_or_after = series[series.index >= incept_ts]
                p0 = float(on_or_after.iloc[0]) if not on_or_after.empty else float(series.iloc[0])
                p1 = float(series.iloc[-1])
                ret = (p1 / p0 - 1) * 100 if p0 > 0 else None
                out[t] = {"price_now": p1, "price_incept": p0, "ret_pct": ret}
            except Exception:
                out[t] = {"price_now": None, "price_incept": None, "ret_pct": None}
    except Exception:
        for t in tickers:
            out[t] = {"price_now": None, "price_incept": None, "ret_pct": None}
    return out

def ai_compute_performance(holdings, inception, benchmark="^NDX"):
    """Model-portfolio return: sum(target_weight_i * stock_return_i), normalized
    by the total invested weight. Returns (strategy_ret, bench_ret, per_holding)."""
    tickers = [h['ticker'] for h in holdings if h.get('ticker')]
    prices = ai_fetch_prices(tuple(tickers), inception)
    bench_prices = ai_fetch_prices((benchmark,), inception)
    bench_ret = bench_prices.get(benchmark, {}).get('ret_pct')

    per = []
    weighted_sum = 0.0
    weight_total = 0.0
    for h in holdings:
        t = h['ticker']
        w = h.get('target_weight') or 0.0
        pr = prices.get(t, {})
        ret = pr.get('ret_pct')
        contrib = None
        if ret is not None and w:
            contrib = (w / 100.0) * ret
            weighted_sum += contrib
            weight_total += w / 100.0
        per.append({**h, "price_now": pr.get('price_now'),
                    "price_incept": pr.get('price_incept'),
                    "ret_pct": ret, "contribution": contrib})
    strategy_ret = (weighted_sum / weight_total) if weight_total > 0 else None
    return strategy_ret, bench_ret, per



# Navigation menu


@st.cache_data(ttl=900)
def ai_equity_curve(holdings_key, inception, benchmark="^NDX", extra_bench="VOO"):
    """Build the daily weighted equity curve of the portfolio since inception,
    plus benchmark curves and the portfolio drawdown series.

    holdings_key is a tuple of (ticker, weight) pairs so the cache invalidates
    when weights change. Returns a dict with DataFrames/levels, or {} on failure.

    Method: each holding indexed to its first available close on/after inception,
    weighted by target weight; cash (100 - sum weights) held flat. Portfolio,
    NDX and VOO are all normalized to 1.0 at inception so they overlay cleanly.
    """
    holdings = list(holdings_key)
    tickers = [t for t, w in holdings if t]
    if not tickers:
        return {}
    try:
        start = (pd.Timestamp(inception) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        syms = list(dict.fromkeys(tickers + [benchmark, extra_bench]))
        data = yf.download(syms, start=start, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return {}
        close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        if not isinstance(close, pd.DataFrame):
            close = close.to_frame()
        incept_ts = pd.Timestamp(inception)
        # restrict to on/after inception
        close = close[close.index >= incept_ts]
        if close.empty:
            return {}

        # portfolio curve
        port = pd.Series(0.0, index=close.index)
        wsum = 0.0
        for t, w in holdings:
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if s.empty:
                continue
            port = port.add((close[t] / s.iloc[0]) * (w / 100.0), fill_value=0)
            wsum += w / 100.0
        cash_w = max(0.0, 1.0 - wsum)
        denom = wsum + cash_w
        if denom <= 0:
            return {}
        port = (port + cash_w) / denom

        def _norm(sym):
            if sym in close.columns:
                s = close[sym].dropna()
                if not s.empty:
                    return close[sym] / s.iloc[0]
            return pd.Series(index=close.index, dtype=float)

        ndx = _norm(benchmark)
        voo = _norm(extra_bench)

        dd = (port / port.cummax() - 1.0) * 100.0
        max_dd = float(dd.min()) if not dd.empty else 0.0
        max_dd_date = dd.idxmin() if not dd.empty else None
        peak_date = port[:max_dd_date].idxmax() if max_dd_date is not None else None
        # recovery check
        rec_date = None
        if peak_date is not None and max_dd_date is not None:
            peak_val = port[peak_date]
            after = port[max_dd_date:]
            rec = after[after >= peak_val]
            rec_date = rec.index[0] if not rec.empty else None

        rets = port.pct_change().dropna()
        ann_vol = float(rets.std() * (252 ** 0.5) * 100) if not rets.empty else 0.0

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in port.index],
            "port": [round((v - 1) * 100, 3) for v in port.values],
            "ndx": [round((v - 1) * 100, 3) if pd.notna(v) else None for v in ndx.reindex(port.index).values],
            "voo": [round((v - 1) * 100, 3) if pd.notna(v) else None for v in voo.reindex(port.index).values],
            "drawdown": [round(v, 3) if pd.notna(v) else None for v in dd.values],
            "max_dd": round(max_dd, 2),
            "max_dd_date": max_dd_date.strftime("%Y-%m-%d") if max_dd_date is not None else None,
            "peak_date": peak_date.strftime("%Y-%m-%d") if peak_date is not None else None,
            "rec_date": rec_date.strftime("%Y-%m-%d") if rec_date is not None else None,
            "ann_vol": round(ann_vol, 1),
            "port_total": round((float(port.iloc[-1]) - 1) * 100, 2),
            "ndx_total": round((float(ndx.reindex(port.index).iloc[-1]) - 1) * 100, 2) if benchmark in close.columns else None,
            "voo_total": round((float(voo.reindex(port.index).iloc[-1]) - 1) * 100, 2) if extra_bench in close.columns else None,
        }
    except Exception:
        return {}

def ai_risk_stats(ec, rf_annual=0.043):
    """Derive risk statistics from the equity-curve dict produced by
    ai_equity_curve(). Returns a dict of fact-sheet metrics. All derived from
    the daily MODEL curve, so they describe the strategy allocation, not fills.

    Computes the benchmark-relative stats (beta, alpha, tracking error, info
    ratio, up/down capture, correlation) against BOTH the NDX (primary benchmark)
    and the S&P 500 via VOO (the barometer most investors anchor to). S&P-relative
    figures are suffixed _sp."""
    import numpy as _np
    stats = {}
    port = ec.get("port") or []
    ndx = ec.get("ndx") or []
    voo = ec.get("voo") or []
    if len(port) < 5:
        return stats
    lvl = _np.array([1 + p / 100 for p in port], dtype=float)
    rets = _np.diff(lvl) / lvl[:-1]
    ann = 252
    mean_d = rets.mean()
    std_d = rets.std(ddof=1)
    ann_ret_geo = (lvl[-1] / lvl[0]) ** (ann / len(rets)) - 1
    ann_vol = std_d * _np.sqrt(ann)
    rf_d = (1 + rf_annual) ** (1 / ann) - 1
    sharpe = ((mean_d - rf_d) / std_d * _np.sqrt(ann)) if std_d > 0 else None
    downside = rets[rets < 0]
    sortino = ((mean_d - rf_d) / downside.std(ddof=1) * _np.sqrt(ann)) if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    max_dd = ec.get("max_dd")
    calmar = (ann_ret_geo * 100 / abs(max_dd)) if max_dd else None

    def _vs_benchmark(bcum):
        """beta, alpha, TE, IR, up/down capture, correlation vs a cumulative-% series."""
        out = dict(beta=None, alpha=None, te=None, ir=None, up=None, down=None, corr=None)
        vals = [x for x in bcum if x is not None]
        if len(vals) < 5:
            return out
        blvl = _np.array([1 + (x or 0) / 100 for x in bcum], dtype=float)
        bret = _np.diff(blvl) / blvl[:-1]
        if len(bret) != len(rets):
            return out
        cov = _np.cov(rets, bret)
        if cov[1, 1] > 0:
            out["beta"] = cov[0, 1] / cov[1, 1]
        out["corr"] = _np.corrcoef(rets, bret)[0, 1]
        active = rets - bret
        te = active.std(ddof=1) * _np.sqrt(ann)
        out["te"] = te
        out["ir"] = (active.mean() * ann) / te if te > 0 else None
        b_ann_geo = (blvl[-1] / blvl[0]) ** (ann / len(bret)) - 1
        if out["beta"] is not None:
            out["alpha"] = (ann_ret_geo - rf_annual) - out["beta"] * (b_ann_geo - rf_annual)
        up = bret > 0; dn = bret < 0
        if up.sum() > 0 and bret[up].sum() != 0:
            out["up"] = rets[up].sum() / bret[up].sum() * 100
        if dn.sum() > 0 and bret[dn].sum() != 0:
            out["down"] = rets[dn].sum() / bret[dn].sum() * 100
        return out

    n = _vs_benchmark(ndx)   # vs NDX
    s = _vs_benchmark(voo)   # vs S&P 500 (VOO)
    pos = (rets > 0).sum()
    win_rate = pos / len(rets) * 100

    def r2(x, m=1): return round(x * m, 2) if x is not None else None
    stats.update({
        "ann_return": round(ann_ret_geo * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": r2(sharpe), "sortino": r2(sortino), "calmar": r2(calmar),
        "max_dd": max_dd,
        # vs NDX (primary)
        "beta": r2(n["beta"]), "alpha": r2(n["alpha"], 100),
        "tracking_error": r2(n["te"], 100), "info_ratio": r2(n["ir"]),
        "up_capture": round(n["up"], 1) if n["up"] is not None else None,
        "down_capture": round(n["down"], 1) if n["down"] is not None else None,
        "correlation": r2(n["corr"]),
        # vs S&P 500 (VOO) — the equity barometer
        "beta_sp": r2(s["beta"]), "alpha_sp": r2(s["alpha"], 100),
        "tracking_error_sp": r2(s["te"], 100), "info_ratio_sp": r2(s["ir"]),
        "up_capture_sp": round(s["up"], 1) if s["up"] is not None else None,
        "down_capture_sp": round(s["down"], 1) if s["down"] is not None else None,
        "correlation_sp": r2(s["corr"]),
        "win_rate": round(win_rate, 1),
        "n_days": len(rets) + 1,
    })
    return stats


def ai_build_factsheet_html(holdings, per, ec, stats, strat_ret, bench_ret,
                            inception, as_of, benchmark_label="Nasdaq-100 (NDX)"):
    """Assemble the fact sheet HTML string from live data."""
    tw = sum(h.get('target_weight') or 0 for h in holdings)
    cash = max(0, 100 - tw)
    active = (strat_ret - bench_ret) if (strat_ret is not None and bench_ret is not None) else None

    # tier allocation
    tier_w = {}
    for h in holdings:
        tier_w[h['tier']] = tier_w.get(h['tier'], 0) + (h.get('target_weight') or 0)
    tier_rows = ""
    tlabel = {"HYPER": "Hyperscalers", "TIER 1": "Tier 1: Direct", "TIER 2": "Tier 2: Secondary", "TIER 3": "Tier 3: Tertiary"}
    for t in ["HYPER", "TIER 1", "TIER 2", "TIER 3"]:
        if tier_w.get(t):
            tier_rows += f"<tr><td>{tlabel[t]}</td><td class='r'>{tier_w[t]:.1f}%</td></tr>"
    if cash > 0.05:
        tier_rows += f"<tr><td>Cash</td><td class='r'>{cash:.1f}%</td></tr>"

    # top 10 holdings by weight
    sorted_h = sorted(per, key=lambda x: (x.get('target_weight') or 0), reverse=True)
    top10 = sorted_h[:10]
    hold_rows = ""
    for h in top10:
        ret = h.get('ret_pct')
        rs = f"{ret:+.1f}%" if ret is not None else "n/a"
        hold_rows += (f"<tr><td class='tk'>{h['ticker']}</td><td>{h['name']}</td>"
                      f"<td>{h['tier'].replace('TIER ','T').replace('HYPER','Hyper')}</td>"
                      f"<td class='r'>{(h.get('target_weight') or 0):.1f}%</td>"
                      f"<td class='r'>{rs}</td></tr>")

    # performance table
    def pct(v): return f"{v:+.2f}%" if v is not None else "n/a"
    perf_rows = (
        f"<tr><td>Strategy (model)</td><td class='r'>{pct(strat_ret)}</td></tr>"
        f"<tr><td>{benchmark_label}</td><td class='r'>{pct(bench_ret)}</td></tr>"
        f"<tr><td>Active return</td><td class='r {'pos' if (active or 0)>=0 else 'neg'}'>{pct(active)}</td></tr>"
    )
    if ec.get("voo_total") is not None:
        perf_rows += f"<tr><td>S&amp;P 500 (VOO)</td><td class='r'>{pct(ec['voo_total'])}</td></tr>"

    def stat(v, suf=""): return f"{v}{suf}" if v is not None else "n/a"
    # Portfolio-level (benchmark-agnostic) stats
    risk_rows = (
        f"<tr><td>Annualized return</td><td class='r'>{stat(stats.get('ann_return'),'%')}</td></tr>"
        f"<tr><td>Annualized volatility (std dev)</td><td class='r'>{stat(stats.get('ann_vol'),'%')}</td></tr>"
        f"<tr><td>Sharpe ratio</td><td class='r'>{stat(stats.get('sharpe'))}</td></tr>"
        f"<tr><td>Sortino ratio</td><td class='r'>{stat(stats.get('sortino'))}</td></tr>"
        f"<tr><td>Max drawdown</td><td class='r neg'>{stat(stats.get('max_dd'),'%')}</td></tr>"
        f"<tr><td>Calmar ratio</td><td class='r'>{stat(stats.get('calmar'))}</td></tr>"
        f"<tr><td>Win rate (daily)</td><td class='r'>{stat(stats.get('win_rate'),'%')}</td></tr>"
    )
    # Two-benchmark relative stats: NDX (primary) and S&P 500 via VOO (the equity barometer)
    bench_rows = (
        "<tr><th>Metric</th><th class='r'>vs NDX</th><th class='r'>vs S&amp;P 500</th></tr>"
        f"<tr><td>Beta</td><td class='r'><strong>{stat(stats.get('beta'))}</strong></td><td class='r'><strong>{stat(stats.get('beta_sp'))}</strong></td></tr>"
        f"<tr><td>Alpha (annualized)</td><td class='r'>{stat(stats.get('alpha'),'%')}</td><td class='r'>{stat(stats.get('alpha_sp'),'%')}</td></tr>"
        f"<tr><td>Correlation</td><td class='r'>{stat(stats.get('correlation'))}</td><td class='r'>{stat(stats.get('correlation_sp'))}</td></tr>"
        f"<tr><td>Tracking error</td><td class='r'>{stat(stats.get('tracking_error'),'%')}</td><td class='r'>{stat(stats.get('tracking_error_sp'),'%')}</td></tr>"
        f"<tr><td>Information ratio</td><td class='r'>{stat(stats.get('info_ratio'))}</td><td class='r'>{stat(stats.get('info_ratio_sp'))}</td></tr>"
        f"<tr><td>Up capture</td><td class='r'>{stat(stats.get('up_capture'),'%')}</td><td class='r'>{stat(stats.get('up_capture_sp'),'%')}</td></tr>"
        f"<tr><td>Down capture</td><td class='r'>{stat(stats.get('down_capture'),'%')}</td><td class='r'>{stat(stats.get('down_capture_sp'),'%')}</td></tr>"
    )

    # sparkline points (cumulative return) as inline SVG polyline
    pts = ec.get("port") or []
    spark = ""
    if len(pts) > 2:
        lo, hi = min(pts), max(pts)
        rng = (hi - lo) or 1
        W, H = 320, 60
        coords = []
        for i, v in enumerate(pts):
            x = i / (len(pts) - 1) * W
            y = H - (v - lo) / rng * H
            coords.append(f"{x:.1f},{y:.1f}")
        spark = (f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
                 f"<polyline fill='none' stroke='#2dd4bf' stroke-width='2' points='{' '.join(coords)}'/></svg>")

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1f2b;margin:0;background:#fff;font-size:12px;line-height:1.45}
    .sheet{max-width:900px;margin:0 auto;padding:28px 32px}
    .hdr{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:3px solid #0c2a45;padding-bottom:14px;margin-bottom:16px}
    .hdr h1{font-size:1.5rem;margin:0;color:#0c2a45;letter-spacing:-.3px}
    .hdr .sub{color:#5b6675;font-size:.82rem;margin-top:3px}
    .hdr .asof{text-align:right;font-size:.72rem;color:#5b6675}
    .hdr .asof .big{font-size:1.4rem;color:#0c7a4b;font-weight:700;display:block}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
    .grid3{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:18px}
    h2{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:#0c2a45;border-bottom:1px solid #d8dee8;padding-bottom:4px;margin:14px 0 6px}
    table{width:100%;border-collapse:collapse}
    td{padding:3px 2px;border-bottom:1px solid #eef1f5;font-size:.78rem}
    td.r{text-align:right;font-variant-numeric:tabular-nums;font-feature-settings:'tnum'}
    td.tk{font-weight:700;font-family:ui-monospace,Menlo,monospace}
    .pos{color:#0c7a4b}.neg{color:#b3261e}
    .box{background:#f6f8fb;border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px;margin-bottom:10px}
    .kv{display:flex;justify-content:space-between;font-size:.78rem;padding:2px 0}
    .kv .k{color:#5b6675}
    .narr{font-size:.78rem;color:#2b3340;margin:4px 0}
    .foot{margin-top:18px;border-top:1px solid #d8dee8;padding-top:10px;font-size:.62rem;color:#8a93a3;line-height:1.5}
    .pill{display:inline-block;background:#0c2a45;color:#fff;font-size:.6rem;padding:1px 7px;border-radius:3px;letter-spacing:.5px}
    """

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>AI Infrastructure Equity Strategy — Fact Sheet</title>
<style>{css}</style></head><body><div class='sheet'>
<div class='hdr'>
  <div><h1>AI Infrastructure Equity Strategy</h1>
    <div class='sub'>Thematic / Tactical Growth Equity &nbsp;·&nbsp; <span class='pill'>MODEL</span> Benchmark: {benchmark_label}</div></div>
  <div class='asof'><span class='big'>{strat_ret:+.1f}%</span>Strategy since inception<br>{inception} → {as_of}</div>
</div>

<div class='grid3'>
  <div>
    <h2>Strategy Overview</h2>
    <div class='kv'><span class='k'>Objective</span><span>Capital appreciation</span></div>
    <div class='kv'><span class='k'>Style</span><span>Thematic / tactical growth</span></div>
    <div class='kv'><span class='k'>Universe</span><span>US-listed equities + ADRs</span></div>
    <div class='kv'><span class='k'>Holdings</span><span>{len(holdings)} (target 30-50)</span></div>
    <div class='kv'><span class='k'>Inception</span><span>{inception}</span></div>
    <div class='kv'><span class='k'>Benchmark</span><span>NDX</span></div>
    <div class='kv'><span class='k'>Reference</span><span>Indxx AI &amp; Big Data</span></div>
  </div>
  <div>
    <h2>Fees</h2>
    <div class='kv'><span class='k'>SMA &lt; $250K</span><span>0.55%</span></div>
    <div class='kv'><span class='k'>SMA $250K-1M</span><span>0.50%</span></div>
    <div class='kv'><span class='k'>SMA $1M+</span><span>0.40%</span></div>
    <div class='kv'><span class='k'>SMA $5M+</span><span>0.35%</span></div>
    <div class='kv'><span class='k'>ETF</span><span>0.65%</span></div>
    <div class='kv'><span class='k'>SMA minimum</span><span>$100,000</span></div>
  </div>
  <div>
    <h2>Performance (cumulative)</h2>
    <table>{perf_rows}</table>
    <div style='margin-top:6px'>{spark}</div>
    <div class='narr' style='font-size:.66rem;color:#8a93a3'>Cumulative return since inception</div>
  </div>
</div>

<div class='grid2' style='margin-top:8px'>
  <div>
    <h2>Risk Statistics</h2>
    <table>{risk_rows}</table>
    <h2 style='margin-top:12px'>Benchmark-Relative (vs NDX &amp; S&amp;P 500)</h2>
    <table>{bench_rows}</table>
    <div class='narr' style='font-size:.66rem;color:#8a93a3'>Annualized from daily model returns over {stats.get('n_days','?')} trading days. Risk-free 4.3%. Beta vs the S&amp;P 500 is the barometer most investors anchor to; beta vs NDX is the strategy's stated benchmark.</div>
  </div>
  <div>
    <h2>Tier Allocation</h2>
    <table>{tier_rows}</table>
    <h2 style='margin-top:14px'>Top 10 Holdings</h2>
    <table>{hold_rows}</table>
  </div>
</div>

<div style='margin-top:10px'>
  <h2>Investment Mandate &amp; Process</h2>
  <div class='narr'>The strategy invests across the full AI infrastructure supply chain through a four-tier conviction framework: Hyperscalers (capex spenders), Tier 1 Direct (GPUs, custom silicon, memory, networking), Tier 2 Secondary (power, cooling, connectors, semi equipment), and Tier 3 Tertiary (power generation, utilities, grid, fuel). Position sizing is driven by a 1-10 conviction score across four factors: Fundamental Strength (35%), Competitive Positioning (30%), AI Revenue Visibility (20%), and Tier Outlook (15%). Hard constraints: 8% max single position (12.5% drift trigger), 2% minimum, tier guardrails (10% floor / 45% ceiling per tier), 30-50 holdings, 0-5% cash. Rebalanced quarterly and on material events. Tracking-error target 3-8% annualized; max GICS Technology 70%.</div>
</div>

<div class='foot'>
<strong>MODEL / HYPOTHETICAL PERFORMANCE.</strong> Performance and risk statistics are derived from a model portfolio: target weights applied to each holding's actual price history since {inception}, with cash held flat and all figures indexed to inception. They reflect the strategy allocation, not the results of any actual account, and do not include the impact of advisory fees, transaction costs, taxes, or the timing of actual purchases and sales. Model and hypothetical results have inherent limitations, including hindsight and the benefit of being designed with knowledge of prior market behavior; no representation is made that any account will achieve similar results. Past performance does not guarantee future results. Investing involves risk including possible loss of principal. This material is for informational purposes only, is not investment advice, and is not an offer or solicitation. Holdings and allocations are subject to change. Generated {as_of} from live data.
</div>
</div></body></html>"""
    return html

selected = option_menu(
    menu_title=None,
    options=["Trading Hub", "Market Dashboard", "Signal History", "Backtest", "Trade Log", "Performance", "Chart Analysis", "Options Chain", "Premium Seller", "AI Strategy", "Macro Dashboard"],
    icons=["activity", "speedometer2", "clock-history", "graph-up-arrow", "list-ul", "trophy", "bar-chart", "currency-exchange", "shield-check", "cpu", "globe"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# ========================================
# TRADING HUB
# ========================================

if selected == "Trading Hub":
    st.header("Trading Hub - Multi-Ticker Analysis")
    
    # Market Overview
    st.subheader("Market Overview")
    cols = st.columns(len(TICKERS))
    for i, ticker in enumerate(TICKERS):
        with cols[i]:
            data = market_data[ticker]
            change_color = "green" if data['change'] >= 0 else "red"
            st.markdown(f"""
            <div style="background:#1e1e1e;padding:15px;border-radius:10px;text-align:center;">
                <h3>{ticker}</h3>
                <h2 style="color:white">${data['price']:.2f}</h2>
                <p style="color:{change_color};font-size:18px;">{data['change']:+.2f} ({data['change_pct']:+.2f}%)</p>
                <p style="font-size:12px;">Vol: {data['volume']/1e6:.1f}M</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()

    # ---- AI Portfolio Snapshot ----
    with st.expander("🤖 AI Portfolio Snapshot", expanded=True):
        snap_holdings = ai_load_portfolio()
        with st.spinner("Loading AI portfolio..."):
            s_strat, s_bench, s_per = ai_compute_performance(
                snap_holdings, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Holdings", f"{len(snap_holdings)}")
        sc2.metric("Strategy / incept", f"{s_strat:+.1f}%" if s_strat is not None else "n/a")
        s_delta = (s_strat - s_bench) if (s_strat is not None and s_bench is not None) else None
        sc3.metric("NDX / incept", f"{s_bench:+.1f}%" if s_bench is not None else "n/a",
                   f"{s_delta:+.1f}% active" if s_delta is not None else None)
        # count names in each entry zone right now
        zone_counts = {"🟢": 0, "🟡": 0, "🔴": 0, "—": 0}
        for p in s_per:
            z, _ = ai_entry_zone(p['ticker'], p.get('price_now'))
            key = z.split()[0] if z and z[0] in "🟢🟡🔴" else "—"
            zone_counts[key] = zone_counts.get(key, 0) + 1
        sc4.metric("In buy zone", f"{zone_counts['🟢']} 🟢",
                   help=f"🟢 at/below secondary · 🟡 below ceiling ({zone_counts['🟡']}) · 🔴 above ceiling ({zone_counts['🔴']})")

        # compact per-holding table with today's move + entry zone
        srows = []
        for p in s_per:
            tk = p['ticker']
            pnow = p.get('price_now')
            zone, tgt = ai_entry_zone(tk, pnow)
            # today's move from 2-day history via the shared market fetch when available
            srows.append({
                "Tier": p['tier'], "Ticker": tk,
                "Price": f"${pnow:,.2f}" if pnow is not None else "n/a",
                "Since Incept": f"{p.get('ret_pct'):+.1f}%" if p.get('ret_pct') is not None else "n/a",
                "Wt": f"{p.get('target_weight') or 0:.1f}%",
                "Zone": zone,
                "Optimal": f"${tgt['optimal']:,.2f}" if tgt else "—",
                "Ceiling": f"${tgt['ceiling']:,.2f}" if tgt else "—",
            })
        tier_order = {"HYPER": 0, "TIER 1": 1, "TIER 2": 2, "TIER 3": 3}
        srows.sort(key=lambda r: (tier_order.get(r["Tier"], 9), r["Ticker"]))
        st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True, height=340)
        st.caption("Zone vs entry targets: 🟢 at/below secondary (buy) · 🟡 below do-not-exceed · 🔴 above ceiling. "
                   "Full detail and the performance chart live on the AI Strategy tab.")

    
    # Market Status
    market_open = is_market_open()
    st.info(f"""
    **Market Status:** {'🟢 OPEN' if market_open else '🔴 CLOSED'}  
    **Current Time (ET):** {datetime.now(ZoneInfo("US/Eastern")).strftime('%I:%M:%S %p')}  
    **Mode:** {TRADING_MODE}  
    **Active Signals:** {len(st.session_state.signal_queue)}  
    **Active Trades:** {len(st.session_state.active_trades)}  
    **Signal Expiration:** {SIGNAL_EXPIRATION_MINUTES} minutes
    """)
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Generate Signals", use_container_width=True):
            generate_signal()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Signals", use_container_width=True):
            st.session_state.signal_queue = []
            save_all_data()
            st.rerun()
    with col3:
        expired_count = expire_old_signals()
        if expired_count > 0:
            st.success(f"Expired {expired_count} signals")
    with col4:
        if st.button("💾 Save All", use_container_width=True):
            save_all_data()
            st.success("Saved!")
    
    st.divider()
    
    # Auto-generate signals if market open
    if market_open:
        generate_signal()
        simulate_exit()
    
    # Display Signals
    st.subheader(f"📊 Trading Signals ({len(st.session_state.signal_queue)} Active)")
    
    if len(st.session_state.signal_queue) == 0:
        st.info("""
        **No active signals.**
        
        Signals generate when:
        - ✅ Market conditions align
        - ✅ Conviction threshold met
        - ✅ Macro filter passed (if enabled)
        
        Try: Generate Signals button or lower Min Conviction in sidebar
        """)
    
    for sig in st.session_state.signal_queue:
        # Calculate age
        signal_age = (datetime.now(ZoneInfo("US/Eastern")) - sig['timestamp']).total_seconds() / 60
        time_left = SIGNAL_EXPIRATION_MINUTES - signal_age
        
        st.markdown(f"""
        <div style="background:#ff6b6b;padding:15px;border-radius:10px;margin-bottom:10px;">
            <h3>🎯 SIGNAL - {sig['type']}</h3>
            <p style="font-size:14px;"><b>Generated:</b> {sig['time']} | <b>Time Left:</b> {time_left:.0f} min | <b>Conviction:</b> {sig['conviction']}/10</p>
            <p style="font-size:16px;"><b>{sig['symbol']}</b> | {sig['action']}</p>
            <p style="font-size:12px;"><b>Strategy:</b> {sig['strategy']}</p>
            <p style="font-size:12px;"><b>Thesis:</b> {sig['thesis']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✅ Take: {sig['id']}", key=f"take_{sig['id']}", use_container_width=True):
                entry_price = market_data[sig['symbol']]['price']
                trade = {
                    'signal_id': sig['id'],
                    'entry_time': datetime.now(ZoneInfo("US/Eastern")),
                    'symbol': sig['symbol'],
                    'action': sig['action'],
                    'size': sig['size'],
                    'entry_price': entry_price,
                    'max_hold': sig.get('max_hold'),
                    'dte': sig.get('dte'),
                    'strategy': sig['strategy'],
                    'thesis': sig['thesis'],
                    'conviction': sig['conviction'],
                    'signal_type': sig['signal_type'],
                    'max_pnl_reached': 0
                }
                st.session_state.active_trades.append(trade)
                
                log_trade(
                    sig['time'], "Open", sig['symbol'], sig['action'], sig['size'],
                    f"${entry_price:.2f}", "Pending", "Open", "Open", sig['id'],
                    entry_numeric=entry_price, dte=sig.get('dte'),
                    strategy=sig['strategy'], thesis=sig['thesis'], max_hold=sig.get('max_hold'),
                    conviction=sig['conviction'], signal_type=sig['signal_type']
                )
                
                # Update history
                for hist_sig in st.session_state.signal_history:
                    if hist_sig['id'] == sig['id']:
                        hist_sig['status'] = 'Taken'
                        break
                
                st.session_state.signal_queue.remove(sig)
                save_all_data()
                st.success("✅ Trade opened!")
                st.rerun()
        
        with col2:
            if st.button(f"❌ Skip: {sig['id']}", key=f"skip_{sig['id']}", use_container_width=True):
                log_trade(
                    sig['time'], "Skipped", sig['symbol'], sig['action'], sig['size'],
                    "---", "---", "---", "Skipped", sig['id'],
                    strategy=sig['strategy'], thesis=sig['thesis'],
                    conviction=sig['conviction'], signal_type=sig['signal_type']
                )
                
                # Update history
                for hist_sig in st.session_state.signal_history:
                    if hist_sig['id'] == sig['id']:
                        hist_sig['status'] = 'Skipped'
                        break
                
                st.session_state.signal_queue.remove(sig)
                save_all_data()
                st.info("Signal skipped")
                st.rerun()
    
    # Display Active Trades
    if st.session_state.active_trades:
        st.subheader(f"📈 Active Trades ({len(st.session_state.active_trades)})")
        for trade in st.session_state.active_trades:
            current_price = market_data[trade['symbol']]['price']
            entry_price = trade['entry_price']
            
            if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
                pnl = (entry_price - current_price) * trade['size']
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            else:
                pnl = (current_price - entry_price) * trade['size']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            minutes_held = (datetime.now(ZoneInfo("US/Eastern")) - trade['entry_time']).total_seconds() / 60
            pnl_color = "green" if pnl >= 0 else "red"
            
            with st.expander(f"{trade['symbol']} - {trade['signal_id']} | P&L: ${pnl:.0f} ({pnl_pct:+.2f}%)"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Entry:** ${entry_price:.2f}")
                col1.write(f"**Current:** ${current_price:.2f}")
                col1.write(f"**Size:** {trade['size']} shares")
                col2.write(f"**Strategy:** {trade['strategy']}")
                col2.write(f"**Conviction:** {trade.get('conviction', 'N/A')}/10")
                col2.write(f"**Signal Type:** {trade.get('signal_type', 'N/A')}")
                col3.write(f"**Time Held:** {minutes_held:.0f} min")
                col3.markdown(f"**P&L:** <span style='color:{pnl_color};font-size:20px;'>${pnl:.0f} ({pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                
                st.write(f"**Thesis:** {trade['thesis']}")
                
                if st.button(f"Close Trade: {trade['signal_id']}", key=f"close_{trade['signal_id']}"):
                    exit_price = current_price
                    
                    if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
                        close_action = 'Buy to Cover'
                    else:
                        close_action = 'Sell'
                    
                    log_trade(
                        datetime.now(ZoneInfo("US/Eastern")).strftime("%m/%d %H:%M"),
                        "Close",
                        trade['symbol'],
                        close_action,
                        trade['size'],
                        f"${entry_price:.2f}",
                        f"${exit_price:.2f}",
                        f"${pnl:.0f}",
                        "Closed (Manual)",
                        trade['signal_id'],
                        entry_numeric=entry_price,
                        exit_numeric=exit_price,
                        pnl_numeric=pnl,
                        dte=trade.get('dte'),
                        strategy=trade.get('strategy'),
                        thesis=trade.get('thesis'),
                        max_hold=trade.get('max_hold'),
                        actual_hold=minutes_held,
                        conviction=trade.get('conviction'),
                        signal_type=trade.get('signal_type')
                    )
                    st.session_state.active_trades.remove(trade)
                    save_all_data()
                    st.success("Trade closed!")
                    st.rerun()

# ========================================
# MARKET DASHBOARD
# ========================================

elif selected == "Market Dashboard":
    st.header("📊 Market Dashboard")
    st.caption("Comprehensive market overview across asset classes, sectors, countries, and factors")

    # Authoritative "Today" reference: the last completed trading day.
    # Everything (Today, MTD, YTD) keys off this single date.
    _ltd = get_last_trading_day()
    if _ltd:
        _ltd_label = _ltd.strftime("%a, %b %d, %Y")
        _today_col_label = _ltd.strftime("%b %d")
        st.info(f"📅 **Today** references the last completed trading day: **{_ltd_label}**. MTD and YTD are measured from this date.")
    else:
        _ltd_label = "unavailable"
        _today_col_label = "Today"
        st.warning("Could not determine the last trading day from data. Showing periods with available data.")
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    with col1:
        view_mode = st.radio("View Mode", ["Standard Periods", "Custom Period"], horizontal=True)
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Custom period selection (only shown if selected)
    if view_mode == "Custom Period":
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col2:
            custom_end = st.date_input("End Date", value=datetime.now())
    
    # Market data configuration - EXPANDED
    MARKET_ETFS = {
        "Equities": {
            "Large Cap": "IVV",
            "Mid Cap": "IJH",
            "Small Cap": "IJR",
            "SMID": "SMMD",
            "All Cap": "ITOT",
            "Developed Markets": "EFA",
            "Emerging Markets": "EEM",
            "World ex-US": "ACWX",
            "World": "ACWI"
        },
        "Fixed Income": {
            "Aggregate": "AGG",
            "Short-Term Treasury": "SHV",
            "Intermediate Treasury": "IEF",
            "Long-Term Treasury": "TLT",
            "TIPS": "TIP",
            "Investment Grade Corp": "LQD",
            "High Yield Corporate": "HYG",
            "Emerging Market Bonds": "EMB",
            "Municipals": "MUB",
            "Mortgage-Backed": "MBB",
            "Floating Rate": "FLOT"
        },
        "Real Assets": {
            "Bitcoin": "IBIT",
            "Gold": "IAU",
            "Silver": "SLV",
            "Commodity Basket": "GSG",
            "Natural Resources": "IGE",
            "Oil": "DBO",
            "Real Estate": "IYR",
            "Infrastructure": "IGF",
            "Timber": "WOOD"
        },
        "S&P Sectors": {
            "Communication Services": "XLC",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Financials": "XLF",
            "Health Care": "XLV",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Technology": "XLK",
            "Utilities": "XLU"
        },
        "Developed Markets": {
            "United States": "SPY",
            "Canada": "EWC",
            "United Kingdom": "EWU",
            "Germany": "EWG",
            "France": "EWQ",
            "Italy": "EWI",
            "Spain": "EWP",
            "Netherlands": "EWN",
            "Switzerland": "EWL",
            "Belgium": "EWK",
            "Sweden": "EWD",
            "Japan": "EWJ",
            "Australia": "EWA",
            "Hong Kong": "EWH",
            "Singapore": "EWS",
            "South Korea": "EWY"
        },
        "Emerging Markets": {
            "China": "MCHI",
            "India": "INDA",
            "Brazil": "EWZ",
            "Mexico": "EWW",
            "South Africa": "EZA",
            "Taiwan": "EWT",
            "Russia": "ERUS",
            "Turkey": "TUR",
            "Indonesia": "EIDO",
            "Thailand": "THD",
            "Malaysia": "EWM",
            "Philippines": "EPHE",
            "Chile": "ECH",
            "Colombia": "GXG",
            "Peru": "EPU",
            "Poland": "EPOL"
        },
        "Factors": {
            "Value": "VLUE",
            "Momentum": "MTUM",
            "Quality": "QUAL",
            "Size (Small Cap)": "SIZE",
            "Low Volatility": "USMV",
            "Dividend": "DVY",
            "Growth": "IVW",
            "High Dividend": "HDV"
        },
        "International Factors": {
            "Intl Value": "IVLU",
            "Intl Growth": "EFG",
            "Intl Momentum": "IMTM",
            "Intl Quality": "IQLT",
            "Intl Low Volatility": "EFAV",
            "Intl Dividend": "IDV",
            "Intl High Dividend": "VYMI",
            "Intl Small Cap": "SCZ",
            "Intl Multi-Factor": "INTF",
            "EAFE Value (Broad)": "EFV"
        }
    }
    
    # Define standard periods
    STANDARD_PERIODS = {
        "Today": 1,
        "MTD": "mtd",
        "YTD": "ytd",
        "1yr": 252,
        "3yr": 756,
        "5yr": 1260,
        "10yr": 2520
    }
    
    @st.cache_data(ttl=300)
    def fetch_multi_period_performance(tickers_dict):
        """Fetch performance data for all periods at once"""
        results = {}

        for category, tickers in tickers_dict.items():
            category_data = []

            for name, ticker in tickers.items():
                try:
                    # Fetch enough data for longest period
                    t = yf.Ticker(ticker)
                    # Get more history to ensure we have enough data
                    hist = t.history(period="max")

                    if hist.empty or len(hist) < 2:
                        continue

                    # Authoritative reference date for MTD/YTD so every ticker
                    # agrees on what "this month" and "this year" mean. Falls
                    # back to this ticker's own latest date if unavailable.
                    ref_day = get_last_trading_day()
                    last_trading_day = hist.index[-1]
                    if ref_day is not None:
                        anchor = pd.Timestamp(ref_day)
                        if last_trading_day.tzinfo is not None:
                            anchor = anchor.tz_localize(last_trading_day.tzinfo)
                    else:
                        anchor = last_trading_day

                    row_data = {'Name': name, 'ETF': ticker}

                    # Calculate returns for each period
                    for period_name, period_value in STANDARD_PERIODS.items():
                        try:
                            if period_value == "mtd":
                                # Month to date: from the first calendar day of the
                                # month that contains the anchor (authoritative) date
                                period_start = anchor.replace(day=1)
                                period_hist = hist[hist.index >= period_start]
                            elif period_value == "ytd":
                                # Year to date: from Jan 1 of the anchor year
                                period_start = anchor.replace(month=1, day=1)
                                period_hist = hist[hist.index >= period_start]
                            elif period_value == 1:
                                # "Today" - last 2 trading days for day-over-day change
                                period_hist = hist.tail(2)
                            else:
                                # Use number of trading days
                                # Add some buffer to ensure we get enough data
                                period_hist = hist.tail(int(period_value * 1.2))
                            
                            if len(period_hist) >= 2:
                                start_price = period_hist['Close'].iloc[0]
                                end_price = period_hist['Close'].iloc[-1]
                                return_pct = ((end_price - start_price) / start_price) * 100
                                row_data[period_name] = return_pct
                            else:
                                row_data[period_name] = None
                        except Exception as e:
                            row_data[period_name] = None
                    
                    category_data.append(row_data)
                except Exception as e:
                    continue
            
            if category_data:
                results[category] = pd.DataFrame(category_data)
        
        return results
    
    @st.cache_data(ttl=300)
    def fetch_custom_period_performance(tickers_dict, start, end):
        """Fetch performance for custom date range"""
        results = {}
        
        for category, tickers in tickers_dict.items():
            category_data = []
            
            for name, ticker in tickers.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(start=start, end=end)
                    
                    if not hist.empty and len(hist) >= 2:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        return_pct = ((end_price - start_price) / start_price) * 100
                        
                        category_data.append({
                            'Name': name,
                            'ETF': ticker,
                            'Return': return_pct
                        })
                except:
                    continue
            
            if category_data:
                results[category] = pd.DataFrame(category_data)
        
        return results
    
    def get_color_from_return(return_val):
        """Get graduated color based on return value with readable text colors"""
        if pd.isna(return_val):
            return '#404040', '#FFFFFF'
        
        # More granular color grading with READABLE text colors
        if return_val >= 50:
            return '#003300', '#FFFFFF'  # Very dark green, white text
        elif return_val >= 30:
            return '#004d00', '#FFFFFF'  # Dark green, white text
        elif return_val >= 20:
            return '#006600', '#FFFFFF'  # Medium-dark green, white text
        elif return_val >= 15:
            return '#008000', '#FFFFFF'  # Medium green, white text
        elif return_val >= 10:
            return '#009900', '#FFFFFF'  # Medium-light green, white text
        elif return_val >= 5:
            return '#00b300', '#FFFFFF'  # Light-medium green, white text
        elif return_val >= 2:
            return '#33cc33', '#000000'  # Lighter green, black text
        elif return_val >= 0:
            return '#99ff99', '#000000'  # Very light green, black text
        elif return_val >= -2:
            return '#ffcccc', '#000000'  # Very light red, black text
        elif return_val >= -5:
            return '#ff9999', '#000000'  # Light red, black text
        elif return_val >= -10:
            return '#ff4d4d', '#000000'  # Medium-light red, black text
        elif return_val >= -15:
            return '#ff0000', '#FFFFFF'  # Medium red, white text
        elif return_val >= -20:
            return '#cc0000', '#FFFFFF'  # Medium-dark red, white text
        elif return_val >= -30:
            return '#990000', '#FFFFFF'  # Dark red, white text
        else:
            return '#660000', '#FFFFFF'  # Very dark red, white text
    
    def annualize_return(total_return, years):
        """Convert total return to annualized return"""
        if years <= 0:
            return total_return
        return ((1 + total_return / 100) ** (1 / years) - 1) * 100
    
    def create_multi_period_table(df, title):
        """Create table showing all periods"""
        st.markdown(f"### {title}")
        
        # Create HTML table
        periods = ["Today", "MTD", "YTD", "1yr", "3yr", "5yr", "10yr"]
        period_years = {"Today": 0, "MTD": 0, "YTD": 0, "1yr": 1, "3yr": 3, "5yr": 5, "10yr": 10}
        # Display labels: show the real date for the "Today" column
        period_labels = {p: p for p in periods}
        period_labels["Today"] = _today_col_label
        
        html = '<table style="width:100%; border-collapse: collapse; font-size:14px;">'
        
        # Header row
        html += '<tr style="background:#1a1a1a; border-bottom: 2px solid #404040;">'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">Name</th>'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">ETF</th>'
        for period in periods:
            html += f'<th style="padding:12px; text-align:right; color:#FFFFFF; {"border-right:1px solid #404040;" if period != periods[-1] else ""}">{period_labels[period]}</th>'
        html += '</tr>'
        
        # Data rows
        for idx, row in df.iterrows():
            html += '<tr style="border-bottom:1px solid #404040;">'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#FFFFFF; border-right:1px solid #404040;">{row["Name"]}</td>'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#DDDDDD; border-right:1px solid #404040;">{row["ETF"]}</td>'
            
            for period in periods:
                return_val = row.get(period)
                
                # Annualize returns for periods > 1 year
                years = period_years[period]
                if years > 1 and not pd.isna(return_val):
                    display_return = annualize_return(return_val, years)
                else:
                    display_return = return_val
                
                bg_color, text_color = get_color_from_return(display_return)
                
                if pd.isna(return_val):
                    display_val = "N/A"
                else:
                    display_val = f"{display_return:+.1f}%"
                
                html += f'<td style="padding:10px; background:{bg_color}; color:{text_color}; text-align:right; font-weight:bold; {"border-right:1px solid #404040;" if period != periods[-1] else ""}">{display_val}</td>'
            
            html += '</tr>'
        
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        st.write("")
    
    def create_custom_period_table(df, title):
        """Create table for custom period"""
        st.markdown(f"### {title}")
        
        html = '<table style="width:100%; border-collapse: collapse; font-size:14px;">'
        
        # Header
        html += '<tr style="background:#1a1a1a; border-bottom: 2px solid #404040;">'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">Name</th>'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">ETF</th>'
        html += '<th style="padding:12px; text-align:right; color:#FFFFFF;">Return</th>'
        html += '</tr>'
        
        # Data rows
        for idx, row in df.iterrows():
            return_val = row['Return']
            bg_color, text_color = get_color_from_return(return_val)
            
            html += '<tr style="border-bottom:1px solid #404040;">'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#FFFFFF; border-right:1px solid #404040;">{row["Name"]}</td>'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#DDDDDD; border-right:1px solid #404040;">{row["ETF"]}</td>'
            html += f'<td style="padding:10px; background:{bg_color}; color:{text_color}; text-align:right; font-weight:bold;">{return_val:+.1f}%</td>'
            html += '</tr>'
        
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        st.write("")
    
    # Fetch data based on view mode
    with st.spinner("Loading market data..."):
        if view_mode == "Standard Periods":
            market_performance = fetch_multi_period_performance(MARKET_ETFS)
        else:
            market_performance = fetch_custom_period_performance(
                MARKET_ETFS, 
                datetime.combine(custom_start, datetime.min.time()),
                datetime.combine(custom_end, datetime.min.time())
            )
    
    # Display data
    if market_performance:
        # Display each category vertically
        if view_mode == "Standard Periods":
            if "Equities" in market_performance:
                create_multi_period_table(market_performance["Equities"], "📈 Equities")
            
            if "Fixed Income" in market_performance:
                create_multi_period_table(market_performance["Fixed Income"], "📊 Fixed Income")
            
            if "Real Assets" in market_performance:
                create_multi_period_table(market_performance["Real Assets"], "💰 Real Assets")
            
            if "S&P Sectors" in market_performance:
                create_multi_period_table(market_performance["S&P Sectors"], "🏭 S&P Sectors")
            
            if "Developed Markets" in market_performance:
                create_multi_period_table(market_performance["Developed Markets"], "🌍 Developed Markets")
            
            if "Emerging Markets" in market_performance:
                create_multi_period_table(market_performance["Emerging Markets"], "🌏 Emerging Markets")
            
            if "Factors" in market_performance:
                create_multi_period_table(market_performance["Factors"], "🎯 Factors")
        
        else:  # Custom period
            period_label = f"{custom_start.strftime('%m/%d/%Y')} - {custom_end.strftime('%m/%d/%Y')}"
            st.subheader(f"Custom Period: {period_label}")
            
            if "Equities" in market_performance:
                create_custom_period_table(market_performance["Equities"], "📈 Equities")
            
            if "Fixed Income" in market_performance:
                create_custom_period_table(market_performance["Fixed Income"], "📊 Fixed Income")
            
            if "Real Assets" in market_performance:
                create_custom_period_table(market_performance["Real Assets"], "💰 Real Assets")
            
            if "S&P Sectors" in market_performance:
                create_custom_period_table(market_performance["S&P Sectors"], "🏭 S&P Sectors")
            
            if "Developed Markets" in market_performance:
                create_custom_period_table(market_performance["Developed Markets"], "🌍 Developed Markets")
            
            if "Emerging Markets" in market_performance:
                create_custom_period_table(market_performance["Emerging Markets"], "🌏 Emerging Markets")
            
            if "Factors" in market_performance:
                create_custom_period_table(market_performance["Factors"], "🎯 Factors")
        
        # Export functionality
        st.divider()
        
        if st.button("📥 Export All Data to CSV"):
            export_data = []
            for category, df in market_performance.items():
                df_copy = df.copy()
                df_copy['Category'] = category
                export_data.append(df_copy)
            
            if export_data:
                combined_df = pd.concat(export_data, ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"market_dashboard_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    else:
        st.warning("No market data available")

# ========================================
# SIGNAL HISTORY PAGE
# ========================================

elif selected == "Signal History":
    st.header("📜 Signal History")
    
    if not st.session_state.signal_history:
        st.info("No signal history yet.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", ["Active", "Taken", "Skipped", "Expired"], 
                                          default=["Active", "Taken", "Skipped", "Expired"])
        with col2:
            ticker_filter = st.multiselect("Ticker", TICKERS, default=TICKERS)
        with col3:
            days_back = st.slider("Days Back", 1, 30, 7)
        
        # Filter signals
        cutoff_time = datetime.now(ZoneInfo("US/Eastern")) - timedelta(days=days_back)
        filtered_signals = [
            sig for sig in st.session_state.signal_history
            if sig.get('status') in status_filter
            and sig['symbol'] in ticker_filter
            and sig['timestamp'] >= cutoff_time
        ]
        
        st.write(f"**{len(filtered_signals)} signals from last {days_back} days**")
        
        if filtered_signals:
            history_df = pd.DataFrame([{
                'Time': sig['time'],
                'Symbol': sig['symbol'],
                'Type': sig['signal_type'],
                'Action': sig['action'],
                'Conviction': sig['conviction'],
                'Status': sig.get('status', 'Unknown'),
                'Age (min)': int((datetime.now(ZoneInfo("US/Eastern")) - sig['timestamp']).total_seconds() / 60)
            } for sig in filtered_signals])
            
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("📥 Export CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"signal_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ========================================
# BACKTEST PAGE (RESTORED)
# ========================================

elif selected == "Backtest":
    st.header("🔬 Strategy Backtest")
    st.caption("Historical performance analysis across all signal types")
    
    col1, col2 = st.columns(2)
    with col1:
        test_mode = st.radio("Backtest Mode", ["Single Ticker", "Full Portfolio"], horizontal=True)
    with col2:
        if test_mode == "Single Ticker":
            test_ticker = st.selectbox("Select Ticker", TICKERS)
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if test_mode == "Single Ticker":
            tickers_to_test = [test_ticker]
        else:
            tickers_to_test = TICKERS
            st.info(f"Running portfolio backtest across {len(TICKERS)} tickers")
        
        @st.cache_data(ttl=3600)
        def run_enhanced_backtest(ticker):
            """Run backtest with all signal types"""
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2y", interval="1d")
                
                if hist.empty or len(hist) < 200:
                    return pd.DataFrame(), f"Insufficient data for {ticker}"
                
                hist = calculate_technical_indicators(hist, periods=[10, 20, 50, 100, 200])
                
                completed_trades = []
                in_position = False
                entry_price = 0
                entry_time = None
                entry_reason = ""
                signal_type = ""
                conviction = 0
                shares = 10
                max_gain = 0
                
                # Simulate trading (using same logic as live signals)
                for i in range(200, len(hist)):
                    current_time = hist.index[i]
                    current_price = hist['Close'].iloc[i]
                    
                    if not in_position:
                        # Check all signal types (same as live)
                        # SVXY Vol Spike
                        if ticker == "SVXY" and i >= 5:
                            five_day_drop = ((current_price - hist['Close'].iloc[i-5]) / hist['Close'].iloc[i-5]) * 100
                            
                            if (five_day_drop < -8 and 
                                'Volume_Ratio' in hist.columns and hist['Volume_Ratio'].iloc[i] > 1.3 and
                                hist['Close'].pct_change().iloc[i] > -0.01):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 8
                                shares = 18
                                signal_type = "SVXY Vol Spike Recovery"
                                entry_reason = f"SVXY dropped {five_day_drop:.1f}% in 5 days"
                                max_gain = 0
                                continue
                        
                        # Golden Cross
                        if ('SMA_50' in hist.columns and 'SMA_200' in hist.columns and len(hist) >= 200):
                            if (hist['SMA_50'].iloc[i] > hist['SMA_200'].iloc[i] and
                                hist['SMA_50'].iloc[i-1] <= hist['SMA_200'].iloc[i-1] and
                                current_price > hist['SMA_50'].iloc[i]):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 9
                                shares = 20
                                signal_type = "Golden Cross"
                                entry_reason = f"Golden Cross at ${current_price:.2f}"
                                max_gain = 0
                                continue
                        
                        # SMA 10/20 Cross
                        if ('SMA_10' in hist.columns and 'SMA_20' in hist.columns):
                            if (hist['SMA_10'].iloc[i] > hist['SMA_20'].iloc[i] and
                                hist['SMA_10'].iloc[i-1] <= hist['SMA_20'].iloc[i-1] and
                                current_price > hist['SMA_10'].iloc[i] and
                                'Volume_Ratio' in hist.columns and hist['Volume_Ratio'].iloc[i] > 1.2):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 7
                                shares = 15
                                signal_type = "SMA 10/20 Cross"
                                entry_reason = f"SMA10/20 cross at ${current_price:.2f}"
                                max_gain = 0
                                continue
                        
                        # Volume Breakout
                        if ('Volume_Ratio' in hist.columns and 'SMA_20' in hist.columns):
                            if (hist['Volume_Ratio'].iloc[i] > 1.5 and
                                hist['Close'].pct_change().iloc[i] > 0.003 and
                                current_price > hist['SMA_20'].iloc[i]):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 7
                                shares = 15
                                signal_type = "Volume Breakout"
                                entry_reason = f"Volume {hist['Volume_Ratio'].iloc[i]:.1f}x"
                                max_gain = 0
                                continue
                        
                        # Oversold Bounce
                        if ('RSI' in hist.columns and 'Stoch_%K' in hist.columns):
                            if (hist['RSI'].iloc[i] < 35 and
                                hist['Stoch_%K'].iloc[i] < 25 and
                                hist['Close'].pct_change().iloc[i] > 0.001):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 6
                                shares = 12
                                signal_type = "Oversold Bounce"
                                entry_reason = f"RSI {hist['RSI'].iloc[i]:.0f} oversold"
                                max_gain = 0
                                continue
                        
                        # Mean Reversion (BB)
                        if ('BB_Lower' in hist.columns and 'RSI' in hist.columns):
                            if (current_price <= hist['BB_Lower'].iloc[i] * 1.01 and
                                hist['RSI'].iloc[i] < 40 and
                                hist['Close'].pct_change().iloc[i] > 0):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 7
                                shares = 15
                                signal_type = "Mean Reversion"
                                entry_reason = f"BB bounce at ${current_price:.2f}"
                                max_gain = 0
                                continue
                    
                    else:
                        # Check exit conditions
                        gain_pct = ((current_price - entry_price) / entry_price) * 100
                        max_gain = max(max_gain, gain_pct)
                        days_held = (current_time - entry_time).days
                        
                        exit_triggered = False
                        exit_reason = ""
                        
                        if USE_DYNAMIC_STOPS:
                            exit_triggered, exit_reason = check_exit_conditions(
                                gain_pct, max_gain, conviction, days_held
                            )
                        else:
                            if gain_pct <= -2.0:
                                exit_triggered = True
                                exit_reason = "Stop Loss (-2%)"
                            elif max_gain >= 4.0 and (max_gain - gain_pct) >= 1.0:
                                exit_triggered = True
                                exit_reason = f"Trailing Stop (from +{max_gain:.1f}%)"
                        
                        if exit_triggered:
                            pnl = (current_price - entry_price) * shares
                            pnl_pct = gain_pct
                            
                            completed_trades.append({
                                'Ticker': ticker,
                                'Entry Date': entry_time,
                                'Exit Date': current_time,
                                'Signal Type': signal_type,
                                'Conviction': conviction,
                                'Entry Price': entry_price,
                                'Exit Price': current_price,
                                'Shares': shares,
                                'P&L': pnl,
                                'P&L %': pnl_pct,
                                'Days Held': days_held,
                                'Exit Reason': exit_reason,
                                'Entry Reason': entry_reason,
                                'Capital_At_Risk': entry_price * shares
                            })
                            
                            in_position = False
                
                df = pd.DataFrame(completed_trades)
                return df, None
                
            except Exception as e:
                return pd.DataFrame(), str(e)
        
        # Run backtest
        with st.spinner("Running backtest..."):
            all_results = []
            
            for ticker in tickers_to_test:
                result_df, error = run_enhanced_backtest(ticker)
                if error:
                    st.warning(f"{ticker}: {error}")
                elif not result_df.empty:
                    all_results.append(result_df)
            
            if all_results:
                portfolio_df = pd.concat(all_results, ignore_index=True)
                
                # Overall metrics
                total_pnl = portfolio_df['P&L'].sum()
                wins = len(portfolio_df[portfolio_df['P&L'] > 0])
                losses = len(portfolio_df[portfolio_df['P&L'] <= 0])
                total_trades = len(portfolio_df)
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                avg_win = portfolio_df[portfolio_df['P&L'] > 0]['P&L'].mean() if wins > 0 else 0
                avg_loss = portfolio_df[portfolio_df['P&L'] <= 0]['P&L'].mean() if losses > 0 else 0
                
                st.success(f"✅ Backtest Complete: {total_trades} trades across {len(tickers_to_test)} tickers")
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total P&L", f"${total_pnl:,.0f}")
                col2.metric("Trades", total_trades)
                col3.metric("Win Rate", f"{win_rate:.1f}%")
                col4.metric("Avg Win", f"${avg_win:.0f}")
                col5.metric("Avg Loss", f"${avg_loss:.0f}")
                
                # P&L chart
                st.subheader("Cumulative P&L")
                portfolio_df = portfolio_df.sort_values('Exit Date')
                portfolio_df['Cumulative P&L'] = portfolio_df['P&L'].cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df['Exit Date'],
                    y=portfolio_df['Cumulative P&L'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='green' if total_pnl > 0 else 'red', width=2)
                ))
                fig.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Date",
                    yaxis_title="P&L ($)",
                    height=400,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # By signal type
                st.subheader("Performance by Signal Type")
                signal_perf = portfolio_df.groupby('Signal Type').agg({
                    'P&L': ['count', 'sum', 'mean'],
                    'P&L %': 'mean'
                }).round(2)
                signal_perf.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Avg P&L %']
                signal_perf = signal_perf.sort_values('Total P&L', ascending=False)
                st.dataframe(signal_perf, use_container_width=True)
                
                # Download
                st.download_button(
                    "📥 Download Backtest Results",
                    portfolio_df.to_csv(index=False),
                    f"backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.warning("No trades generated in backtest")

# ========================================
# TRADE LOG
# ========================================

elif selected == "Trade Log":
    st.header("📋 Trade Log")
    
    if st.session_state.trade_log.empty:
        st.info("No trades logged yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show All Trades", value=True)
        with col2:
            if st.button("📥 Export"):
                csv = st.session_state.trade_log.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        
        if show_all:
            st.dataframe(st.session_state.trade_log, use_container_width=True)
        else:
            st.dataframe(st.session_state.trade_log.tail(50), use_container_width=True)

# ========================================
# PERFORMANCE
# ========================================

elif selected == "Performance":
    st.header("📊 Performance Metrics")
    
    if st.session_state.trade_log.empty:
        st.info("No trades to analyze yet.")
    else:
        closed_trades = st.session_state.trade_log[st.session_state.trade_log['Type'] == 'Close']
        
        if not closed_trades.empty:
            total_pnl = closed_trades['P&L Numeric'].sum()
            winning_trades = len(closed_trades[closed_trades['P&L Numeric'] > 0])
            losing_trades = len(closed_trades[closed_trades['P&L Numeric'] <= 0])
            total_trades = len(closed_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = closed_trades[closed_trades['P&L Numeric'] > 0]['P&L Numeric'].mean() if winning_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['P&L Numeric'] <= 0]['P&L Numeric'].mean() if losing_trades > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total P&L", f"${total_pnl:,.0f}")
            col2.metric("Total Trades", total_trades)
            col3.metric("Win Rate", f"{win_rate:.1f}%")
            col4.metric("Avg Win", f"${avg_win:.0f}")
            col5.metric("Avg Loss", f"${avg_loss:.0f}")
            
            # P&L chart
            st.subheader("Cumulative P&L")
            closed_trades['Cumulative P&L'] = closed_trades['P&L Numeric'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed_trades.index,
                y=closed_trades['Cumulative P&L'],
                mode='lines',
                line=dict(color='green' if total_pnl > 0 else 'red', width=2)
            ))
            fig.update_layout(
                title="Cumulative P&L",
                xaxis_title="Trade Number",
                yaxis_title="P&L ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # By signal type
            st.subheader("Performance by Signal Type")
            signal_perf = closed_trades.groupby('Signal Type').agg({
                'P&L Numeric': ['sum', 'count', 'mean']
            }).round(2)
            signal_perf.columns = ['Total P&L', 'Count', 'Avg P&L']
            st.dataframe(signal_perf, use_container_width=True)

# ========================================
# CHART ANALYSIS
# ========================================

elif selected == "Chart Analysis":
    st.header("📈 Chart Analysis")
    st.caption("Type any stock or ETF symbol, or quick-pick a default / AI-portfolio holding.")

    # symbol entry row: free text + two quick-pick sources
    ec1, ec2, ec3 = st.columns([2, 1, 1])
    with ec1:
        typed = st.text_input("Search symbol", placeholder="e.g. NVDA, AAPL, VOO, SMH", key="chart_typed").strip().upper()
    with ec2:
        default_pick = st.selectbox("Defaults", ["—"] + TICKERS, key="chart_default_pick")
    with ec3:
        ai_list = ai_portfolio_tickers()
        ai_pick = st.selectbox("AI portfolio", ["—"] + ai_list, key="chart_ai_pick")

    # precedence: typed > AI pick > default pick > SPY
    chosen = typed or (ai_pick if ai_pick != "—" else "") or (default_pick if default_pick != "—" else "") or "SPY"

    period = st.radio("History", ["3mo", "6mo", "1y", "2y"], index=1, horizontal=True, key="chart_period")

    valid, info = validate_ticker(chosen)
    if not valid:
        st.error(f"Could not find data for '{chosen}'. Check the symbol (US-listed stocks and ETFs work; some ADRs/foreign listings may not).")
    else:
        nm = info.get("name", chosen)
        st.markdown(f"### {chosen} — {nm}")
        try:
            t = yf.Ticker(chosen)
            hist = t.history(period=period, interval="1d")

            if not hist.empty:
                df = calculate_technical_indicators(hist)

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=(f'{chosen} Price', 'RSI', 'Volume')
                )
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                for p, color in [(20, 'orange'), (50, 'blue'), (200, 'purple')]:
                    if f'SMA_{p}' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{p}'],
                                     name=f'SMA {p}', line=dict(color=color, width=1)), row=1, col=1)
                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                 line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color='lightblue'), row=3, col=1)
                fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # current stats
                st.subheader("Current Stats")
                s1, s2, s3, s4, s5 = st.columns(5)
                cur = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
                vr = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1.0
                adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
                s1.metric("Price", f"${cur:.2f}", f"{info.get('change_pct',0):+.2f}%")
                s2.metric("RSI", f"{rsi:.1f}")
                s3.metric("Volume Ratio", f"{vr:.2f}x")
                s4.metric("ADX", f"{adx:.1f}")
                # period return
                try:
                    pret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                    s5.metric(f"{period} Return", f"{pret:+.1f}%")
                except Exception:
                    s5.metric(f"{period} Return", "n/a")

                # if this symbol is an AI holding, surface its entry targets
                t_tgt = AI_ENTRY_TARGETS.get(chosen)
                if t_tgt:
                    zone, _ = ai_entry_zone(chosen, cur)
                    st.info(f"**AI portfolio holding — entry targets:** optimal ${t_tgt['optimal']:,.2f} · "
                            f"secondary ${t_tgt['secondary']:,.2f} · do-not-exceed ${t_tgt['ceiling']:,.2f}  →  **{zone}**")
        except Exception as e:
            st.error(f"Error loading chart: {e}")


# ========================================
# OPTIONS CHAIN
# ========================================

elif selected == "Options Chain":
    st.header("💱 Options Chain")
    st.caption("Type any optionable stock or ETF, or quick-pick a default / AI-portfolio holding.")

    oc1, oc2, oc3 = st.columns([2, 1, 1])
    with oc1:
        opt_typed = st.text_input("Search symbol", placeholder="e.g. NVDA, AAPL, SMH", key="opt_typed").strip().upper()
    with oc2:
        opt_default = st.selectbox("Defaults", ["—"] + TICKERS, key="opt_default_pick")
    with oc3:
        opt_ai = st.selectbox("AI portfolio", ["—"] + ai_portfolio_tickers(), key="opt_ai_pick")

    ticker_choice = opt_typed or (opt_ai if opt_ai != "—" else "") or (opt_default if opt_default != "—" else "") or "SPY"

    col1, col2 = st.columns(2)
    with col1:
        dte_min = st.number_input("Min DTE", value=14, min_value=1, max_value=365)
    with col2:
        dte_max = st.number_input("Max DTE", value=45, min_value=1, max_value=365)

    if st.button("Load Options Chain"):
        with st.spinner(f"Loading {ticker_choice} options..."):
            valid, _ = validate_ticker(ticker_choice)
            if not valid:
                st.error(f"Could not find '{ticker_choice}'. Check the symbol.")
                st.session_state.opt_chain_df = None
            else:
                options_df = get_options_chain(ticker_choice, dte_min, dte_max)
                if not options_df.empty:
                    st.session_state.opt_chain_df = options_df
                    st.session_state.opt_chain_ticker = ticker_choice
                else:
                    st.session_state.opt_chain_df = None
                    st.warning(f"No options data for {ticker_choice} in this DTE window. "
                               "The symbol may not be optionable, or try widening the DTE range.")

    # Render from session state so the Calls/Puts/Both radio works without a rescan
    chain_df = st.session_state.get('opt_chain_df')
    if chain_df is not None and not chain_df.empty:
        st.success(f"Loaded {len(chain_df)} contracts for {st.session_state.get('opt_chain_ticker', ticker_choice)}")

        option_type = st.radio("Type", ["Calls", "Puts", "Both"], horizontal=True, key="opt_type_radio")

        if option_type == "Calls":
            filtered = chain_df[chain_df['type'] == 'Call']
        elif option_type == "Puts":
            filtered = chain_df[chain_df['type'] == 'Put']
        else:
            filtered = chain_df

        display_cols = ['strike', 'type', 'dte', 'expiration', 'bid', 'ask', 'mid',
                        'lastPrice', 'impliedVolatility', 'volume', 'openInterest']
        available_cols = [col for col in display_cols if col in filtered.columns]

        if filtered.empty:
            st.info(f"No {option_type.lower()} found in this DTE window.")
        else:
            st.dataframe(filtered[available_cols].sort_values(['dte', 'strike']),
                         use_container_width=True, hide_index=True)
            st.caption("When selling, you collect the **bid**, not the mid. Strikes showing bid = 0 cannot be sold at any positive price.")


# ========================================
# PREMIUM SELLER
# ========================================

elif selected == "Premium Seller":
    st.header("🛡️ Premium Seller")
    st.caption("Find option strikes where implied vol is rich vs realized, with high probability of expiring worthless. Ranks by edge and expected value computed from live mid prices.")

    st.info(
        "**What this targets:** not high probability alone (any far-OTM strike clears that). "
        "It scores where you are paid the most relative to the real risk, then filters by your probability floor. "
        "Probabilities use a fat-tail (Student-t) model, so they sit below naive Black-Scholes by design. "
        "Decision support, not trade signals. Index options are generally permitted under your compliance policy; verify before trading."
    )

    # ---- inputs ----
    c1, c2, c3 = st.columns(3)
    with c1:
        _ps_typed = st.text_input("Underlying (type any optionable symbol)",
                                  placeholder="e.g. SPY, NVDA, SMH", key="ps_typed").strip().upper()
        _ps_default = st.selectbox("…or pick", ["—"] + TICKERS + ai_portfolio_tickers(), key="ps_symbol_pick")
        ps_symbol = _ps_typed or (_ps_default if _ps_default != "—" else "") or "SPY"
        ps_structures = st.multiselect(
            "Structures",
            ["Cash-Secured Put", "Put Credit Spread", "Iron Condor", "Short Strangle"],
            default=["Cash-Secured Put", "Put Credit Spread", "Iron Condor"],
            help="CSP and spreads are put-side income. Condors and strangles collect on both sides. Spreads and condors cap your loss; CSP and strangle do not."
        )
    with c2:
        ps_dte_min = st.number_input("Min DTE", value=1, min_value=0, max_value=120,
            help="Days to expiration. 0-2 = daily/0DTE, 3-9 = weekly.")
        ps_dte_max = st.number_input("Max DTE", value=9, min_value=1, max_value=120,
            help="Upper bound of the expiration window to scan.")
        ps_width = st.number_input("Spread width ($)", value=2.0, min_value=1.0, step=1.0,
            help="Distance between short and long strikes for spreads and condors. Narrow widths ($1-3) reach positive EV far more easily: a $1 spread needs ~75% win rate to break even, a $5 spread needs ~94%. Wider = more credit per trade but worse expected value.")
    with c3:
        ps_min_pop = st.slider("Min probability of expiring worthless (%)", 70, 99, 85,
            help="POP: your win-rate floor. Higher POP sits further OTM with thinner credit. 85% is the sweet spot where positive EV is still reachable. A 90%+ floor often returns an empty table because the math does not support it.")
        ps_event = st.selectbox("Event in window",
            ["none", "macro", "earn", "both"],
            format_func=lambda x: {"none": "None known", "macro": "Macro print (CPI/NFP/FOMC)", "earn": "Earnings (single name)", "both": "Multiple events"}[x],
            help="Flags known volatility events inside the window. Events fatten the modeled tails and lower probabilities, because realized moves are larger around them.")
        ps_ev_filter = st.checkbox("Only show positive expected value", value=True,
            help="Expected value = (POP × credit) − ((1−POP) × expected loss). Positive EV means the trade pays you over many repetitions. On index options this is often empty at high POP; that empty result is honest, not a bug.")

    ps_rfr = st.number_input("Risk-free rate (%)", value=4.3, step=0.1,
        help="Annual short-term rate, roughly the T-bill yield. Used in Black-Scholes pricing.") / 100

    with st.expander("What the columns mean"):
        st.markdown("""
- **POP (fat-tail)**: probability the short option expires worthless under a Student-t distribution with fat tails. This is the honest win probability.
- **BS POP**: Black-Scholes probability under normal tails. Usually higher than fat-tail POP. The gap is your *steamroller discount*: how much naive models overstate safety.
- **Credit**: premium you actually collect, priced at the live **bid** (what a buyer will pay you now). Spreads net the short bid minus the long ask. Strikes with no bid are rejected as untradeable. This avoids phantom mid-price trades.
- **Max Loss**: most you can lose. Defined for spreads/condors (width − credit). Undefined (very large) for naked CSP and strangles.
- **ROC**: return on capital = credit ÷ capital at risk. How hard your money works on the trade.
- **EV (Expected Value)**: (POP × credit) − ((1−POP) × expected loss). The single most important number. Negative EV means the trade loses over time even if it usually wins.
- **IV−RV**: implied vol minus realized vol, in vol points. The *vol risk premium*. Positive and wide = the market is paying you more than recent movement justifies. This is the core edge.
- **Kelly**: the bet size fraction that maximizes long-run growth, from win probability and payoff ratio. Treat full Kelly as a ceiling; most size at one-quarter to one-half of it.
- **Edge**: 0-100 composite score, 50% IV Rank + 50% IV−RV spread, penalized for negative EV and active events. Higher is better.
        """)

    run_ps = st.button("▶ Run Premium Seller Scan", type="primary", use_container_width=True)

    if run_ps:
        with st.spinner("Pulling live chain and running the model..."):
            chain = get_options_chain(ps_symbol, ps_dte_min, ps_dte_max)

            if chain.empty:
                st.warning("No options chain returned for that symbol and DTE window. Try widening the DTE range, or the market data feed may be delayed.")
            else:
                # spot price
                try:
                    spot = float(yf.Ticker(ps_symbol).history(period="1d")['Close'].iloc[-1])
                except Exception:
                    spot = float(chain['strike'].median())

                rv = ps_realized_vol(ps_symbol) or 0.0
                # ATM IV: closest strike to spot
                atm_row = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
                atm_iv = float(atm_row['impliedVolatility'].iloc[0]) if 'impliedVolatility' in chain.columns and not atm_row.empty else 0.0
                iv_rank = ps_iv_rank(ps_symbol, atm_iv)
                iv_rv_spread = (atm_iv - rv) * 100  # vol points
                df_tail = ps_event_df(ps_event)
                has_event = ps_event != "none"

                # environment classification (rendered persistently below, not here)
                env = "SELLER" if iv_rv_spread >= 2.5 else ("NEUTRAL" if iv_rv_spread >= -1 else "BUYER")

                # spread/IV component for edge
                spread_comp = max(0, min(100, ((iv_rv_spread + 2) / 8) * 100))
                ivr_comp = iv_rank if iv_rank is not None else 50.0

                puts = chain[chain['type'] == 'Put'].copy()
                calls = chain[chain['type'] == 'Call'].copy()

                def live_iv(row, K, T):
                    """Use the chain's own IV when present, else modeled skew."""
                    v = row.get('impliedVolatility', None)
                    if v is not None and not pd.isna(v) and v > 0:
                        return float(v)
                    return ps_skew_iv(K, spot, atm_iv if atm_iv > 0 else 0.15, T)

                def _row_for(df_side, strike):
                    r = df_side[df_side['strike'] == strike]
                    return None if r.empty else r.iloc[0]

                def sell_price(df_side, strike):
                    """Price you actually COLLECT when selling this strike = the bid.
                    Returns None if there is no real bid (untradeable) or the
                    bid/ask is implausible. This is the fix for phantom mid prices."""
                    r = _row_for(df_side, strike)
                    if r is None:
                        return None
                    bid = float(r['bid']) if not pd.isna(r['bid']) else 0.0
                    ask = float(r['ask']) if not pd.isna(r['ask']) else 0.0
                    # no bid => nobody will buy it from you => not tradeable
                    if bid <= 0:
                        return None
                    # reject absurd spreads (bid/ask wider than the ask itself is illiquid garbage)
                    if ask > 0 and (ask - bid) / ask > 0.80 and ask > 0.10:
                        return None
                    return bid

                def buy_price(df_side, strike):
                    """Price you PAY to buy the long (protective) leg = the ask.
                    For the long leg, missing ask means we cannot price the spread."""
                    r = _row_for(df_side, strike)
                    if r is None:
                        return None
                    ask = float(r['ask']) if not pd.isna(r['ask']) else 0.0
                    if ask <= 0:
                        # if there is no ask, treat the long leg as free (best case for us)
                        # but only if a bid exists to confirm the strike trades at all
                        bid = float(r['bid']) if not pd.isna(r['bid']) else 0.0
                        return 0.0 if bid >= 0 else None
                    return ask

                rows = []

                def score_and_append(rec):
                    if rec['pop'] * 100 < ps_min_pop:
                        return
                    if rec['credit'] < 0.02:
                        return
                    edge = 0.5 * ivr_comp + 0.5 * spread_comp
                    if rec['ev'] < 0:
                        edge -= 18
                    if has_event:
                        edge -= 10
                    if rec['defined']:
                        edge += 4
                    edge = max(0, min(100, edge))
                    rec['edge'] = round(edge)
                    rec['grade'] = 'A' if edge >= 72 else ('B' if edge >= 58 else ('C' if edge >= 45 else 'D'))
                    rows.append(rec)

                # iterate unique expirations to keep DTE consistent per structure
                for exp in sorted(puts['expiration'].unique()):
                    p_exp = puts[puts['expiration'] == exp].sort_values('strike')
                    c_exp = calls[calls['expiration'] == exp].sort_values('strike')
                    if p_exp.empty:
                        continue
                    dte = int(p_exp['dte'].iloc[0])
                    T = max(dte, 0.5) / 365

                    # ---- Cash-Secured Put ----
                    if "Cash-Secured Put" in ps_structures:
                        for _, pr in p_exp[p_exp['strike'] < spot].iterrows():
                            K = float(pr['strike'])
                            iv = live_iv(pr, K, T)
                            pop = ps_pop_fattail(spot, K, T, ps_rfr, iv, True, df_tail)
                            if pop * 100 < ps_min_pop:
                                continue
                            credit = sell_price(p_exp, K)
                            if credit is None:
                                continue
                            bs = ps_pop_bs(spot, K, T, ps_rfr, iv, True)
                            capital = K - credit
                            roc = credit / capital * 100 if capital > 0 else 0
                            el = min(K * 0.04 + credit, K * 0.5)
                            ev = pop * credit - (1 - pop) * el
                            ratio = credit / el if el > 0 else 0.01
                            kelly = max(0, pop - (1 - pop) / max(ratio, 0.01))
                            score_and_append(dict(struct="CSP", legs=f"{K:.0f}P", exp=exp, dte=dte,
                                pop=pop, bs=bs, credit=credit, maxloss=capital, roc=roc, ev=ev,
                                ivrv=iv_rv_spread, kelly=kelly, defined=False, undef=True,
                                Kshort=K, Kcall=None, strike_iv=iv, el_breach=el, breakeven=K-credit))

                    # ---- Put Credit Spread ----
                    if "Put Credit Spread" in ps_structures:
                        for _, pr in p_exp[p_exp['strike'] < spot].iterrows():
                            K = float(pr['strike'])
                            Klong = K - ps_width
                            iv = live_iv(pr, K, T)
                            pop = ps_pop_fattail(spot, K, T, ps_rfr, iv, True, df_tail)
                            if pop * 100 < ps_min_pop:
                                continue
                            short_mid = sell_price(p_exp, K)
                            long_mid = buy_price(p_exp, Klong)
                            if short_mid is None or long_mid is None:
                                continue
                            credit = short_mid - long_mid
                            if credit < 0.02:
                                continue
                            bs = ps_pop_bs(spot, K, T, ps_rfr, iv, True)
                            maxloss = max(ps_width - credit, 0.01)
                            roc = credit / maxloss * 100
                            ev = pop * credit - (1 - pop) * maxloss
                            kelly = max(0, pop - (1 - pop) / (credit / maxloss))
                            score_and_append(dict(struct="Put Spread", legs=f"{K:.0f}/{Klong:.0f}P", exp=exp, dte=dte,
                                pop=pop, bs=bs, credit=credit, maxloss=maxloss, roc=roc, ev=ev,
                                ivrv=iv_rv_spread, kelly=kelly, defined=True, undef=False,
                                Kshort=K, Kcall=None, strike_iv=iv, el_breach=maxloss, breakeven=K-credit))

                    # ---- Iron Condor ----
                    if "Iron Condor" in ps_structures and not c_exp.empty:
                        for _, pr in p_exp[p_exp['strike'] < spot].iterrows():
                            Kp = float(pr['strike'])
                            dist = spot - Kp
                            Kc = spot + dist
                            # nearest available call strike
                            c_near = c_exp.iloc[(c_exp['strike'] - Kc).abs().argsort()[:1]]
                            if c_near.empty:
                                continue
                            Kc = float(c_near['strike'].iloc[0])
                            ivp = live_iv(pr, Kp, T)
                            ivc = live_iv(c_near.iloc[0], Kc, T)
                            popP = ps_pop_fattail(spot, Kp, T, ps_rfr, ivp, True, df_tail)
                            popC = ps_pop_fattail(spot, Kc, T, ps_rfr, ivc, False, df_tail)
                            pop = max(0, popP + popC - 1)
                            if pop * 100 < ps_min_pop:
                                continue
                            KpL, KcL = Kp - ps_width, Kc + ps_width
                            ps_short = sell_price(p_exp, Kp); ps_long = buy_price(p_exp, KpL)
                            cs_short = sell_price(c_exp, Kc); cs_long = buy_price(c_exp, KcL)
                            if None in (ps_short, ps_long, cs_short, cs_long):
                                continue
                            credit = (ps_short - ps_long) + (cs_short - cs_long)
                            if credit < 0.02:
                                continue
                            bsP = ps_pop_bs(spot, Kp, T, ps_rfr, ivp, True)
                            bsC = ps_pop_bs(spot, Kc, T, ps_rfr, ivc, False)
                            bs = max(0, bsP + bsC - 1)
                            maxloss = max(ps_width - credit, 0.01)
                            roc = credit / maxloss * 100
                            ev = pop * credit - (1 - pop) * maxloss
                            kelly = max(0, pop - (1 - pop) / (credit / maxloss))
                            score_and_append(dict(struct="Iron Condor", legs=f"{Kp:.0f}/{KpL:.0f}P · {Kc:.0f}/{KcL:.0f}C", exp=exp, dte=dte,
                                pop=pop, bs=bs, credit=credit, maxloss=maxloss, roc=roc, ev=ev,
                                ivrv=iv_rv_spread, kelly=kelly, defined=True, undef=False,
                                Kshort=Kp, Kcall=Kc, strike_iv=ivp, el_breach=maxloss, breakeven=None))

                    # ---- Short Strangle ----
                    if "Short Strangle" in ps_structures and not c_exp.empty:
                        sig1 = (atm_iv if atm_iv > 0 else 0.15) * _math.sqrt(T) * spot
                        for _, pr in p_exp[p_exp['strike'] < spot].iterrows():
                            Kp = float(pr['strike'])
                            dist = spot - Kp
                            Kc = spot + dist
                            c_near = c_exp.iloc[(c_exp['strike'] - Kc).abs().argsort()[:1]]
                            if c_near.empty:
                                continue
                            Kc = float(c_near['strike'].iloc[0])
                            ivp = live_iv(pr, Kp, T)
                            ivc = live_iv(c_near.iloc[0], Kc, T)
                            popP = ps_pop_fattail(spot, Kp, T, ps_rfr, ivp, True, df_tail)
                            popC = ps_pop_fattail(spot, Kc, T, ps_rfr, ivc, False, df_tail)
                            pop = max(0, popP + popC - 1)
                            if pop * 100 < ps_min_pop:
                                continue
                            p_mid = sell_price(p_exp, Kp); c_mid = sell_price(c_exp, Kc)
                            if p_mid is None or c_mid is None:
                                continue
                            credit = p_mid + c_mid
                            if credit < 0.02:
                                continue
                            bsP = ps_pop_bs(spot, Kp, T, ps_rfr, ivp, True)
                            bsC = ps_pop_bs(spot, Kc, T, ps_rfr, ivc, False)
                            bs = max(0, bsP + bsC - 1)
                            stress = 4 * sig1
                            maxloss = max(stress - credit, credit * 8)
                            capital = Kp * 0.20
                            roc = credit / capital * 100 if capital > 0 else 0
                            el = min(2 * sig1, maxloss * 0.4)
                            ev = pop * credit - (1 - pop) * el
                            ratio = credit / el if el > 0 else 0.01
                            kelly = max(0, pop - (1 - pop) / max(ratio, 0.01))
                            score_and_append(dict(struct="Strangle", legs=f"{Kp:.0f}P / {Kc:.0f}C", exp=exp, dte=dte,
                                pop=pop, bs=bs, credit=credit, maxloss=maxloss, roc=roc, ev=ev,
                                ivrv=iv_rv_spread, kelly=kelly, defined=False, undef=True,
                                Kshort=Kp, Kcall=Kc, strike_iv=ivp, el_breach=el, breakeven=None))

                # ---- assemble, rank, store in session for filtering/inspection ----
                rows.sort(key=lambda r: r['edge'], reverse=True)
                st.session_state.ps_results = {
                    'rows': rows,
                    'spot': spot,
                    'width': ps_width,
                    'symbol': ps_symbol,
                    'ev_filter': ps_ev_filter,
                    'min_pop': ps_min_pop,
                    'env': env,
                    'iv_rv_spread': iv_rv_spread,
                    'atm_iv': atm_iv,
                    'rv': rv,
                    'iv_rank': iv_rank,
                }

    # ---- render from session state (runs every rerun so filters/detail persist) ----
    res = st.session_state.get('ps_results')
    if res:
        all_rows = res['rows']
        spot = res['spot']
        width = res['width']
        symbol = res['symbol']

        # environment metrics strip (persists across reruns)
        em1, em2, em3, em4, em5 = st.columns(5)
        em1.metric("Environment", res.get('env', '—'), help="SELLER when IV is rich vs realized; BUYER when vol is underpriced.")
        em2.metric("IV − RV", f"{res.get('iv_rv_spread',0):+.1f}v", help="Vol risk premium in vol points. Positive favors selling.")
        em3.metric("ATM IV", f"{res.get('atm_iv',0)*100:.1f}%", help="At-the-money implied volatility from the live chain.")
        em4.metric("Realized Vol", f"{res.get('rv',0)*100:.1f}%", help="20-day annualized historical volatility.")
        _ivr = res.get('iv_rank')
        em5.metric("IV Rank", f"{_ivr:.0f}" if _ivr is not None else "n/a", help="Where current vol sits in its 1-year range (proxy on free data).")

        # positive-EV filter applied here so toggling does not require a rescan
        rows = [r for r in all_rows if r['ev'] >= 0] if res['ev_filter'] else list(all_rows)

        if not rows:
            if res['ev_filter']:
                st.warning(
                    f"**No positive-EV candidates at {res['min_pop']}% POP.** This is the honest result, not an error. "
                    "The vol risk premium on this underlying is too thin right now to pay for the tail risk at that probability. "
                    "Try: lower the POP floor toward 80%, uncheck the positive-EV filter, or wait for an elevated-vol environment (Environment = SELLER)."
                )
            else:
                st.warning("No candidates passed the probability and credit filters. Lower the Min POP, or the chain may be too thin at this DTE.")
        else:
            # ----- Edge / Grade filters -----
            st.divider()
            fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
            with fcol1:
                grades_present = sorted({r['grade'] for r in rows})
                grade_pick = st.multiselect(
                    "Filter by grade", ["A", "B", "C", "D"],
                    default=grades_present,
                    help="A = rich premium and safe. B = solid. C = thin edge. D = avoid. Deselect the weak grades to hide trades that qualify but are not worth taking."
                )
            with fcol2:
                max_edge = max(r['edge'] for r in rows)
                min_edge_filter = st.slider(
                    "Minimum edge", 0, 100, 0,
                    help="Hide candidates below this composite edge score (0-100). Raise it to keep only the strongest setups."
                )
            with fcol3:
                sort_choice = st.selectbox(
                    "Sort by",
                    ["Edge (high→low)", "Expected value (high→low)", "POP (high→low)", "ROC (high→low)", "Credit (high→low)"],
                    help="Reorder the table. Edge is the overall quality score; the others isolate a single dimension."
                )

            filtered = [r for r in rows if r['grade'] in grade_pick and r['edge'] >= min_edge_filter]

            sort_keys = {
                "Edge (high→low)": 'edge',
                "Expected value (high→low)": 'ev',
                "POP (high→low)": 'pop',
                "ROC (high→low)": 'roc',
                "Credit (high→low)": 'credit',
            }
            filtered.sort(key=lambda r: r[sort_keys[sort_choice]], reverse=True)

            if not filtered:
                st.info("No candidates match the current grade/edge filters. Loosen them above.")
            else:
                # ----- table -----
                out = pd.DataFrame(filtered)
                out['POP %'] = (out['pop'] * 100).round(1)
                out['BS POP %'] = (out['bs'] * 100).round(1)
                out['Credit $'] = out['credit'].round(2)
                out['Max Loss $'] = out['maxloss'].round(2)
                out['ROC %'] = out['roc'].round(1)
                out['EV $'] = out['ev'].round(3)
                out['IV−RV'] = out['ivrv'].round(1)
                out['Kelly %'] = (out['kelly'] * 100).round(1)
                disp = out[['struct', 'legs', 'exp', 'dte', 'POP %', 'BS POP %',
                            'Credit $', 'Max Loss $', 'ROC %', 'EV $', 'IV−RV', 'Kelly %', 'edge', 'grade']]
                disp = disp.rename(columns={'struct': 'Structure', 'legs': 'Strikes',
                                            'exp': 'Expiration', 'dte': 'DTE', 'edge': 'Edge', 'grade': 'Grade'})
                st.success(f"Showing {len(filtered)} of {len(rows)} candidates after filters.")
                st.dataframe(disp, use_container_width=True, hide_index=True, height=420)

                if any(r['undef'] for r in filtered):
                    st.warning("⚠️ Rows marked **CSP** or **Strangle** carry undefined risk. A gap move past your strike has no floor. Size tiny or convert to a defined-risk spread.")

                csv = disp.to_csv(index=False)
                st.download_button("Download candidates (CSV)", csv, f"premium_seller_{symbol}.csv", "text/csv")

                # ----- candidate detail panel (top auto-selected, selectbox to switch) -----
                st.divider()
                st.subheader("🔎 Candidate detail")

                labels = [f"{r['struct']} · {r['legs']} · {r['exp']} · EV ${r['ev']:.2f} · {r['grade']}" for r in filtered]
                pick = st.selectbox(
                    "Inspect a candidate (defaults to the top-ranked)",
                    range(len(filtered)),
                    format_func=lambda i: labels[i],
                )
                rec = filtered[pick]

                dcol1, dcol2 = st.columns([1, 1])
                with dcol1:
                    st.markdown("**Probability**")
                    steamroller = (rec['bs'] - rec['pop']) * 100
                    st.markdown(
                        f"- Fat-tail POP: **{rec['pop']*100:.2f}%**\n"
                        f"- Black-Scholes POP: **{rec['bs']*100:.2f}%**\n"
                        f"- Steamroller discount: **{steamroller:.2f}%**\n"
                        f"- Strike IV: **{rec.get('strike_iv',0)*100:.1f}%**\n"
                        f"- Prob touch (≈2×ITM): **{min(100,(1-rec['pop'])*200):.1f}%**"
                    )
                    st.markdown("**Risk / Reward**")
                    be = rec.get('breakeven')
                    be_str = f"${be:.2f}" if be is not None else "—"
                    ml_str = f"undefined (~${rec['maxloss']*100:,.0f})" if rec['undef'] else f"${rec['maxloss']*100:,.2f}"
                    st.markdown(
                        f"- Credit (per contract): **${rec['credit']*100:,.2f}**\n"
                        f"- Max loss: **{ml_str}**\n"
                        f"- Return on capital: **{rec['roc']:.1f}%**\n"
                        f"- Expected value: **${rec['ev']*100:,.2f}**\n"
                        f"- Breakeven: **{be_str}**\n"
                        f"- Kelly fraction: **{rec['kelly']*100:.1f}%** (½K: {rec['kelly']*50:.1f}%)\n"
                        f"- Edge score: **{rec['edge']}/100 · grade {rec['grade']}**"
                    )
                with dcol2:
                    st.markdown("**Payoff at expiration**")
                    st.plotly_chart(ps_payoff_figure(rec, spot, width), use_container_width=True)
                    if rec['undef']:
                        st.warning("Undefined risk. A gap move past your strike has no floor. Size tiny or convert to a spread.")


# ========================================
# AI STRATEGY
# ========================================

elif selected == "AI Strategy":
    st.header("🤖 AI Infrastructure Equity Strategy")
    st.caption("Live tracking for the AI infrastructure portfolio. Inception 2026-02-10. Benchmark: Nasdaq-100 (NDX).")

    ai_tabs = st.tabs(["📊 Portfolio", "📈 Performance", "✏️ Manage", "👁️ Bench", "📜 Mandate", "🎯 Conviction", "📄 Fact Sheet"])

    # ---------------- PORTFOLIO ----------------
    with ai_tabs[0]:
        holdings = ai_load_portfolio()
        st.subheader("Model Portfolio: Live Holdings")
        with st.spinner("Fetching live prices..."):
            strat_ret, bench_ret, per = ai_compute_performance(holdings, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK)

        # headline metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Holdings", f"{len(holdings)}")
        tw = sum(h.get('target_weight') or 0 for h in holdings)
        mc2.metric("Invested Weight", f"{tw:.0f}%", help="Sum of target weights. Remainder is cash.")
        mc3.metric("Strategy (since inception)", f"{strat_ret:+.1f}%" if strat_ret is not None else "n/a",
                   help="Target-weighted price change of holdings since 2026-02-10.")
        delta_v = (strat_ret - bench_ret) if (strat_ret is not None and bench_ret is not None) else None
        mc4.metric("NDX (benchmark)", f"{bench_ret:+.1f}%" if bench_ret is not None else "n/a",
                   f"{delta_v:+.1f}% active" if delta_v is not None else None)

        # tier allocation summary
        st.markdown("##### Tier Allocation vs. Target")
        tier_now = {}
        for h in holdings:
            tier_now[h['tier']] = tier_now.get(h['tier'], 0) + (h.get('target_weight') or 0)
        tcols = st.columns(len(AI_TIER_TARGETS))
        for i, (tier, target) in enumerate(AI_TIER_TARGETS.items()):
            cur = tier_now.get(tier, 0)
            if tier == "CASH":
                cur = max(0, 100 - tw)
            tcols[i].metric(tier.title(), f"{cur:.0f}%", f"target {target}%", delta_color="off")

        # ----- entry-target view toggle -----
        st.markdown("##### Holdings")
        show_entries = st.checkbox("Show entry-target columns (optimal / secondary / do-not-exceed)", value=True,
                                   help="Limit-order anchors from the buildout plan. Zone shows where the live price sits now.")

        df = pd.DataFrame(per)
        if not df.empty:
            df['Price'] = df['price_now'].apply(lambda x: f"${x:,.2f}" if x is not None else "n/a")
            df['Since Incept'] = df['ret_pct'].apply(lambda x: f"{x:+.1f}%" if x is not None else "n/a")
            df['Contribution'] = df['contribution'].apply(lambda x: f"{x:+.2f}%" if x is not None else "n/a")
            df['Weight'] = df['target_weight'].apply(lambda x: f"{x:.0f}%" if x is not None else "")
            df['Score'] = df['score'].apply(lambda x: f"{x:.1f}" if x is not None else "")

            # entry-target columns
            def _fmt_price(v):
                return f"${v:,.2f}" if v is not None else "—"
            zones, opt, sec, ceil = [], [], [], []
            for _, r in df.iterrows():
                z, t = ai_entry_zone(r['ticker'], r['price_now'])
                zones.append(z)
                opt.append(_fmt_price(t["optimal"]) if t else "—")
                sec.append(_fmt_price(t["secondary"]) if t else "—")
                ceil.append(_fmt_price(t["ceiling"]) if t else "—")
            df['Optimal'] = opt
            df['Secondary'] = sec
            df['Do-Not-Exceed'] = ceil
            df['Zone'] = zones

            tier_order = ["HYPER", "TIER 1", "TIER 2", "TIER 3"]
            df['_ord'] = df['tier'].apply(lambda t: tier_order.index(t) if t in tier_order else 99)
            df = df.sort_values(['_ord', 'target_weight'], ascending=[True, False])

            if show_entries:
                cols = ['tier', 'ticker', 'name', 'Score', 'Weight', 'Price', 'Zone',
                        'Optimal', 'Secondary', 'Do-Not-Exceed', 'Since Incept', 'Contribution']
            else:
                cols = ['tier', 'ticker', 'name', 'Score', 'Weight', 'Price', 'Since Incept', 'Contribution']
            show = df[cols].rename(columns={'tier': 'Tier', 'ticker': 'Ticker', 'name': 'Name'})
            st.dataframe(show, use_container_width=True, hide_index=True, height=560)
            if show_entries:
                st.caption("Zone: 🟢 at/below optimal or in buy zone · 🟡 below the do-not-exceed ceiling · 🔴 above ceiling · "
                           "🔄 RE-RATE/Stale = a recent earnings or guidance event (or a gap >12% past the ceiling) has made the static levels obsolete — re-derive off new guidance before acting. "
                           "Ceilings are anchored to a forward-P/E multiple (shown in the thesis view), so they travel with estimates rather than staying fixed. "
                           "Optimal = the pullback worth waiting for; Secondary = scale-in; Do-Not-Exceed = ceiling. Limit-order guides, not forecasts.")
                _reval = [h['ticker'] for h in holdings if (AI_ENTRY_TARGETS.get(h['ticker']) or {}).get('reval')]
                if _reval:
                    st.warning("🔄 Targets need refreshing after a recent re-rating event: " + ", ".join(_reval)
                               + ". These names reported or gapped on news; re-derive their optimal/secondary/ceiling off the updated guidance.")
            else:
                st.caption("Since Incept = each name's price change since 2026-02-10. Contribution = target weight × that return.")

            # per-holding thesis expander
            with st.expander("📝 View conviction thesis per holding"):
                pick = st.selectbox("Holding", [h['ticker'] for h in holdings], key="ai_thesis_pick")
                rec = next((h for h in holdings if h['ticker'] == pick), None)
                if rec:
                    st.markdown(f"**{rec['ticker']} — {rec['name']}** · {rec['tier']} · Score {rec.get('score','?')} · Target {rec.get('target_weight','?')}%")
                    t = AI_ENTRY_TARGETS.get(rec['ticker'])
                    if t:
                        st.markdown(f"Entry targets — optimal **${t['optimal']:,.2f}** · secondary **${t['secondary']:,.2f}** · do-not-exceed **${t['ceiling']:,.2f}**")
                    st.write(rec.get('thesis', 'No thesis recorded.'))
        else:
            st.warning("No holdings to display.")

    # ---------------- PERFORMANCE ----------------
    with ai_tabs[1]:
        holdings = ai_load_portfolio()
        st.subheader("Performance Since Inception (2026-02-10)")
        with st.spinner("Computing performance..."):
            strat_ret, bench_ret, per = ai_compute_performance(holdings, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK)

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Strategy (Model)", f"{strat_ret:+.2f}%" if strat_ret is not None else "n/a")
        pc2.metric("Nasdaq-100", f"{bench_ret:+.2f}%" if bench_ret is not None else "n/a")
        active = (strat_ret - bench_ret) if (strat_ret is not None and bench_ret is not None) else None
        pc3.metric("Active Return", f"{active:+.2f}%" if active is not None else "n/a",
                   help="Strategy minus benchmark. Positive = outperforming NDX.")

        st.info("This is **model** performance: target weights × each holding's actual price change since inception. "
                "It reflects how the strategy's allocation has performed, not your exact brokerage P&L (which would need your real share counts and fills).")

        # contribution chart
        dfp = pd.DataFrame([p for p in per if p.get('contribution') is not None])
        if not dfp.empty:
            dfp = dfp.sort_values('contribution', ascending=False)
            st.markdown("##### Contribution to Return by Holding")
            fig = go.Figure()
            colors = ['#27763d' if c >= 0 else '#9b2226' for c in dfp['contribution']]
            fig.add_trace(go.Bar(x=dfp['ticker'], y=dfp['contribution'], marker_color=colors))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#c9d4e0'),
                              xaxis=dict(title="", gridcolor='rgba(120,130,140,0.12)'),
                              yaxis=dict(title="Contribution (%)", gridcolor='rgba(120,130,140,0.12)'))
            st.plotly_chart(fig, use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Top 5 Contributors**")
                top = dfp.head(5)[['ticker', 'ret_pct', 'contribution']].copy()
                top['Return'] = top['ret_pct'].apply(lambda x: f"{x:+.1f}%")
                top['Contribution'] = top['contribution'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(top[['ticker', 'Return', 'Contribution']].rename(columns={'ticker': 'Ticker'}),
                             use_container_width=True, hide_index=True)
            with cc2:
                st.markdown("**Bottom 5 Contributors**")
                bot = dfp.tail(5)[['ticker', 'ret_pct', 'contribution']].copy()
                bot['Return'] = bot['ret_pct'].apply(lambda x: f"{x:+.1f}%")
                bot['Contribution'] = bot['contribution'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(bot[['ticker', 'Return', 'Contribution']].rename(columns={'ticker': 'Ticker'}),
                             use_container_width=True, hide_index=True)
        else:
            st.warning("Could not compute contributions. Price data may be unavailable.")


        # ===== Performance over time + Max Drawdown =====
        st.divider()
        st.markdown("##### Performance Over Time (since inception)")
        hk = tuple((h['ticker'], h.get('target_weight') or 0) for h in holdings)
        with st.spinner("Building equity curve..."):
            ec = ai_equity_curve(hk, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK, "VOO")

        if not ec or not ec.get("dates"):
            st.warning("Could not build the equity curve (price history unavailable right now). Try refreshing.")
        else:
            dts = pd.to_datetime(ec["dates"])
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                row_heights=[0.68, 0.32],
                subplot_titles=("Cumulative Return: Strategy vs NDX vs VOO", "Strategy Drawdown")
            )
            fig.add_trace(go.Scatter(x=dts, y=ec["port"], name="Strategy",
                          line=dict(color="#2dd4bf", width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=dts, y=ec["ndx"], name="NDX",
                          line=dict(color="#fbbf24", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=dts, y=ec["voo"], name="VOO",
                          line=dict(color="#94a3b8", width=1.5, dash="dot")), row=1, col=1)
            fig.add_hline(y=0, line=dict(color="#6b7785", width=1, dash="dash"), row=1, col=1)
            # drawdown area
            fig.add_trace(go.Scatter(x=dts, y=ec["drawdown"], name="Drawdown",
                          line=dict(color="#9b2226", width=1.5), fill="tozeroy",
                          fillcolor="rgba(155,34,38,0.18)", showlegend=False), row=2, col=1)
            # mark the max drawdown point
            if ec.get("max_dd_date"):
                fig.add_vline(x=pd.Timestamp(ec["max_dd_date"]), line=dict(color="#9b2226", width=1, dash="dot"), row=2, col=1)
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.08, x=0),
                              hovermode="x unified")
            fig.update_yaxes(title_text="Return (%)", gridcolor="rgba(120,130,140,0.12)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", gridcolor="rgba(120,130,140,0.12)", row=2, col=1)
            fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # drawdown + risk metric strip
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Max Drawdown", f"{ec['max_dd']:.1f}%",
                      help=f"Largest peak-to-trough decline. Peak {ec.get('peak_date','?')} → trough {ec.get('max_dd_date','?')}.")
            if ec.get("rec_date"):
                d2.metric("Recovered", ec["rec_date"], help="Date the strategy regained its prior peak.")
            else:
                d2.metric("Recovered", "Not yet", help="Strategy has not regained its pre-drawdown peak.")
            d3.metric("Annualized Vol", f"{ec['ann_vol']:.1f}%", help="Annualized daily-return volatility since inception.")
            # active vs NDX
            if ec.get("ndx_total") is not None:
                act = ec["port_total"] - ec["ndx_total"]
                d4.metric("Active vs NDX", f"{act:+.1f}%", help="Strategy total return minus NDX over the same window.")

            st.caption(f"Strategy {ec['port_total']:+.1f}% · NDX {ec.get('ndx_total','n/a')}% · "
                       f"VOO {ec.get('voo_total','n/a')}% since {AI_STRATEGY_INCEPTION}. "
                       "Model curve: target weights × each holding's price path, cash held flat, all indexed to 100 at inception. "
                       "Computed live from current holdings, so editing weights on the Manage tab updates this chart.")

    # ---------------- MANAGE ----------------
    with ai_tabs[2]:
        st.subheader("Manage Portfolio & Bench")
        st.caption("Add or remove names, edit weights/scores, upload a CSV to replace the full list, or download the current list. Changes persist across app restarts. Use Reset to revert to the original 30-holding model.")

        # choose which list to manage
        manage_target = st.radio("Editing", ["Portfolio", "Bench"], horizontal=True, key="ai_manage_target")
        editing_bench = manage_target == "Bench"

        # ---- CSV upload (replaces the active list) ----
        with st.expander("📤 Upload CSV to replace the current list", expanded=False):
            st.caption(
                "Columns recognized (header row required, case-insensitive): "
                "**ticker, name, tier, score, target_weight** (portfolio) or **ticker, name, tier, score, note** (bench). "
                "Extra columns are ignored. Tier accepts HYPER / TIER 1 / TIER 2 / TIER 3. "
                "Uploading replaces the entire active list, so download a backup first if needed."
            )
            up = st.file_uploader("Choose CSV", type=["csv"], key="ai_csv_upload")
            if up is not None:
                try:
                    up_df = pd.read_csv(up)
                    up_df.columns = [c.strip().lower() for c in up_df.columns]
                    if "ticker" not in up_df.columns:
                        st.error("CSV must include a 'ticker' column.")
                    else:
                        recs = []
                        for _, r in up_df.iterrows():
                            tk = str(r.get("ticker", "")).strip().upper()
                            if not tk or tk == "NAN":
                                continue
                            tier_val = str(r.get("tier", "TIER 2")).strip().upper().replace("HYPERSCALER", "HYPER")
                            if tier_val not in ("HYPER", "TIER 1", "TIER 2", "TIER 3"):
                                tier_val = "TIER 2"
                            rec = {
                                "ticker": tk,
                                "name": str(r.get("name", tk)).strip() if not pd.isna(r.get("name", tk)) else tk,
                                "tier": tier_val,
                                "score": float(r["score"]) if "score" in up_df.columns and not pd.isna(r.get("score")) else 0.0,
                            }
                            if editing_bench:
                                rec["note"] = str(r.get("note", "")).strip() if not pd.isna(r.get("note", "")) else ""
                            else:
                                rec["target_weight"] = float(r["target_weight"]) if "target_weight" in up_df.columns and not pd.isna(r.get("target_weight")) else 0.0
                                rec["conviction_date"] = str(r.get("conviction_date", "")).strip() if not pd.isna(r.get("conviction_date", "")) else ""
                                rec["thesis"] = str(r.get("thesis", "")).strip() if not pd.isna(r.get("thesis", "")) else ""
                            recs.append(rec)
                        if not recs:
                            st.warning("No valid rows found in the CSV.")
                        else:
                            st.success(f"Parsed {len(recs)} names. Click below to apply.")
                            if st.button(f"✅ Replace {manage_target} with uploaded list", type="primary", key="ai_apply_csv"):
                                if editing_bench:
                                    ai_save_bench(recs)
                                else:
                                    ai_save_portfolio(recs)
                                st.cache_data.clear()
                                st.success(f"Replaced {manage_target.lower()} with {len(recs)} names from CSV.")
                                st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

        # ---- quick add-a-name form ----
        with st.expander("➕ Add a single name", expanded=False):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                add_tk = st.text_input("Ticker", key="ai_add_tk", placeholder="PLTR")
                add_tier = st.selectbox("Tier", ["HYPER", "TIER 1", "TIER 2", "TIER 3"], index=2, key="ai_add_tier")
            with ac2:
                add_name = st.text_input("Name", key="ai_add_name", placeholder="Palantir")
                add_score = st.number_input("Score", 0.0, 10.0, 5.0, 0.1, key="ai_add_score")
            with ac3:
                if editing_bench:
                    add_note = st.text_input("Note", key="ai_add_note", placeholder="Why watching / trigger to add")
                    add_weight = 0.0
                else:
                    add_weight = st.number_input("Target Weight %", 0.0, 100.0, 2.0, 0.5, key="ai_add_weight")
                    add_note = ""
            if st.button(f"➕ Add to {manage_target}", key="ai_add_btn"):
                tk = add_tk.strip().upper()
                if not tk:
                    st.error("Ticker required.")
                else:
                    if editing_bench:
                        cur = ai_load_bench()
                        if any(b["ticker"] == tk for b in cur):
                            st.warning(f"{tk} already on the bench.")
                        else:
                            cur.append({"ticker": tk, "name": add_name.strip() or tk, "tier": add_tier, "score": add_score, "note": add_note})
                            ai_save_bench(cur)
                            st.cache_data.clear(); st.success(f"Added {tk} to bench."); st.rerun()
                    else:
                        cur = ai_load_portfolio()
                        if any(h["ticker"] == tk for h in cur):
                            st.warning(f"{tk} already in the portfolio.")
                        else:
                            cur.append({"ticker": tk, "name": add_name.strip() or tk, "tier": add_tier, "score": add_score,
                                        "target_weight": add_weight, "conviction_date": "", "thesis": ""})
                            ai_save_portfolio(cur)
                            st.cache_data.clear(); st.success(f"Added {tk} to portfolio."); st.rerun()

        st.divider()

        # ---- editable table for the chosen list ----
        if editing_bench:
            bench_list = ai_load_bench()
            edit_df = pd.DataFrame(bench_list)[['ticker', 'name', 'tier', 'score', 'note']]
            edited = st.data_editor(
                edit_df, use_container_width=True, num_rows="dynamic", height=520,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", required=True, width="small"),
                    "name": st.column_config.TextColumn("Name"),
                    "tier": st.column_config.SelectboxColumn("Tier", options=["HYPER", "TIER 1", "TIER 2", "TIER 3"]),
                    "score": st.column_config.NumberColumn("Score", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
                    "note": st.column_config.TextColumn("Note", width="large"),
                },
                key="ai_bench_editor",
            )
            bb1, bb2, bb3 = st.columns([1, 1, 2])
            with bb1:
                if st.button("💾 Save Bench", type="primary", use_container_width=True):
                    nb = [b for b in edited.to_dict('records') if b.get('ticker') and str(b['ticker']).strip()]
                    for b in nb:
                        b['ticker'] = str(b['ticker']).strip().upper()
                    ai_save_bench(nb)
                    st.cache_data.clear(); st.success(f"Saved {len(nb)} bench names."); st.rerun()
            with bb2:
                if st.button("↩️ Reset Bench", use_container_width=True):
                    ai_reset_bench()
                    st.cache_data.clear(); st.success("Bench reverted to default."); st.rerun()
            with bb3:
                st.download_button("⬇️ Export Bench CSV", edited.to_csv(index=False), "ai_bench.csv", "text/csv", use_container_width=True)

        else:
            holdings = ai_load_portfolio()
            edit_df = pd.DataFrame(holdings)[['ticker', 'name', 'tier', 'score', 'target_weight', 'conviction_date', 'thesis']]
            edited = st.data_editor(
                edit_df, use_container_width=True, num_rows="dynamic", height=500,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", required=True, width="small"),
                    "name": st.column_config.TextColumn("Name"),
                    "tier": st.column_config.SelectboxColumn("Tier", options=["HYPER", "TIER 1", "TIER 2", "TIER 3"]),
                    "score": st.column_config.NumberColumn("Score", min_value=0.0, max_value=10.0, step=0.1, format="%.1f"),
                    "target_weight": st.column_config.NumberColumn("Target %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                    "conviction_date": st.column_config.TextColumn("Conv. Date", width="small"),
                    "thesis": st.column_config.TextColumn("Thesis", width="large"),
                },
                key="ai_editor",
            )

            total_w = edited['target_weight'].fillna(0).sum()
            wcol1, wcol2 = st.columns([1, 3])
            wcol1.metric("Total Weight", f"{total_w:.1f}%")
            if total_w > 100:
                wcol2.error(f"Weights sum to {total_w:.1f}%, over 100%. Trim before saving.")
            elif total_w < 98:
                wcol2.warning(f"Weights sum to {total_w:.1f}%. Remainder ({100-total_w:.1f}%) is treated as cash.")
            else:
                wcol2.success(f"Weights sum to {total_w:.1f}%. Within range.")

            bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
            with bcol1:
                if st.button("💾 Save Changes", type="primary", use_container_width=True):
                    new_holdings = edited.to_dict('records')
                    new_holdings = [h for h in new_holdings if h.get('ticker') and str(h['ticker']).strip()]
                    for h in new_holdings:
                        h['ticker'] = str(h['ticker']).strip().upper()
                    if ai_save_portfolio(new_holdings):
                        st.cache_data.clear()
                        st.success(f"Saved {len(new_holdings)} holdings.")
                        st.rerun()
            with bcol2:
                if st.button("↩️ Reset to Default", use_container_width=True):
                    ai_reset_portfolio()
                    st.cache_data.clear()
                    st.success("Reverted to the original 30-holding model.")
                    st.rerun()
            with bcol3:
                csv = edited.to_csv(index=False)
                st.download_button("⬇️ Export CSV", csv, "ai_portfolio.csv", "text/csv", use_container_width=True)

    # ---------------- BENCH ----------------
    with ai_tabs[3]:
        st.subheader("Bench & Watch List")
        st.caption("Names under consideration, not currently held. Live prices for monitoring. Edit the bench on the Manage tab (toggle to 'Bench').")
        bench_list = ai_load_bench()
        bench_tickers = [b['ticker'] for b in bench_list]
        with st.spinner("Fetching bench prices..."):
            bprices = ai_fetch_prices(tuple(bench_tickers), AI_STRATEGY_INCEPTION)
        brows = []
        for b in bench_list:
            pr = bprices.get(b['ticker'], {})
            bz, btgt = ai_bench_zone(b['ticker'], pr.get('price_now'))
            brows.append({
                "Tier": b['tier'], "Ticker": b['ticker'], "Name": b['name'],
                "Score": f"{b['score']:.1f}" if b.get('score') is not None else "",
                "Price": f"${pr.get('price_now'):,.2f}" if pr.get('price_now') is not None else "n/a",
                "Since Incept": f"{pr.get('ret_pct'):+.1f}%" if pr.get('ret_pct') is not None else "n/a",
                "Zone": bz,
                "Optimal": f"${btgt['optimal']:,.2f}" if btgt else "—",
                "Ceiling": f"${btgt['ceiling']:,.2f}" if btgt else "—",
                "Note": b.get('note', ''),
            })
        tier_order = {"HYPER": 0, "TIER 1": 1, "TIER 2": 2, "TIER 3": 3}
        brows.sort(key=lambda r: (tier_order.get(r["Tier"], 99), r["Ticker"]))
        st.dataframe(pd.DataFrame(brows), use_container_width=True, hide_index=True, height=560)
        st.caption("Zone/Optimal/Ceiling are tracking targets for bench names (not held). "
                   "Korea-listed OTC names (SK Hynix, Samsung) and pre-revenue names show no targets. "
                   "Same schedule as portfolio: 🟢 buy zone · 🟡 below ceiling · 🔴 above ceiling.")

        # bench equal-weight performance summary
        rets = [bprices.get(b['ticker'], {}).get('ret_pct') for b in bench_list]
        rets = [r for r in rets if r is not None]
        if rets:
            st.metric("Bench (equal-weight, since inception)", f"{sum(rets)/len(rets):+.1f}%",
                      help="Simple average price return of all bench names since 2026-02-10. Compare against the Strategy figure on the Performance tab.")

    # ---------------- MANDATE ----------------
    with ai_tabs[4]:
        st.markdown(AI_MANDATE_MD)

    # ---------------- CONVICTION ----------------
    with ai_tabs[5]:
        st.markdown(AI_CONVICTION_MD)

    # ---------------- FACT SHEET ----------------
    with ai_tabs[6]:
        st.subheader("Manager Fact Sheet")
        st.caption("One-page institutional fact sheet generated from live holdings, weights, performance, and risk statistics. Download the HTML and print to PDF (Ctrl/Cmd-P → Save as PDF).")

        holdings = ai_load_portfolio()
        with st.spinner("Computing performance, risk statistics, and equity curve..."):
            strat_ret, bench_ret, per = ai_compute_performance(holdings, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK)
            hk = tuple((h['ticker'], h.get('target_weight') or 0) for h in holdings)
            ec = ai_equity_curve(hk, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK, "VOO")
            stats = ai_risk_stats(ec) if ec else {}

        # live preview of the headline risk stats
        st.markdown("##### Risk Statistics (live)")
        if stats:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Sharpe", stats.get("sharpe", "n/a"))
            r2.metric("Std Dev (ann)", f"{stats.get('ann_vol','n/a')}%")
            r3.metric("Max Drawdown", f"{stats.get('max_dd','n/a')}%")
            r4.metric("Beta vs NDX", stats.get("beta", "n/a"))
            r5, r6, r7, r8 = st.columns(4)
            r5.metric("Sortino", stats.get("sortino", "n/a"))
            r6.metric("Calmar", stats.get("calmar", "n/a"))
            r7.metric("Alpha (ann)", f"{stats.get('alpha','n/a')}%")
            r8.metric("Info Ratio", stats.get("info_ratio", "n/a"))
        else:
            st.warning("Could not compute risk statistics (price history unavailable). The fact sheet will still generate with performance and holdings data.")

        as_of = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
        html = ai_build_factsheet_html(holdings, per, ec or {}, stats,
                                       strat_ret if strat_ret is not None else 0.0,
                                       bench_ret if bench_ret is not None else 0.0,
                                       AI_STRATEGY_INCEPTION, as_of)

        st.download_button(
            "⬇️ Download Fact Sheet (HTML → print to PDF)",
            html, f"AI_Infrastructure_Strategy_FactSheet_{as_of}.html",
            "text/html", type="primary", use_container_width=True)

        st.markdown("##### Preview")
        st.components.v1.html(html, height=1100, scrolling=True)


# ========================================
# MACRO DASHBOARD
# ========================================

elif selected == "Macro Dashboard":
    st.header("📊 Macro Economic Indicators")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh"):
            with st.spinner("Fetching..."):
                try:
                    st.session_state.macro_analyzer.fetch_macro_data()
                    st.success("✅ Refreshed!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    try:
        regime = st.session_state.macro_analyzer.detect_regime()
        
        st.subheader("Current Market Regime")
        col1, col2, col3 = st.columns(3)
        col1.metric("Regime Type", regime['type'])
        col2.metric("Environment", regime['environment'])
        col3.metric("Equity Bias", regime['equity_bias'])
        
        if regime['preferred_tickers']:
            st.info(f"**Preferred:** {', '.join(regime['preferred_tickers'])}")
        if regime['avoid_tickers']:
            st.warning(f"**Avoid:** {', '.join(regime['avoid_tickers'])}")
        
    except Exception as e:
        st.warning("Macro data unavailable")

st.divider()
st.caption("DJR Trading System - Full Trading System with Persistent Storage")
