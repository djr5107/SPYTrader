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
import io as _io_std
import gzip as _gzip_std
import base64 as _base64_std

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

# ========================================
# NAVIGATION — grouped sidebar (Home / Trading Hub / Markets / Research / My Portfolios)
# Sets `selected` to the same page-name strings the page branches below expect,
# so all existing pages keep working. Collapses to a hamburger on mobile.
# ========================================
NAV_GROUPS = {
    "🏠 Home": ["Home"],
    "📈 Trading Hub": ["Trading Hub", "Signal History", "Backtest", "Trade Log", "Performance", "Premium Seller"],
    "🌐 Markets": ["Market Dashboard", "Macro Dashboard"],
    "🔬 Research": ["Chart Analysis", "Options Chain"],
    "💼 My Portfolios": ["AI Strategy"],
}
PAGE_ICONS = {
    "Home": "house", "Trading Hub": "activity", "Signal History": "clock-history",
    "Backtest": "graph-up-arrow", "Trade Log": "list-ul", "Performance": "trophy",
    "Premium Seller": "shield-check", "Market Dashboard": "speedometer2",
    "Macro Dashboard": "globe", "Chart Analysis": "bar-chart", "Options Chain": "currency-exchange",
    "AI Strategy": "cpu",
}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

with st.sidebar:
    st.markdown("### DJR Trading System")
    for group, pages in NAV_GROUPS.items():
        if len(pages) == 1:
            # single-page group: one button
            if st.button(group, use_container_width=True, key=f"navbtn_{pages[0]}"):
                st.session_state.nav_page = pages[0]
        else:
            with st.expander(group, expanded=(st.session_state.nav_page in pages)):
                for pg in pages:
                    label = ("▸ " if st.session_state.nav_page == pg else "　") + pg
                    if st.button(label, use_container_width=True, key=f"navbtn_{pg}"):
                        st.session_state.nav_page = pg
    st.divider()

selected = st.session_state.nav_page

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


# Trading settings now live in the Trading Hub page (not the global sidebar).
# Module-level defaults so signal functions always have values regardless of page.
TRADING_MODE = "Paper Trading"
MIN_CONVICTION = 5
ENABLE_MACRO_FILTER = True
USE_DYNAMIC_STOPS = True
SIGNAL_EXPIRATION_MINUTES = DEFAULT_SIGNAL_EXPIRATION
ENABLE_CONTINUATION_SIGNALS = True
ENABLE_SUPPORT_RESISTANCE = True
ENABLE_OPTIONS_SIGNALS = True
ENABLE_TREND_FOLLOWING = True
ENABLE_MEAN_REVERSION = True
STOP_LOSS_PCT = -2.0
TRAILING_STOP_PCT = 1.0


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

# Fetch market data
@st.cache_data(ttl=60)
def fetch_market_data():
    data = {}
    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")
            
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                data[ticker] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': hist['Volume'].iloc[-1]
                }
            else:
                data[ticker] = {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0}
        except:
            data[ticker] = {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0}
    
    return data

market_data = fetch_market_data()

# Calculate technical indicators
@st.cache_data(ttl=300)
def calculate_technical_indicators(df, periods=[10, 20, 50, 100, 200]):
    """Calculate comprehensive technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    
    # SMAs
    for period in periods:
        if len(df) >= period:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # RSI
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    if len(df) >= 26:
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ADX
    if len(df) >= 14:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['ADX'] = dx.ewm(alpha=1/14).mean()
        df['ATR'] = atr
    
    # Stochastic
    if len(df) >= 14:
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    
    # Volume ratio
    if len(df) >= 20:
        df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_Avg']
    
    # Bollinger Bands
    if len(df) >= 20:
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
        df['BB_Middle'] = sma_20
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20
    
    # Support and Resistance levels (pivot points)
    if len(df) >= 5:
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])

    try:
        _dir = (df['Close'].diff() > 0).astype(int) - (df['Close'].diff() < 0).astype(int)
        df['OBV'] = (_dir * df['Volume']).fillna(0).cumsum()
    except Exception:
        pass
    try:
        for _ep in [12, 26, 50]:
            df[f'EMA_{_ep}'] = df['Close'].ewm(span=_ep, adjust=False).mean()
    except Exception:
        pass

    return df


# ---------- Technical scorecard: multi-indicator buy/sell/hold ----------
def ta_scorecard(df):
    """Evaluate a panel of technical indicators and return per-indicator signals
    plus an aggregate buy/sell/hold verdict. Each indicator votes -1 (bearish),
    0 (neutral), or +1 (bullish). Expects the columns from
    calculate_technical_indicators (plus OBV/EMA added there)."""
    if df is None or df.empty or len(df) < 30:
        return None
    last = df.iloc[-1]
    px = float(last["Close"])
    rows = []

    def add(name, signal, detail):
        rows.append({"indicator": name, "signal": signal, "detail": detail})

    # RSI (14)
    if "RSI" in df and not pd.isna(last["RSI"]):
        rsi = float(last["RSI"])
        sig = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
        add("RSI (14)", sig, f"{rsi:.0f} — {'oversold' if rsi<30 else ('overbought' if rsi>70 else 'neutral')}")

    # MACD vs signal
    if "MACD" in df and "MACD_Signal" in df and not pd.isna(last["MACD"]):
        macd, sigl = float(last["MACD"]), float(last["MACD_Signal"])
        sig = 1 if macd > sigl else (-1 if macd < sigl else 0)
        add("MACD", sig, f"{'above' if macd>sigl else 'below'} signal line")

    # Moving-average trend: price vs SMA50 / SMA200, and golden/death cross
    if "SMA_50" in df and "SMA_200" in df and not pd.isna(last.get("SMA_200")):
        s50, s200 = float(last["SMA_50"]), float(last["SMA_200"])
        sig = 1 if (px > s50 and s50 > s200) else (-1 if (px < s50 and s50 < s200) else 0)
        add("Trend (50/200 MA)", sig,
            f"price {'>' if px>s50 else '<'} 50-MA, 50-MA {'>' if s50>s200 else '<'} 200-MA")
    elif "SMA_50" in df and not pd.isna(last.get("SMA_50")):
        s50 = float(last["SMA_50"])
        sig = 1 if px > s50 else -1
        add("Trend (50-MA)", sig, f"price {'above' if px>s50 else 'below'} 50-day MA")

    # ADX: trend strength (direction from DI if present, else from MA slope)
    if "ADX" in df and not pd.isna(last["ADX"]):
        adx = float(last["ADX"])
        # ADX measures strength, not direction; combine with short MA slope for direction
        direction = 0
        if "SMA_20" in df and len(df) > 22 and not pd.isna(df["SMA_20"].iloc[-1]) and not pd.isna(df["SMA_20"].iloc[-2]):
            direction = 1 if df["SMA_20"].iloc[-1] > df["SMA_20"].iloc[-2] else -1
        sig = direction if adx >= 25 else 0
        add("ADX (trend strength)", sig,
            f"{adx:.0f} — {'strong' if adx>=25 else 'weak/range'} trend" + (", rising" if sig>0 else (", falling" if sig<0 else "")))

    # Stochastic %K/%D
    if "Stoch_%K" in df and "Stoch_%D" in df and not pd.isna(last["Stoch_%K"]):
        k, d = float(last["Stoch_%K"]), float(last["Stoch_%D"])
        sig = 1 if (k < 20 and k > d) else (-1 if (k > 80 and k < d) else 0)
        add("Stochastic", sig, f"%K {k:.0f}/%D {d:.0f} — {'oversold' if k<20 else ('overbought' if k>80 else 'mid')}")

    # Bollinger position
    if "BB_Upper" in df and "BB_Lower" in df and not pd.isna(last["BB_Upper"]):
        u, l = float(last["BB_Upper"]), float(last["BB_Lower"])
        sig = 1 if px <= l else (-1 if px >= u else 0)
        add("Bollinger Bands", sig, f"{'at/below lower' if px<=l else ('at/above upper' if px>=u else 'within bands')}")

    # OBV trend (volume confirmation)
    if "OBV" in df and len(df) > 20 and not pd.isna(last.get("OBV")):
        obv_now = float(last["OBV"]); obv_prev = float(df["OBV"].iloc[-20])
        sig = 1 if obv_now > obv_prev else (-1 if obv_now < obv_prev else 0)
        add("OBV (volume)", sig, f"{'accumulation' if sig>0 else ('distribution' if sig<0 else 'flat')} over 20d")

    if not rows:
        return None
    score = sum(r["signal"] for r in rows)
    n = len(rows)
    # verdict thresholds scale with how many indicators we have
    if score >= max(2, n * 0.34):
        verdict = "BUY"
    elif score <= -max(2, n * 0.34):
        verdict = "SELL"
    else:
        verdict = "HOLD"
    bull = sum(1 for r in rows if r["signal"] > 0)
    bear = sum(1 for r in rows if r["signal"] < 0)
    neut = sum(1 for r in rows if r["signal"] == 0)
    return {"rows": rows, "score": score, "n": n, "verdict": verdict,
            "bull": bull, "bear": bear, "neutral": neut}

TA_GLOSSARY = [
    ("RSI (Relative Strength Index)", "Momentum oscillator, 0–100. Below 30 = oversold (possible bounce); above 70 = overbought (possible pullback)."),
    ("MACD", "Moving Average Convergence Divergence. Trend/momentum gauge; the MACD line crossing above its signal line is bullish, below is bearish."),
    ("Moving Averages (50/200)", "Average price over 50 and 200 days. Price above both and 50 above 200 (a 'golden cross') signals an uptrend; the reverse ('death cross') signals a downtrend."),
    ("ADX (Average Directional Index)", "Measures trend STRENGTH (not direction), 0–100. Above 25 = a strong trend worth following; below 20 = choppy/range-bound. We pair it with the 20-day MA slope for direction."),
    ("Stochastic Oscillator", "Compares the close to its recent range, 0–100. Below 20 with %K crossing above %D = oversold buy setup; above 80 with %K below %D = overbought."),
    ("Bollinger Bands", "Volatility bands two standard deviations around the 20-day average. Price riding the upper band can mean overbought; the lower band, oversold."),
    ("ATR (Average True Range)", "Measures volatility (average daily range). Higher ATR = bigger swings; used for stop-loss and position sizing, not direction."),
    ("OBV (On-Balance Volume)", "Running total of volume that adds on up days and subtracts on down days. Rising OBV confirms buying pressure (accumulation); falling confirms selling (distribution)."),
]

# Logging functions
def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id, 
               entry_numeric=None, exit_numeric=None, pnl_numeric=None, dte=None,
               strategy=None, thesis=None, max_hold=None, actual_hold=None, 
               conviction=None, signal_type=None):
    """Log a trade to the trade log"""
    new_row = pd.DataFrame([{
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
        'Actual Hold Minutes': actual_hold,
        'Conviction': conviction,
        'Signal Type': signal_type
    }])
    
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, new_row], ignore_index=True)
    save_json(TRADE_LOG_FILE, st.session_state.trade_log.to_dict('records'))

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
    {"ticker": "NVDA", "name": "NVIDIA", "tier": "TIER 1", "score": 8.7, "target_weight": 8.0, "conviction_date": "Q1 25", "thesis": "85%+ GPU share for AI training. PEG 0.73. Blackwell demand exceeds supply into H2 2026. 73% gross margins, $45B net cash. CUDA lock-in. ER 2/25 is major catalyst. Risk: custom ASIC, China export controls, DeepSeek efficiency."},
    {"ticker": "AVGO", "name": "Broadcom", "tier": "TIER 1", "score": 8.0, "target_weight": 5.0, "conviction_date": "Q1 25", "thesis": "Leading custom ASIC designer (Google TPU, Meta MTIA). AI revenue tripled YoY. VMware adds recurring SW revenue. 2025 return +50.6%. Strong networking portfolio. Risk: customer concentration, premium valuation."},
    {"ticker": "TSM", "name": "TSMC (ADR)", "tier": "TIER 1", "score": 8.5, "target_weight": 4.0, "conviction_date": "Q1 25", "thesis": "Fabs every advanced AI chip (NVDA, AMD, AVGO). 60%+ gross margins on advanced nodes. Only true monopoly in AI supply chain. Arizona fab de-risks geopolitics. Risk: Taiwan/China (primary), cyclicality."},
    {"ticker": "AMD", "name": "AMD", "tier": "TIER 1", "score": 6.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "#2 GPU with MI300X gaining enterprise. Best PEG in semis (0.62). DC revenue +70% YoY. Xilinx for edge AI. Underperformed NVDA in 2025. Add on MI350 evidence. Risk: NVDA dominance."},
    {"ticker": "MU", "name": "Micron", "tier": "TIER 1", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "HBM3E critical bottleneck for AI training. 2026 HBM sold out. Memory pricing favorable. Cyclical history but AI creates structural shift. Risk: oversupply cycles, Samsung/SK competition."},
    {"ticker": "ANET", "name": "Arista Networks", "tier": "TIER 1", "score": 7.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "DC networking leader, 157% ROIC. 800G/1.6T for AI clusters. META/MSFT top customers. ER 2/12 will update outlook. Risk: customer concentration, Cisco competitive threat."},
    {"ticker": "ARM", "name": "ARM Holdings", "tier": "TIER 1", "score": 7.0, "target_weight": 1.0, "conviction_date": "Q2 26", "thesis": "AI compute IP/royalty leverage; added 6/1 as a starter. High multiple (~95x) — re-derive entry on any print."},
    {"ticker": "DELL", "name": "Dell Technologies", "tier": "TIER 1", "score": 6.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Cheapest AI infra play at ~10x fwd P/E. $6B+ AI server pipeline. Enterprise refresh cycle. ER 2/26 critical. Risk: low-margin hardware, SMCI competition."},
    {"ticker": "MRVL", "name": "Marvell Tech", "tier": "TIER 1", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Custom ASIC + networking silicon. Revenue +45% from DC segment. Electro-optics for AI clusters. Higher growth potential than AVGO but more volatile. Risk: execution, customer concentration."},
    {"ticker": "VRT", "name": "Vertiv", "tier": "TIER 2", "score": 7.5, "target_weight": 4.0, "conviction_date": "Q1 25", "thesis": "Pure-play AI DC cooling/power. $9.5B+ backlog +30% QoQ. AI racks 3-10x more heat. 2025 return +42.8%. Expanding liquid cooling. Risk: conversion timing, Schneider/ABB competition."},
    {"ticker": "ETN", "name": "Eaton Corp", "tier": "TIER 2", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "DC electrical infra (UPS, switchgear, PDUs). $13B+ backlog. 20%+ ROIC. Div aristocrat 2%+ yield. More diversified than VRT. Risk: industrial cycle, premium valuation."},
    {"ticker": "ASML", "name": "ASML (ADR)", "tier": "TIER 2", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "EUV lithography monopoly. $35B+ backlog. Every advanced AI chip runs through ASML machines. Risk: cyclical equipment spend, China restriction reduces TAM, premium valuation."},
    {"ticker": "APH", "name": "Amphenol", "tier": "TIER 2", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q4 25", "thesis": "High-speed connectors/cables for AI racks. Every GPU cluster needs Amphenol interconnects. 25%+ ROE, 20%+ ROIC. M&A machine. Risk: diversified industrial dampens AI sensitivity."},
    {"ticker": "AMAT", "name": "Applied Materials", "tier": "TIER 2", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q2 25", "thesis": "Semi equipment leader, tools for TSMC/Samsung/Intel. AI chip demand drives fab buildout. Picks-and-shovels play. Risk: cyclical spend, China export restrictions."},
    {"ticker": "TT", "name": "Trane Technologies", "tier": "TIER 2", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q4 25", "thesis": "DC HVAC/cooling. 35% ROE, consistent compounder. AI cooling demand structural. More diversified than VRT. Risk: indirect AI exposure, premium valuation."},
    {"ticker": "CSCO", "name": "Cisco", "tier": "TIER 2", "score": 5.5, "target_weight": 2.0, "conviction_date": "Q1 25", "thesis": "Defensive. 2.8% yield. Splunk adds AI observability. Enterprise networking refresh. Lower beta ballast. Risk: slow innovation vs Arista, legacy decline."},
    {"ticker": "PANW", "name": "Palo Alto Networks", "tier": "TIER 2", "score": 7.5, "target_weight": 2.5, "conviction_date": "Q2 26", "thesis": "Cybersecurity platform leader, AI-driven demand; added 6/1. Reports can re-rate levels — refresh after prints."},
    {"ticker": "VST", "name": "Vistra Corp", "tier": "TIER 3", "score": 7.5, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Largest competitive US power gen (nuclear+gas+solar). ~35% off ATH, entry opportunity. AI DCs consume 1-2 GW each. Nuclear renaissance thesis. Risk: regulation, nat gas exposure."},
    {"ticker": "CEG", "name": "Constellation Energy", "tier": "TIER 3", "score": 7.0, "target_weight": 3.0, "conviction_date": "Q1 25", "thesis": "Largest US nuclear fleet. TMI restart deal with MSFT (20-yr PPA). Clean energy premium. Nuclear capacity irreplaceable. Risk: TMI execution, regulatory timelines."},
    {"ticker": "GEV", "name": "GE Vernova", "tier": "TIER 3", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Gas turbine orders surging for DC baseload. Grid equipment (transformers) multi-year backlogs. Only near-term bridge for AI power. Risk: turbine execution, offshore wind losses."},
    {"ticker": "PWR", "name": "Quanta Services", "tier": "TIER 3", "score": 6.5, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Largest US electrical contractor. Builds transmission/substations. $30B+ backlog. Grid investment deferred 20 years. Risk: labor shortages, execution."},
    {"ticker": "CCJ", "name": "Cameco", "tier": "TIER 3", "score": 6.0, "target_weight": 2.0, "conviction_date": "Q3 25", "thesis": "Premier uranium producer + Westinghouse JV. 10+ yrs underinvestment in fuel supply. Long-term contracts. Risk: uranium volatility, Kazakhstan competition."},
    {"ticker": "NEE", "name": "NextEra Energy", "tier": "TIER 3", "score": 4.5, "target_weight": 1.0, "conviction_date": "Q1 25", "thesis": "LOWERED conviction: NEE/Dominion deal (announced 5/18) makes it the 3rd-largest US energy co and a likely poster child for data-center affordability attacks. Still a fine company, but public-sentiment/regulatory risk rose and the deal adds a 12-18mo overhang. Trimmed to watch-level; paired with SO for the regulated slot."},
    {"ticker": "SO", "name": "Southern Company", "tier": "TIER 3", "score": 5.5, "target_weight": 1.0, "conviction_date": "Q2 26", "thesis": "Regulated Southeast utility; cleanest like-for-like swap for NEE's ballast role without the affordability-poster-child risk from the NEE/Dominion deal. 50+ GW data-center pipeline (Georgia Power), $81B regulated capex driving ~9% rate-base growth through 2030. Q1 EPS $1.32 beat $1.21; targets raised to $104-105. Lower political-target profile than the new NEE. Defensive, ~3% yield. Risk: Vogtle cost history, rate-case timing, not cheap after the utility rally."},
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
# ===== AI STRATEGY: entry targets, search, performance, fact sheet, monitor (added) =====

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
    "SO": {"optimal": 86.95, "secondary": 90.19, "ceiling": 98.05, "pe_anchor": "~20x", "reval": False},
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

def ai_risk_stats(ec, rf_annual=0.043):
    """Derive risk statistics from the equity-curve dict produced by
    ai_equity_curve(). Returns a dict of fact-sheet metrics. All derived from
    the daily MODEL curve, so they describe the strategy allocation, not fills."""
    import numpy as _np
    stats = {}
    port = ec.get("port") or []
    ndx = ec.get("ndx") or []
    if len(port) < 5:
        return stats
    # rebuild daily levels from cumulative % to compute daily returns
    lvl = _np.array([1 + p / 100 for p in port], dtype=float)
    rets = _np.diff(lvl) / lvl[:-1]
    # benchmark daily returns (align lengths)
    bret = None
    nlvl = None
    if ndx and len([x for x in ndx if x is not None]) >= 5:
        nlvl = _np.array([1 + (x or 0) / 100 for x in ndx], dtype=float)
        bret = _np.diff(nlvl) / nlvl[:-1]
    ann = 252
    mean_d = rets.mean()
    std_d = rets.std(ddof=1)
    ann_ret_geo = (lvl[-1] / lvl[0]) ** (ann / len(rets)) - 1   # annualized (geometric)
    ann_vol = std_d * _np.sqrt(ann)
    rf_d = (1 + rf_annual) ** (1 / ann) - 1
    sharpe = ((mean_d - rf_d) / std_d * _np.sqrt(ann)) if std_d > 0 else None
    downside = rets[rets < 0]
    sortino = ((mean_d - rf_d) / downside.std(ddof=1) * _np.sqrt(ann)) if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    # drawdown already in ec
    max_dd = ec.get("max_dd")
    calmar = (ann_ret_geo * 100 / abs(max_dd)) if max_dd else None
    # beta / alpha / tracking error / IR vs NDX
    beta = alpha = te = ir = up_capture = down_capture = corr = None
    if bret is not None and len(bret) == len(rets):
        cov = _np.cov(rets, bret)
        if cov[1, 1] > 0:
            beta = cov[0, 1] / cov[1, 1]
        corr = _np.corrcoef(rets, bret)[0, 1]
        active = rets - bret
        te = active.std(ddof=1) * _np.sqrt(ann)
        ir = (active.mean() * ann) / te if te > 0 else None
        b_ann_geo = (nlvl[-1] / nlvl[0]) ** (ann / len(bret)) - 1
        rf_ann = rf_annual
        if beta is not None:
            alpha = (ann_ret_geo - rf_ann) - beta * (b_ann_geo - rf_ann)
        up = bret > 0
        dn = bret < 0
        if up.sum() > 0 and bret[up].sum() != 0:
            up_capture = rets[up].sum() / bret[up].sum() * 100
        if dn.sum() > 0 and bret[dn].sum() != 0:
            down_capture = rets[dn].sum() / bret[dn].sum() * 100
    pos = (rets > 0).sum()
    win_rate = pos / len(rets) * 100
    stats.update({
        "ann_return": round(ann_ret_geo * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "calmar": round(calmar, 2) if calmar is not None else None,
        "max_dd": max_dd,
        "beta": round(beta, 2) if beta is not None else None,
        "alpha": round(alpha * 100, 2) if alpha is not None else None,
        "tracking_error": round(te * 100, 2) if te is not None else None,
        "info_ratio": round(ir, 2) if ir is not None else None,
        "up_capture": round(up_capture, 1) if up_capture is not None else None,
        "down_capture": round(down_capture, 1) if down_capture is not None else None,
        "correlation": round(corr, 2) if corr is not None else None,
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
    risk_rows = (
        f"<tr><td>Annualized return</td><td class='r'>{stat(stats.get('ann_return'),'%')}</td></tr>"
        f"<tr><td>Annualized volatility (std dev)</td><td class='r'>{stat(stats.get('ann_vol'),'%')}</td></tr>"
        f"<tr><td>Sharpe ratio</td><td class='r'>{stat(stats.get('sharpe'))}</td></tr>"
        f"<tr><td>Sortino ratio</td><td class='r'>{stat(stats.get('sortino'))}</td></tr>"
        f"<tr><td>Max drawdown</td><td class='r neg'>{stat(stats.get('max_dd'),'%')}</td></tr>"
        f"<tr><td>Calmar ratio</td><td class='r'>{stat(stats.get('calmar'))}</td></tr>"
        f"<tr><td>Win rate (daily)</td><td class='r'>{stat(stats.get('win_rate'),'%')}</td></tr>"
    )
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

@st.cache_data(ttl=3600)
def em_earnings_info(ticker):
    """Next earnings date + most recent past earnings date from yfinance.
    Returns dict with days_until (or None) and reported_recently (<=3 trading days)."""
    out = {"next": None, "last": None, "days_until": None, "reported_recently": False}
    try:
        t = yf.Ticker(ticker)
        cal = None
        try:
            cal = t.get_earnings_dates(limit=8)
        except Exception:
            cal = None
        now = pd.Timestamp.now(tz="America/New_York")
        if cal is not None and not cal.empty:
            idx = cal.index.tz_convert("America/New_York") if cal.index.tz is not None else cal.index.tz_localize("America/New_York")
            future = [d for d in idx if d >= now]
            past = [d for d in idx if d < now]
            if future:
                nd = min(future)
                out["next"] = nd.strftime("%Y-%m-%d")
                out["days_until"] = (nd.normalize() - now.normalize()).days
            if past:
                ld = max(past)
                out["last"] = ld.strftime("%Y-%m-%d")
                # reported within ~3 trading days (use 5 calendar days as proxy)
                out["reported_recently"] = (now.normalize() - ld.normalize()).days <= 5
    except Exception:
        pass
    return out

@st.cache_data(ttl=1800)
def em_recent_move(ticker):
    """1-day % move vs prior close, to drive the gap-based re-rate flag."""
    try:
        h = yf.Ticker(ticker).history(period="5d", interval="1d")
        if len(h) >= 2:
            return float((h['Close'].iloc[-1] / h['Close'].iloc[-2] - 1) * 100)
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800)
def em_forward_pe_eps(ticker):
    """Forward EPS + forward P/E from yfinance info, used to recompute the
    P/E-anchored ceiling. Returns (forward_eps, forward_pe) or (None, None)."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("forwardEps"), info.get("forwardPE")
    except Exception:
        return None, None

def em_anchor_multiple(ticker):
    """Parse the numeric multiple out of the AI_ENTRY_TARGETS pe_anchor string,
    e.g. '~40x FY28' -> 40.0. Returns None if not parseable."""
    t = AI_ENTRY_TARGETS.get(ticker) or {}
    s = str(t.get("pe_anchor", ""))
    import re as _re
    m = _re.search(r'(\d+(?:\.\d+)?)\s*x', s)
    return float(m.group(1)) if m else None

def em_recomputed_ceiling(ticker):
    """If we have forward EPS and a parseable anchor multiple, recompute the
    do-not-exceed ceiling = anchor_multiple x forward_eps. Returns float or None.
    This is the deterministic, no-LLM way the ceiling 'travels with estimates'."""
    fwd_eps, _ = em_forward_pe_eps(ticker)
    mult = em_anchor_multiple(ticker)
    if fwd_eps and mult and fwd_eps > 0:
        return round(mult * fwd_eps, 2)
    return None

@st.cache_data(ttl=1800)
def em_headlines(ticker, n=5):
    """Recent headlines from yfinance. Returns list of (title, publisher, url)."""
    out = []
    try:
        news = yf.Ticker(ticker).news or []
        for item in news[:n]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title", "")
            pub = (content.get("provider", {}) or {}).get("displayName") or item.get("publisher", "")
            url = (content.get("canonicalUrl", {}) or {}).get("url") or item.get("link", "")
            if title:
                out.append((title, pub, url))
    except Exception:
        pass
    return out

def em_review_flag(ticker, price_now, gap_threshold=10.0):
    """Combine signals into a single review status. Pure rules, no LLM.
    Priority: explicit reval flag > recent earnings > large gap > stale-vs-ceiling > clear."""
    reasons = []
    t = AI_ENTRY_TARGETS.get(ticker) or {}
    if t.get("reval"):
        reasons.append("flagged for re-rate")
    info = em_earnings_info(ticker)
    if info.get("reported_recently"):
        reasons.append(f"reported {info.get('last')}")
    if info.get("days_until") is not None and info["days_until"] <= 7:
        reasons.append(f"reports in {info['days_until']}d ({info.get('next')})")
    mv = em_recent_move(ticker)
    if mv is not None and abs(mv) >= gap_threshold:
        reasons.append(f"gapped {mv:+.0f}% today")
    if price_now is not None and t.get("ceiling") and price_now > t["ceiling"] * 1.12:
        reasons.append("price >12% past ceiling")
    status = "🔴 REVIEW" if reasons else "🟢 OK"
    return status, reasons

@st.cache_data(ttl=1800)
def em_earnings_data(ticker):
    """Gather a structured earnings data package for the analysis prompt.
    Returns a dict with all fields, each gracefully None on failure."""
    out = {"ticker": ticker, "eps_history": [], "revenue": None, "revenue_growth": None,
           "eps_ttm": None, "eps_fwd": None, "eps_growth": None, "target_price": None,
           "analyst_rec": None, "analyst_count": None, "rec_breakdown": {},
           "recent_move_1d": None, "recent_move_5d": None, "recent_move_ytd": None,
           "market_cap": None, "pe_fwd": None, "pe_trail": None, "peg": None,
           "gross_margin": None, "oper_margin": None, "free_cashflow": None,
           "next_earnings": None, "last_earnings_date": None}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["eps_ttm"]       = info.get("trailingEps")
        out["eps_fwd"]       = info.get("forwardEps")
        out["eps_growth"]    = info.get("earningsGrowth")
        out["revenue"]       = info.get("totalRevenue")
        out["revenue_growth"]= info.get("revenueGrowth")
        out["target_price"]  = info.get("targetMeanPrice")
        out["analyst_rec"]   = info.get("recommendationKey", "").replace("_", " ").title()
        out["analyst_count"] = info.get("numberOfAnalystOpinions")
        out["market_cap"]    = info.get("marketCap")
        out["pe_fwd"]        = info.get("forwardPE")
        out["pe_trail"]      = info.get("trailingPE")
        out["peg"]           = info.get("trailingPegRatio") or info.get("pegRatio")
        out["gross_margin"]  = info.get("grossMargins")
        out["oper_margin"]   = info.get("operatingMargins")
        out["free_cashflow"] = info.get("freeCashflow")
        # Recommendation breakdown (strong buy/buy/hold/sell)
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                recent_recs = recs.tail(20)
                for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                    if col in recent_recs.columns:
                        out["rec_breakdown"][col] = int(recent_recs[col].sum())
        except Exception:
            pass
        # EPS beat-miss history from get_earnings_dates (yfinance 0.2.x+)
        try:
            edates = t.get_earnings_dates(limit=8)
            if edates is not None and not edates.empty:
                for idx, row in edates.iterrows():
                    est = row.get("EPS Estimate"); rep = row.get("Reported EPS")
                    surp = row.get("Surprise(%)")
                    if pd.notna(est) or pd.notna(rep):
                        out["eps_history"].append({
                            "date": str(idx.date()),
                            "estimate": round(float(est), 3) if pd.notna(est) else None,
                            "reported": round(float(rep), 3) if pd.notna(rep) else None,
                            "surprise_pct": round(float(surp), 1) if pd.notna(surp) else None,
                        })
        except Exception:
            pass
        # Recent price action
        try:
            h = t.history(period="1y", interval="1d")
            if not h.empty:
                p_now = float(h["Close"].iloc[-1])
                p_1d  = float(h["Close"].iloc[-2]) if len(h) > 1 else None
                p_5d  = float(h["Close"].iloc[-6]) if len(h) > 5 else None
                p_ytd_start = h[h.index.year == h.index[-1].year]["Close"].iloc[0]
                if p_1d:  out["recent_move_1d"]  = round((p_now/p_1d  - 1)*100, 2)
                if p_5d:  out["recent_move_5d"]  = round((p_now/p_5d  - 1)*100, 2)
                out["recent_move_ytd"] = round((p_now/float(p_ytd_start) - 1)*100, 2)
        except Exception:
            pass
        # Next/last earnings
        try:
            ei = em_earnings_info(ticker)
            out["next_earnings"]      = ei.get("next")
            out["last_earnings_date"] = ei.get("last")
        except Exception:
            pass
    except Exception:
        pass
    return out

def em_earnings_analysis(api_key, ticker, model="claude-haiku-4-5-20251001"):
    """Full earnings analysis using the user's own Anthropic API key.
    Pulls a comprehensive data package and asks Claude for a structured
    analysis covering: last-quarter results, guidance, thesis validation,
    and whether conviction should change. Never uses a built-in key."""
    if not api_key:
        return None, "No API key provided."
    # gather data
    data = em_earnings_data(ticker)
    heads = em_headlines(ticker, n=8)
    # find this holding's thesis from the portfolio (if loaded)
    thesis = ""
    try:
        for h in ai_load_portfolio():
            if h["ticker"] == ticker:
                thesis = h.get("thesis", ""); break
    except Exception:
        pass
    # format the data package for the prompt
    def _fmt(v, suffix="", scale=1, decimals=2):
        return f"{v*scale:.{decimals}f}{suffix}" if v is not None else "n/a"
    def _fmt_big(v):
        if v is None: return "n/a"
        if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"
    eps_block = ""
    if data["eps_history"]:
        eps_block = "Quarterly EPS history (most recent first):\n"
        for e in data["eps_history"][:4]:
            surp = f" (surprise {e['surprise_pct']:+.1f}%)" if e['surprise_pct'] is not None else ""
            eps_block += f"  {e['date']}: est {e['estimate']}, reported {e['reported']}{surp}\n"
    else:
        eps_block = "Quarterly EPS history: not available from data provider.\n"
    rec_block = ""
    if data["rec_breakdown"]:
        r = data["rec_breakdown"]
        rec_block = (f"Recent analyst votes: strong buy {r.get('strongBuy',0)}, "
                     f"buy {r.get('buy',0)}, hold {r.get('hold',0)}, "
                     f"sell {r.get('sell',0)}, strong sell {r.get('strongSell',0)}")
    news_block = "\n".join(f"- {h[0]} ({h[1]})" for h in heads) or "No recent headlines."
    prompt = f"""You are a buy-side equity analyst at an AI-focused hedge fund.
Analyze the following data for {ticker} and produce a concise, actionable earnings analysis.

=== FUNDAMENTAL DATA ===
EPS (trailing/forward): {_fmt(data['eps_ttm'])}/{_fmt(data['eps_fwd'])}
EPS growth (YoY): {_fmt(data['eps_growth'], '%', 100)}
Revenue (TTM): {_fmt_big(data['revenue'])} | Revenue growth: {_fmt(data['revenue_growth'], '%', 100)}
Gross margin: {_fmt(data['gross_margin'], '%', 100)} | Operating margin: {_fmt(data['oper_margin'], '%', 100)}
Free cash flow: {_fmt_big(data['free_cashflow'])}
Fwd P/E: {_fmt(data['pe_fwd'])} | PEG: {_fmt(data['peg'])}
Market cap: {_fmt_big(data['market_cap'])} | Analyst price target: {_fmt(data['target_price'], '$' if data['target_price'] else '')}
Analyst consensus: {data['analyst_rec'] or 'n/a'} ({data['analyst_count'] or '?'} analysts)
{rec_block}

=== RECENT PRICE ACTION ===
1-day: {_fmt(data['recent_move_1d'], '%')} | 5-day: {_fmt(data['recent_move_5d'], '%')} | YTD: {_fmt(data['recent_move_ytd'], '%')}
Last earnings date: {data['last_earnings_date'] or 'n/a'} | Next: {data['next_earnings'] or 'n/a'}

=== QUARTERLY EPS HISTORY ===
{eps_block}
=== RECENT HEADLINES ===
{news_block}

=== PORTFOLIO THESIS ===
{thesis or '(no thesis stored)'}

=== TASK ===
Produce a structured earnings analysis with EXACTLY these four sections:

**LAST QUARTER RESULTS**
Two-to-three sentences on the most recent quarter: EPS beat or miss vs estimate, revenue growth, margin trajectory. Quantify where possible.

**GUIDANCE & FORWARD OUTLOOK**
Two sentences on what the company is signaling about the next quarter or year, based on the data and headlines.

**THESIS VALIDATION**
One or two sentences: do the results confirm, challenge, or are neutral to the portfolio thesis above? Cite specific data points.

**CONVICTION: RAISE / MAINTAIN / LOWER**
One sentence verdict with the word RAISE, MAINTAIN, or LOWER. Then one sentence on the single biggest risk to watch.
"""
    import urllib.request as _ur, json as _js, urllib.error as _ue
    # Strip whitespace (common mobile-paste issue) and validate format.
    key = api_key.strip()
    if not key.startswith("sk-ant-"):
        return data, (
            "⚠️ Key format looks wrong — Anthropic keys start with `sk-ant-`. "
            f"Got {len(key)} chars starting with `{key[:8]}...`. "
            "Copy the full key from console.anthropic.com → API Keys."
        )
    try:
        body = _js.dumps({
            "model": model,
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = _ur.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        with _ur.urlopen(req, timeout=45) as r:
            resp = _js.loads(r.read())
        text = "".join(p.get("text", "") for p in resp.get("content", []) if p.get("type") == "text")
        return data, text or "No response."
    except _ue.HTTPError as e:
        code = e.code
        if code == 401:
            return data, (
                "❌ **401 Unauthorized** — the API key was rejected. Most likely causes:\n"
                "- The key is truncated (mobile paste often cuts it off). "
                f"Your key is {len(key)} chars; Anthropic keys are ~108 chars.\n"
                "- The key has been revoked or expired.\n"
                "- Billing isn't set up on this key.\n\n"
                "Check at [console.anthropic.com → API Keys](https://console.anthropic.com/settings/keys)."
            )
        if code == 429:
            return data, "❌ **429 Rate limited** — you've hit your API rate limit. Wait a moment and try again."
        if code == 400:
            try:
                detail = _js.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                detail = ""
            return data, f"❌ **400 Bad request**: {detail or e.reason}"
        return data, f"❌ **HTTP {code}**: {e.reason}"
    except Exception as e:
        return data, f"❌ Unexpected error: {e}"
# ===== Transactions / batting average / attribution helpers (added) =====
AI_REBALANCE_DATE = "2026-06-01"   # rebalance executed at this session's close
AI_PORTFOLIO_NOTIONAL = 100000.0   # $100k example book for share counts

@st.cache_data(ttl=900)
def ai_prices_on_date(tickers, target_date):
    """Close price on (or the last trading day before) target_date for each ticker.
    Returns dict ticker -> float or None. Uses auto-adjusted closes for consistency
    with the rest of the app; over a few months the adjustment effect is negligible."""
    out = {}
    if not tickers:
        return out
    try:
        start = (pd.Timestamp(target_date) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(target_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        data = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return {t: None for t in tickers}
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']] if 'Close' in data else data
            if len(tickers) == 1:
                close.columns = list(tickers)
        tgt = pd.Timestamp(target_date)
        for t in tickers:
            try:
                s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
                s = s[s.index <= tgt]
                out[t] = float(s.iloc[-1]) if not s.empty else None
            except Exception:
                out[t] = None
    except Exception:
        out = {t: None for t in tickers}
    return out

def ai_build_transactions(holdings, exec_date, notional=AI_PORTFOLIO_NOTIONAL):
    """Model the rebalance as establishing every target position at exec_date's
    close, sized to `notional`. Returns (rows, totals). Whole-share rounding."""
    tickers = [h['ticker'] for h in holdings if h.get('ticker') and (h.get('target_weight') or 0) > 0]
    px = ai_prices_on_date(tuple(tickers), exec_date)
    rows = []
    invested = 0.0
    for h in holdings:
        tk = h['ticker']
        w = h.get('target_weight') or 0
        if w <= 0:
            continue
        p = px.get(tk)
        if p and p > 0:
            target_dollars = notional * (w / 100.0)
            shares = int(target_dollars // p)   # whole shares, no fractional
            value = shares * p
            invested += value
        else:
            shares, value = None, None
        rows.append({"ticker": tk, "name": h.get('name', tk), "tier": h.get('tier', ''),
                     "weight": w, "price": p, "shares": shares, "value": value})
    cash = notional - invested
    totals = {"invested": invested, "cash": cash, "notional": notional}
    return rows, totals

def ai_batting_average(per, ndx_total, voo_total):
    """Per-holding outperformance vs NDX (primary) and VOO/S&P 500 (secondary),
    measured on since-inception return. Returns dict with counts, rates, rows."""
    rows = []
    beat_ndx = beat_voo = total = 0
    for h in per:
        r = h.get('ret_pct')
        if r is None:
            continue
        total += 1
        bn = ndx_total is not None and r > ndx_total
        bv = voo_total is not None and r > voo_total
        beat_ndx += 1 if bn else 0
        beat_voo += 1 if bv else 0
        rows.append({"ticker": h['ticker'], "tier": h.get('tier', ''), "ret": r,
                     "vs_ndx": (r - ndx_total) if ndx_total is not None else None,
                     "vs_voo": (r - voo_total) if voo_total is not None else None,
                     "beat_ndx": bn, "beat_voo": bv})
    return {
        "rows": rows, "total": total,
        "beat_ndx": beat_ndx, "beat_voo": beat_voo,
        "ba_ndx": (beat_ndx / total * 100) if total else None,
        "ba_voo": (beat_voo / total * 100) if total else None,
        "ndx_total": ndx_total, "voo_total": voo_total,
    }

def ai_attribution_by_tier(per, bench_per):
    """Brinson-Fachler tier attribution: portfolio vs the equal-weighted bench
    (watchlist) universe. Segments = the four tiers.
      Allocation_i = (wp_i - wb_i)(Rb_i - Rb)
      Selection_i  = wb_i (Rp_i - Rb_i)
      Interaction_i= (wp_i - wb_i)(Rp_i - Rb_i)
    Sum of all three across tiers = Rp - Rb (active vs the bench universe).
    Returns (rows, totals). Returns are in % points."""
    tiers = ["HYPER", "TIER 1", "TIER 2", "TIER 3"]
    p_w = {t: 0.0 for t in tiers}; p_wr = {t: 0.0 for t in tiers}; tot_w = 0.0
    for h in per:
        t = h.get('tier'); w = h.get('target_weight') or 0; r = h.get('ret_pct')
        if t not in tiers or r is None or w <= 0:
            continue
        p_w[t] += w; p_wr[t] += w * r; tot_w += w
    Rp_i = {t: (p_wr[t] / p_w[t] if p_w[t] > 0 else 0.0) for t in tiers}
    wp_i = {t: (p_w[t] / tot_w if tot_w > 0 else 0.0) for t in tiers}

    b_names = [b for b in bench_per if b.get('ret_pct') is not None and b.get('tier') in tiers]
    Nb = len(b_names)
    b_cnt = {t: 0 for t in tiers}; b_sum = {t: 0.0 for t in tiers}
    for b in b_names:
        t = b['tier']; b_cnt[t] += 1; b_sum[t] += b['ret_pct']
    Rb_i = {t: (b_sum[t] / b_cnt[t] if b_cnt[t] > 0 else 0.0) for t in tiers}
    wb_i = {t: (b_cnt[t] / Nb if Nb > 0 else 0.0) for t in tiers}
    Rb = sum(wb_i[t] * Rb_i[t] for t in tiers)
    Rp = sum(wp_i[t] * Rp_i[t] for t in tiers)

    rows = []; t_alloc = t_sel = t_int = 0.0
    for t in tiers:
        alloc = (wp_i[t] - wb_i[t]) * (Rb_i[t] - Rb)
        sel = wb_i[t] * (Rp_i[t] - Rb_i[t])
        inter = (wp_i[t] - wb_i[t]) * (Rp_i[t] - Rb_i[t])
        t_alloc += alloc; t_sel += sel; t_int += inter
        rows.append({"tier": t, "wp": wp_i[t] * 100, "wb": wb_i[t] * 100,
                     "Rp": Rp_i[t], "Rb": Rb_i[t],
                     "alloc": alloc, "sel": sel, "inter": inter,
                     "total": alloc + sel + inter})
    totals = {"alloc": t_alloc, "sel": t_sel, "inter": t_int,
              "active": Rp - Rb, "Rp": Rp, "Rb": Rb}
    return rows, totals


# ===== Home dashboard data helpers (added) =====
import urllib.request as _urlreq
import xml.etree.ElementTree as _ET

# ---- Ticker tape indices (compact, Yahoo-like) ----
# ── DYNAMIC TICKER TAPE ──────────────────────────────────────────────────────
# Two symbol sets: live equity indices when US markets are open, futures after.
# Non-equity instruments (VIX, Gold, Oil, BTC, rates, Dollar) always show live.

HOME_TAPE_CLOSED = [      # pre/after hours + weekends: show futures
    ("ES=F", "S&P Fut"), ("YM=F", "Dow Fut"), ("NQ=F", "Nasdaq Fut"),
    ("RTY=F", "Rus 2K Fut"), ("^VIX", "VIX"), ("GC=F", "Gold"),
    ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("^TNX", "10Y Yield"),
    ("DX-Y.NYB", "Dollar"),
]
HOME_TAPE_OPEN = [        # 9:30–4:00 ET Mon–Fri: show live index prices
    ("^GSPC", "S&P 500"), ("^DJI", "Dow"), ("^IXIC", "Nasdaq"),
    ("^RUT", "Russell 2K"), ("^VIX", "VIX"), ("GC=F", "Gold"),
    ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("^TNX", "10Y Yield"),
    ("DX-Y.NYB", "Dollar"),
]

def _market_is_open():
    """Return True if US equity markets are currently in the regular session."""
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:                  # Saturday/Sunday
            return False
        op = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        cl = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return op <= now <= cl
    except Exception:
        return False

@st.cache_data(ttl=60)   # 1-min refresh during session; fine for after-hours too
def home_tape_quotes():
    """Compact last/change/%-change. Uses live index prices during open,
    futures symbols after hours. Non-equity items always live."""
    open_now = _market_is_open()
    tape = HOME_TAPE_OPEN if open_now else HOME_TAPE_CLOSED
    syms = [s for s, _ in tape]
    # During open: 5-min intraday for the freshest last price.
    # After close: daily bars (2-day window gives a prev-close for the delta).
    interval = "5m" if open_now else "1d"
    period   = "1d" if open_now else "5d"
    try:
        data = yf.download(syms, period=period, interval=interval,
                           progress=False, auto_adjust=True, group_by="ticker")
    except Exception:
        data = None
    out = []
    for sym, label in tape:
        last = chg = pct = None
        try:
            if data is not None:
                s = (data[sym]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]).dropna()
                if open_now:
                    # intraday: last tick vs session open (first row of today)
                    if len(s) >= 2:
                        last = float(s.iloc[-1])
                        prev = float(s.iloc[0])   # session open
                        chg  = last - prev
                        pct  = chg / prev * 100 if prev else None
                    elif len(s) == 1:
                        last = float(s.iloc[-1])
                else:
                    # daily: last close vs prior close
                    if len(s) >= 2:
                        last = float(s.iloc[-1]); prev = float(s.iloc[-2])
                        chg  = last - prev
                        pct  = chg / prev * 100 if prev else None
                    elif len(s) == 1:
                        last = float(s.iloc[-1])
        except Exception:
            pass
        disp = last
        if sym == "^TNX" and last is not None and last > 100:
            disp = last / 10.0
        out.append({"label": label, "last": disp, "chg": chg, "pct": pct, "open": open_now})
    return out

# ---- Treasury par yield curve (free Treasury fiscal data API) ----
@st.cache_data(ttl=3600)
def home_yield_curve():
    """Latest Treasury par yields across maturities + the prior month for shape
    comparison. Returns dict: {'date':..., 'latest':[(label,yield)...],
    'prior':[(label,yield)...], 'prior_date':...} or {} on failure."""
    base = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
            "/v1/accounting/od/avg_interest_rates")
    # That endpoint is monthly avg; for daily par yields use the daily dataset:
    daily = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
             "/v2/accounting/od/daily_treasury_yield_curve"
             "?sort=-record_date&page[size]=40")
    cols = [("bc_1month","1M"),("bc_2month","2M"),("bc_3month","3M"),("bc_4month","4M"),
            ("bc_6month","6M"),("bc_1year","1Y"),("bc_2year","2Y"),("bc_3year","3Y"),
            ("bc_5year","5Y"),("bc_7year","7Y"),("bc_10year","10Y"),("bc_20year","20Y"),
            ("bc_30year","30Y")]
    try:
        req = _urlreq.Request(daily, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=20) as r:
            js = json.loads(r.read())
        rows = js.get("data", [])
        if not rows:
            return {}
        def parse(row):
            pts = []
            for key, lab in cols:
                v = row.get(key)
                try:
                    if v not in (None, "", "null"):
                        pts.append((lab, float(v)))
                except Exception:
                    pass
            return pts
        latest = parse(rows[0])
        prior = None; prior_date = None
        # find a row ~21 trading rows back (≈1 month) for shape comparison
        if len(rows) > 21:
            prior = parse(rows[21]); prior_date = rows[21].get("record_date")
        return {"date": rows[0].get("record_date"), "latest": latest,
                "prior": prior, "prior_date": prior_date}
    except Exception:
        return {}

def home_yield_curve_figure(yc):
    """Interactive Plotly yield-curve chart: today vs ~1 month ago, with 2s10s."""
    if not yc or not yc.get("latest"):
        return None
    labs = [l for l, _ in yc["latest"]]
    ys = [v for _, v in yc["latest"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labs, y=ys, mode="lines+markers", name=f"Today ({yc.get('date','')})",
                             line=dict(color="#2dd4bf", width=2.5), marker=dict(size=6)))
    if yc.get("prior"):
        pl = [l for l, _ in yc["prior"]]; pv = [v for _, v in yc["prior"]]
        fig.add_trace(go.Scatter(x=pl, y=pv, mode="lines+markers",
                                 name=f"~1mo prior ({yc.get('prior_date','')})",
                                 line=dict(color="#94a3b8", width=1.5, dash="dot"), marker=dict(size=4)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Yield (%)", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(title_text="Maturity", gridcolor="rgba(120,130,140,0.12)")
    return fig

def home_curve_spreads(yc):
    """Key curve spreads (2s10s, 3m10y) in bps from the latest curve."""
    if not yc or not yc.get("latest"):
        return {}
    d = dict(yc["latest"])
    out = {}
    if "2Y" in d and "10Y" in d:
        out["2s10s"] = round((d["10Y"] - d["2Y"]) * 100)
    if "3M" in d and "10Y" in d:
        out["3m10y"] = round((d["10Y"] - d["3M"]) * 100)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# FED WATCH — meeting-date-weighted (CME methodology) — FIXED v12
# ──────────────────────────────────────────────────────────────────────────────
# 2026 FOMC decision dates (decision day = second day of each meeting).
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16"]

# Current Fed Funds target midpoint.  This CANNOT be reliably inferred from the
# meeting-month ZQ contract because that contract already blends both pre- and
# post-meeting rates (circular dependency). Instead we hardcode it and expose a
# UI control so you can update it whenever the Fed moves.
# Update this line when the Fed changes the target. Unit: % (e.g. 3.625 = 350-375 range midpoint).
FED_CURRENT_MID_DEFAULT = 3.625   # 350-375 bps range as of June 2026

def _fed_current_mid():
    """Return the current Fed Funds target midpoint: session override first, else default."""
    return float(st.session_state.get("fed_current_mid", FED_CURRENT_MID_DEFAULT))

def _next_fomc(today=None):
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    for d in FOMC_2026:
        dd = datetime.strptime(d, "%Y-%m-%d").date()
        if dd >= today:
            return dd
    return None

@st.cache_data(ttl=900)
def home_fed_probabilities(current_mid=None):
    """CME FedWatch methodology: the meeting-month ZQ futures contract settles
    to the month's average daily EFFR. We solve for the implied post-meeting rate
    by weighting pre- and post-meeting days, then convert the expected move vs
    the current target midpoint into probabilities.

    Key fix v12: current_mid is passed explicitly (not inferred from the futures
    price, which is circular when the meeting is in the current month)."""
    try:
        import calendar as _cal
        if current_mid is None:
            current_mid = _fed_current_mid()
        meeting = _next_fomc()
        if meeting is None:
            return {}
        zq = yf.Ticker("ZQ=F")
        h = zq.history(period="10d", interval="1d")
        if h.empty:
            return {}
        front_price = float(h["Close"].iloc[-1])
        front_implied = 100.0 - front_price

        # Intra-month day weighting (standard CME approach):
        # Days at OLD rate = meeting_day - 1 (days 1 to meeting_day-1)
        # Days at NEW rate = N - (meeting_day - 1)  (meeting_day through end of month)
        N = _cal.monthrange(meeting.year, meeting.month)[1]
        days_old = max(0, meeting.day - 1)
        days_new = max(1, N - days_old)

        today = datetime.now(ZoneInfo("America/New_York")).date()
        if meeting.month == today.month and meeting.year == today.year:
            # Meeting is this month: front contract IS the meeting-month contract.
            # Solve: front_implied = (days_old * current_mid + days_new * post) / N
            post_rate = (front_implied * N - days_old * current_mid) / days_new
        else:
            # Meeting is in a future month. The front contract reflects the current
            # effective rate (no upcoming meeting this month to blend). Use it
            # directly as the post-meeting implied (good approximation for near-term).
            post_rate = front_implied

        move = (post_rate - current_mid) * 100.0   # bps; negative = cut

        # Probabilities across ±25 and ±50 bps increments.
        # steps = expected move / 25bp. e.g. -0.8 = market pricing 80% of a 25bp cut.
        # Threshold at ±0.5 steps: below -0.5 is a blend of cut25+cut50; above +0.5 hike25+hike50.
        steps = move / 25.0
        probs = {"cut50": 0, "cut25": 0, "hold": 0, "hike25": 0, "hike50": 0}
        if steps <= -1.5:
            probs["cut50"] = 100
        elif steps <= -0.5:
            frac = (-steps - 0.5) / 1.0          # fraction between 25 and 50bp cut
            probs["cut50"] = round(frac * 100); probs["cut25"] = 100 - probs["cut50"]
        elif steps < 0:
            probs["cut25"] = round(-steps * 100)  # e.g. -0.033 → 3%; -0.8 → 80%
            probs["hold"] = 100 - probs["cut25"]
        elif steps == 0:
            probs["hold"] = 100
        elif steps < 0.5:
            probs["hike25"] = round(steps * 100)  # e.g. 0.4 → 40%
            probs["hold"] = 100 - probs["hike25"]
        elif steps < 1.5:
            frac = (steps - 0.5) / 1.0
            probs["hike50"] = round(frac * 100); probs["hike25"] = 100 - probs["hike50"]
        else:
            probs["hike50"] = 100

        return {"meeting": meeting.strftime("%b %d, %Y"),
                "implied_rate": round(post_rate, 3), "target_mid": round(current_mid, 3),
                "move_bps": round(move, 1), "front_price": round(front_price, 4),
                "p_cut": probs["cut25"], "p_hold": probs["hold"], "p_hike": probs["hike25"],
                "p_cut50": probs["cut50"], "p_hike50": probs["hike50"]}
    except Exception:
        return {}

# ---- Macro snapshot (rates, curve, vol, dollar, commodities) ----
@st.cache_data(ttl=300)
def home_macro_snapshot():
    """A compact set of macro reads for the dashboard + Macro page."""
    out = {}
    syms = {"^TNX": "10Y", "^FVX": "5Y", "^IRX": "13W TBill", "^TYX": "30Y",
            "^VIX": "VIX", "DX-Y.NYB": "DXY", "CL=F": "WTI", "GC=F": "Gold", "HG=F": "Copper"}
    try:
        data = yf.download(list(syms), period="5d", interval="1d", progress=False, auto_adjust=False, group_by="ticker")
        for sym, lab in syms.items():
            try:
                s = data[sym]["Close"].dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"].dropna()
                if len(s) >= 1:
                    v = float(s.iloc[-1])
                    if sym in ("^TNX", "^FVX", "^IRX", "^TYX") and v > 100:
                        v /= 10.0
                    chg = (float(s.iloc[-1]) - float(s.iloc[-2])) if len(s) >= 2 else None
                    if chg is not None and sym in ("^TNX", "^FVX", "^IRX", "^TYX") and abs(chg) > 1:
                        chg /= 10.0
                    out[lab] = {"value": v, "change": chg}
            except Exception:
                pass
    except Exception:
        pass
    return out

# ---- News feed: free RSS from reputable sources, bucketed by theme ----
# Map each source feed to one of our factor / asset-class buckets.
HOME_NEWS_FEEDS = {
    # Factors
    "Fundamentals": [("https://feeds.content.dowjones.io/public/rss/mw_topstories", "MarketWatch"),
                     ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC")],
    "Valuation":    [("https://www.cnbc.com/id/15839135/device/rss/rss.html", "CNBC Markets")],
    "Interest Rates": [("https://www.federalreserve.gov/feeds/press_all.xml", "Federal Reserve"),
                       ("https://home.treasury.gov/system/files/126/rss.xml", "US Treasury")],
    "Policy":       [("https://apnews.com/index.rss", "AP")],
    "Behavioral / Trends": [("https://www.cnbc.com/id/10000664/device/rss/rss.html", "CNBC Econ")],
    # Asset classes
    "US Equities":  [("https://www.cnbc.com/id/15839135/device/rss/rss.html", "CNBC")],
    "International": [("https://www.cnbc.com/id/19794221/device/rss/rss.html", "CNBC World")],
    "Fixed Income": [("https://www.cnbc.com/id/15839069/device/rss/rss.html", "CNBC Bonds")],
    "Commodities / Real Assets": [("https://www.cnbc.com/id/15839074/device/rss/rss.html", "CNBC Commodities")],
}

@st.cache_data(ttl=900)
def home_news(bucket, limit=5):
    """Pull recent headlines for a bucket from its RSS feeds. Returns list of
    (title, source, link). Degrades to [] if feeds are unreachable."""
    feeds = HOME_NEWS_FEEDS.get(bucket, [])
    items = []
    for url, source in feeds:
        try:
            req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _urlreq.urlopen(req, timeout=12) as r:
                raw = r.read()
            root = _ET.fromstring(raw)
            # RSS <item> or Atom <entry>
            for it in root.iter():
                tag = it.tag.lower()
                if tag.endswith("item") or tag.endswith("entry"):
                    title = None; link = None
                    for ch in it:
                        ct = ch.tag.lower()
                        if ct.endswith("title") and title is None:
                            title = (ch.text or "").strip()
                        if ct.endswith("link") and link is None:
                            link = (ch.text or "").strip() or ch.get("href")
                    if title:
                        items.append((title, source, link))
        except Exception:
            continue
    # de-dup by title, keep order
    seen = set(); uniq = []
    for t, s, l in items:
        if t not in seen:
            seen.add(t); uniq.append((t, s, l))
    return uniq[:limit]




# ===== Two-date transactions (added v9) =====
AI_INCEPTION_DATE = "2026-02-10"   # original book established
# AI_REBALANCE_DATE ("2026-06-01") already defined earlier.

# The ORIGINAL 2/10 book (before the 6/1 rebalance): includes BABA, HPE, full NEE;
# excludes the 6/1 adds (PANW, ARM, SO). Names map to (name, tier, target_weight%).
AI_PORTFOLIO_ORIGINAL = [
    ("GOOGL", "Alphabet", "HYPER", 7.0), ("META", "Meta Platforms", "HYPER", 6.0),
    ("AMZN", "Amazon", "HYPER", 5.0), ("MSFT", "Microsoft", "HYPER", 5.0),
    ("ORCL", "Oracle", "HYPER", 4.0), ("BABA", "Alibaba", "HYPER", 3.0),
    ("NVDA", "NVIDIA", "TIER 1", 8.0), ("AVGO", "Broadcom", "TIER 1", 5.0),
    ("TSM", "TSMC", "TIER 1", 4.0), ("AMD", "AMD", "TIER 1", 3.0),
    ("MU", "Micron", "TIER 1", 3.0), ("ANET", "Arista Networks", "TIER 1", 3.0),
    ("DELL", "Dell Technologies", "TIER 1", 3.0), ("MRVL", "Marvell", "TIER 1", 3.0),
    ("VRT", "Vertiv", "TIER 2", 4.0), ("ETN", "Eaton", "TIER 2", 3.0),
    ("ASML", "ASML", "TIER 2", 2.0), ("APH", "Amphenol", "TIER 2", 2.0),
    ("AMAT", "Applied Materials", "TIER 2", 2.0), ("TT", "Trane Technologies", "TIER 2", 2.0),
    ("CSCO", "Cisco", "TIER 2", 2.0), ("HPE", "Hewlett Packard Enterprise", "TIER 2", 2.0),
    ("VST", "Vistra", "TIER 3", 3.0), ("CEG", "Constellation Energy", "TIER 3", 3.0),
    ("GEV", "GE Vernova", "TIER 3", 2.0), ("PWR", "Quanta Services", "TIER 3", 2.0),
    ("CCJ", "Cameco", "TIER 3", 2.0), ("NEE", "NextEra Energy", "TIER 3", 2.0),
    ("CAT", "Caterpillar", "TIER 3", 2.0), ("FSLR", "First Solar", "TIER 3", 2.0),
]

def ai_build_two_date_transactions(notional=AI_PORTFOLIO_NOTIONAL):
    """Two-date transaction model with DRIFT-based rebalancing:
      1) 2/10 inception BUYs (original 30 at the 2/10 close).
      2) 6/1 rebalance = bring every position back to its current target weight,
         accounting for price drift since inception. A name that appreciated to
         above its target (e.g. Dell drifting to ~7.4%) is TRIMMED back; one that
         lagged is ADDED to; dropped names are SOLD; new names are BOUGHT.
    Returns (inception_rows, rebalance_rows, summary)."""
    orig = {tk: {"name": nm, "tier": ti, "w": w} for tk, nm, ti, w in AI_PORTFOLIO_ORIGINAL}
    cur_holdings = ai_load_portfolio()
    cur = {h["ticker"]: {"name": h.get("name", h["ticker"]), "tier": h.get("tier", ""),
                         "w": (h.get("target_weight") or 0)} for h in cur_holdings if (h.get("target_weight") or 0) > 0}

    # ---- 1) Inception buys at 2/10 ----
    px0 = ai_prices_on_date(tuple(orig.keys()), AI_INCEPTION_DATE)
    inception_rows = []
    inv0 = 0.0
    shares0 = {}
    for tk, d in orig.items():
        p = px0.get(tk)
        dollars = notional * (d["w"] / 100.0)
        sh = int(dollars // p) if (p and p > 0) else None
        shares0[tk] = sh or 0
        val = (sh * p) if (sh is not None and p) else None
        if val:
            inv0 += val
        inception_rows.append({"date": AI_INCEPTION_DATE, "action": "BUY", "ticker": tk,
                               "name": d["name"], "tier": d["tier"], "weight": d["w"],
                               "price": p, "shares": sh, "value": val})

    # ---- 2) Value the inception book at 6/1 to get DRIFTED weights ----
    all_tk = sorted(set(orig) | set(cur))
    px1 = ai_prices_on_date(tuple(all_tk), AI_REBALANCE_DATE)
    # portfolio value at 6/1 (original shares at 6/1 prices)
    pv1 = 0.0
    drift_val = {}
    for tk in orig:
        p = px1.get(tk)
        v = (shares0.get(tk, 0) * p) if p else 0.0
        drift_val[tk] = v
        pv1 += v
    if pv1 <= 0:
        pv1 = notional  # fallback so we don't divide by zero

    rebalance_rows = []
    for tk in all_tk:
        ow = orig.get(tk, {}).get("w", 0.0)          # original target
        cw = cur.get(tk, {}).get("w", 0.0)           # current target
        nm = (cur.get(tk) or orig.get(tk) or {}).get("name", tk)
        ti = (cur.get(tk) or orig.get(tk) or {}).get("tier", "")
        p = px1.get(tk)
        drift_w = (drift_val.get(tk, 0.0) / pv1 * 100.0) if tk in orig else 0.0  # weight at 6/1 before rebalance

        # classify
        if ow == 0 and cw > 0:
            action = "BUY (new)"
        elif cw == 0 and ow > 0:
            action = "SELL (exit)"
        else:
            # held in both: compare DRIFTED weight to the current target
            if drift_w - cw > 0.25:          # drifted above target by >0.25pp -> trim
                action = "TRIM"
            elif cw - drift_w > 0.25:         # below target -> add
                action = "ADD"
            else:
                action = None                 # already on target, no trade

        if action is None:
            continue

        # share delta: target dollars at current weight vs current (drifted) dollars
        target_dollars = notional * (cw / 100.0)
        cur_dollars = drift_val.get(tk, 0.0) if tk in orig else 0.0
        dShares = None; dVal = None
        if p and p > 0:
            dShares = int(round((target_dollars - cur_dollars) / p))
            dVal = dShares * p
        rebalance_rows.append({"date": AI_REBALANCE_DATE, "action": action, "ticker": tk,
                               "name": nm, "tier": ti, "from_w": (drift_w if tk in orig else 0.0),
                               "target_w": cw, "orig_w": ow, "price": p,
                               "shares": dShares, "value": dVal})

    summary = {"inception_invested": inv0, "notional": notional, "pv_at_rebalance": pv1,
               "n_inception": len([r for r in inception_rows if r["shares"]]),
               "n_sell": len([r for r in rebalance_rows if r["action"].startswith("SELL")]),
               "n_trim": len([r for r in rebalance_rows if r["action"] == "TRIM"]),
               "n_buy": len([r for r in rebalance_rows if r["action"].startswith("BUY")]),
               "n_add": len([r for r in rebalance_rows if r["action"] == "ADD"])}
    return inception_rows, rebalance_rows, summary


# ===== FRED, sentiment, improved news (added v8) =====
def _fred_key():
    """Read a FRED API key from Streamlit secrets or a session-pasted key.
    Returns '' if none configured."""
    try:
        k = st.secrets.get("FRED_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get("user_fred_key", "")

@st.cache_data(ttl=3600)
def fred_series(series_id, start="2018-01-01", _key=""):
    """Fetch a FRED series, preferring the official API (more reliable from cloud
    IPs) and falling back to the no-key CSV endpoint. Returns [(date,value)]."""
    key = _key or _fred_key()
    if key:
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
                   f"&api_key={key}&file_type=json&observation_start={start}&sort_order=asc")
            req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _urlreq.urlopen(req, timeout=20) as r:
                js = json.loads(r.read())
            out = []
            for o in js.get("observations", []):
                v = o.get("value")
                if v in ("", ".", None):
                    continue
                try:
                    out.append((o["date"], float(v)))
                except Exception:
                    pass
            if out:
                return out
        except Exception:
            pass
    # fallback: CSV endpoint (no key)
    return fred_csv(series_id, start)

@st.cache_data(ttl=3600)
def fred_csv(series_id, start="2018-01-01"):
    """Fetch a FRED series via the public CSV endpoint (no API key required).
    Returns a list of (date_str, float_value) oldest->newest, or [] on failure."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", "replace")
        out = []
        for line in raw.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            d, v = parts[0], parts[1]
            if v in ("", ".", "null"):
                continue
            try:
                out.append((d, float(v)))
            except Exception:
                pass
        return out
    except Exception:
        return []

def _latest_two(series):
    if not series:
        return None, None
    if len(series) == 1:
        return series[-1], None
    return series[-1], series[-2]

def _yoy(series):
    """Year-over-year % change from a monthly index series (latest vs ~12 obs prior)."""
    if not series or len(series) < 13:
        return None
    latest = series[-1][1]
    prior = series[-13][1]
    return (latest / prior - 1) * 100 if prior else None

@st.cache_data(ttl=3600)
def fred_macro_board():
    """Assemble the macro data board: labor, inflation, activity. Each entry is
    {label: {value, units, asof, sub}} computed from FRED CSV series."""
    board = {}

    # ---- Labor ----
    icsa = fred_series("ICSA", "2022-01-01")
    if icsa:
        cur, prev = _latest_two(icsa)
        board["Initial Jobless Claims"] = {
            "value": f"{cur[1]:,.0f}", "asof": cur[0],
            "sub": (f"{(cur[1]-prev[1]):+,.0f} wk/wk" if prev else "weekly")}
    ccsa = fred_series("CCSA", "2022-01-01")
    if ccsa:
        cur, prev = _latest_two(ccsa)
        board["Continuing Claims"] = {
            "value": f"{cur[1]:,.0f}", "asof": cur[0],
            "sub": (f"{(cur[1]-prev[1]):+,.0f} wk/wk" if prev else "weekly")}
    pay = fred_series("PAYEMS", "2022-01-01")
    if pay:
        cur, prev = _latest_two(pay)
        board["Nonfarm Payrolls (MoM)"] = {
            "value": (f"{(cur[1]-prev[1])*1000:+,.0f}" if prev else f"{cur[1]*1000:,.0f}"),
            "asof": cur[0], "sub": "jobs added, monthly"}
    unrate = fred_series("UNRATE", "2022-01-01")
    if unrate:
        cur, prev = _latest_two(unrate)
        board["Unemployment Rate"] = {
            "value": f"{cur[1]:.1f}%", "asof": cur[0],
            "sub": (f"{(cur[1]-prev[1]):+.1f}pp" if prev else "monthly")}
    jolts = fred_series("JTSJOL", "2022-01-01")
    if jolts:
        cur, prev = _latest_two(jolts)
        board["Job Openings (JOLTS)"] = {
            "value": f"{cur[1]*1000:,.0f}", "asof": cur[0],
            "sub": (f"{(cur[1]-prev[1])*1000:+,.0f} m/m" if prev else "monthly")}

    # ---- Inflation (YoY from index) ----
    for sid, lab in [("CPIAUCSL", "CPI (YoY)"), ("CPILFESL", "Core CPI (YoY)"),
                     ("PPIFIS", "PPI Final Demand (YoY)"), ("PCEPI", "PCE (YoY)"),
                     ("PCEPILFE", "Core PCE (YoY)")]:
        s = fred_csv(sid, "2022-01-01")
        y = _yoy(s)
        if y is not None:
            board[lab] = {"value": f"{y:.1f}%", "asof": s[-1][0], "sub": "year-over-year"}

    return board

@st.cache_data(ttl=3600)
def fred_sentiment_board():
    """Sentiment indicators with reliable free data: CNN Fear & Greed (live JSON),
    University of Michigan sentiment (FRED), and OECD business/consumer confidence
    (FRED) as widely-used proxies. AAII and Conference Board's official index are
    not available via a reliable free feed and are intentionally omitted."""
    board = {}
    # CNN Fear & Greed (unofficial JSON)
    try:
        req = _urlreq.Request("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                              headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with _urlreq.urlopen(req, timeout=15) as r:
            js = json.loads(r.read())
        fg = js.get("fear_and_greed", {})
        score = fg.get("score"); rating = fg.get("rating")
        if score is not None:
            board["CNN Fear & Greed"] = {"value": f"{float(score):.0f}",
                                         "sub": (rating or "").title(), "asof": fg.get("timestamp", "")[:10]}
    except Exception:
        pass
    # Michigan consumer sentiment (FRED)
    um = fred_series("UMCSENT", "2022-01-01")
    if um:
        cur, prev = _latest_two(um)
        board["U. Michigan Sentiment"] = {"value": f"{cur[1]:.1f}", "asof": cur[0],
                                          "sub": (f"{(cur[1]-prev[1]):+.1f} m/m" if prev else "monthly")}
    # OECD confidence proxies (FRED)
    bc = fred_series("BSCICP03USM665S", "2022-01-01")
    if bc:
        cur, prev = _latest_two(bc)
        board["Business Confidence (OECD)"] = {"value": f"{cur[1]:.1f}", "asof": cur[0],
                                               "sub": "OECD US, 100=avg"}
    cc = fred_series("CSCICP03USM665S", "2022-01-01")
    if cc:
        cur, prev = _latest_two(cc)
        board["Consumer Confidence (OECD)"] = {"value": f"{cur[1]:.1f}", "asof": cur[0],
                                               "sub": "OECD US, 100=avg"}
    return board

# ---- FRED-based Treasury yield curve (replaces the Treasury-endpoint version) ----
FRED_CURVE = [("DGS1MO","1M"),("DGS3MO","3M"),("DGS6MO","6M"),("DGS1","1Y"),
              ("DGS2","2Y"),("DGS3","3Y"),("DGS5","5Y"),("DGS7","7Y"),
              ("DGS10","10Y"),("DGS20","20Y"),("DGS30","30Y")]

@st.cache_data(ttl=3600)
def fred_yield_curve():
    """Build today's and ~1-month-prior par yield curves from FRED daily series.
    Returns dict matching the old home_yield_curve() shape so the chart helper
    is unchanged."""
    latest = []; prior = []; latest_date = None; prior_date = None
    series_cache = {}
    for sid, lab in FRED_CURVE:
        s = fred_series(sid, "2024-01-01")
        series_cache[lab] = s
        if s:
            latest.append((lab, s[-1][1]))
            if latest_date is None or s[-1][0] > latest_date:
                latest_date = s[-1][0]
    # prior ≈ 21 business days back, per series
    for sid, lab in FRED_CURVE:
        s = series_cache.get(lab) or []
        if len(s) > 22:
            prior.append((lab, s[-22][1]))
            prior_date = s[-22][0]
    if not latest:
        return {}
    return {"date": latest_date, "latest": latest,
            "prior": prior or None, "prior_date": prior_date}

# ============================================================
# IMPROVED NEWS: quality feeds + keyword bucketing + fluff filter
# ============================================================
# Higher-signal feeds. WSJ and Economist publish free RSS headline feeds; the
# article click-through uses the reader's own login for paywalled outlets.
HOME_QUALITY_FEEDS = [
    # WSJ (free RSS headlines; full article via your login)
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ Markets"),
    ("https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "WSJ Business"),
    ("https://feeds.a.dj.com/rss/RSSWorldNews.xml", "WSJ World"),
    ("https://feeds.a.dj.com/rss/RSSWSJD.xml", "WSJ Tech"),
    # Reuters (high signal-to-noise, real-time business/markets coverage)
    ("https://feeds.reuters.com/reuters/businessNews", "Reuters Business"),
    ("https://feeds.reuters.com/reuters/technologyNews", "Reuters Tech"),
    ("https://feeds.reuters.com/reuters/topNews", "Reuters Top"),
    # Yahoo Finance
    ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance"),
    ("https://finance.yahoo.com/rss/topfinstories", "Yahoo Finance"),
    # The Economist
    ("https://www.economist.com/finance-and-economics/rss.xml", "Economist"),
    ("https://www.economist.com/business/rss.xml", "Economist Business"),
    # CNBC
    ("https://www.cnbc.com/id/15839135/device/rss/rss.html", "CNBC Markets"),
    ("https://www.cnbc.com/id/20910258/device/rss/rss.html", "CNBC Economy"),
    # AP
    ("https://apnews.com/index.rss", "AP"),
]

# Patterns that flag advertorial / personal-finance fluff — dropped entirely.
NEWS_FLUFF = [
    "i'm ", "i am ", "my husband", "my wife", "my son", "my daughter", "my mom", "my dad",
    "should i", "how i ", "how to ", "dear ", "suze orman", "ask ", "long-term care",
    "credit card", "best credit", "top 5", "top 10", "5 things", "3 things", "10 things",
    "millionaire", "retire early", "401(k) mistake", "save money", "deal of", "discount",
    "sponsored", "advertisement", "horoscope", "recipe", "gift guide", "prime day",
    "mortgage rate", "personal loan", "savings account", "i bought", "i sold", "i tried",
    "here's how much", "this is how", "could make you rich", "want to retire",
    "crypto portfolio", "how to build a", "best way to", "ad ·", "citi®", "discover®",
    "trillionaire", "celebrity", "elon musk poised",
]

# Keyword sets per bucket. A headline must hit a bucket's keywords to appear there.
NEWS_FACTOR_KW = {
    "Fundamentals": ["earnings", "profit", "revenue", "guidance", "margin", "results",
                     "sales", "outlook", "forecast", "beat estimates", "misses", "quarter", "eps"],
    "Valuation": ["valuation", "p/e", "price target", "rating", "upgrade", "downgrade",
                  "overvalued", "undervalued", "bubble", "expensive", "cheap", "multiple"],
    "Interest Rates": ["fed", "rate", "yield", "treasury", "fomc", "powell", "monetary",
                       "basis point", "rate cut", "rate hike", "central bank", "bond yield"],
    "Policy": ["tariff", "trade war", "regulation", "fiscal", "tax", "congress", "white house",
               "sanction", "antitrust", "legislation", "tradedeal", "trade deal", "stimulus"],
    "Behavioral / Trends": ["rally", "selloff", "sell-off", "volatility", "fear", "greed",
                            "momentum", "record high", "plunge", "surge", "rebound", "sentiment", "rout"],
}
NEWS_ASSET_KW = {
    "US Equities": ["s&p", "nasdaq", "dow", "wall street", "u.s. stocks", "us stocks",
                    "shares", "equities", "stock market"],
    "International": ["china", "europe", "japan", "emerging market", "eurozone", "germany",
                      "india", "uk ", "global stocks", "overseas", "eafe", "asia"],
    "Fixed Income": ["bond", "treasury", "yield", "credit", "debt", "fixed income",
                     "duration", "high-yield", "investment-grade", "coupon", "sovereign"],
    "Commodities / Real Assets": ["oil", "crude", "gold", "copper", "commodity", "energy",
                                  "metal", "natural gas", "silver", "wheat", "real estate", "reit"],
}

import email.utils as _eut

def _parse_news_date(it):
    """Extract a timezone-aware datetime from an RSS <pubDate> (RFC822) or Atom
    <published>/<updated> (ISO8601). Returns datetime or None."""
    for ch in it:
        ct = ch.tag.lower()
        if ct.endswith("pubdate") or ct.endswith("date") or ct.endswith("published") or ct.endswith("updated"):
            txt = (ch.text or "").strip()
            if not txt:
                continue
            # RFC822 (WSJ/CNBC/AP): "Wed, 04 Jun 2026 13:22:00 GMT"
            try:
                dt = _eut.parsedate_to_datetime(txt)
                if dt is not None:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    return dt
            except Exception:
                pass
            # ISO8601 (Atom/Economist): "2026-06-04T13:22:00Z"
            try:
                iso = txt.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                return dt
            except Exception:
                pass
    return None

@st.cache_data(ttl=900)
def home_news_pool(max_age_hours=24):
    """Fetch quality feeds, drop fluff, and keep only items published within
    max_age_hours. Returns [(title, source, link, iso_dt)] sorted newest-first.
    If no items carry parseable dates (feed quirk), falls back to the newest
    available rather than showing nothing, with dates left blank."""
    now = datetime.now(ZoneInfo("UTC"))
    dated = []      # (dt, title, source, link)
    undated = []    # (title, source, link)
    for url, source in HOME_QUALITY_FEEDS:
        try:
            req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _urlreq.urlopen(req, timeout=12) as r:
                root = _ET.fromstring(r.read())
            for it in root.iter():
                tag = it.tag.lower()
                if tag.endswith("item") or tag.endswith("entry"):
                    title = link = None
                    for ch in it:
                        ct = ch.tag.lower()
                        if ct.endswith("title") and title is None:
                            title = (ch.text or "").strip()
                        if ct.endswith("link") and link is None:
                            link = (ch.text or "").strip() or ch.get("href")
                    if not (title and len(title) > 12):
                        continue
                    low = title.lower()
                    if any(f in low for f in NEWS_FLUFF):
                        continue
                    dt = _parse_news_date(it)
                    if dt is not None:
                        dated.append((dt, title, source, link))
                    else:
                        undated.append((title, source, link))
        except Exception:
            continue
    # de-dup by title prefix
    seen = set()
    fresh = []
    for dt, t, s, l in sorted(dated, key=lambda x: x[0], reverse=True):
        k = t.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        age_h = (now - dt).total_seconds() / 3600.0
        if age_h <= max_age_hours:
            fresh.append((t, s, l, dt.astimezone(ZoneInfo("America/New_York")).strftime("%b %d %I:%M%p")))
    if fresh:
        return fresh
    # fallback 1: most recent dated items even if >24h (so the panel isn't empty)
    if dated:
        out = []
        for dt, t, s, l in sorted(dated, key=lambda x: x[0], reverse=True)[:40]:
            k = t.lower()[:60]
            if k in seen and out:
                continue
            seen.add(k)
            out.append((t, s, l, dt.astimezone(ZoneInfo("America/New_York")).strftime("%b %d %I:%M%p")))
        return out
    # fallback 2: undated
    out = []
    for t, s, l in undated:
        k = t.lower()[:60]
        if k not in seen:
            seen.add(k); out.append((t, s, l, ""))
    return out

import urllib.parse as _up

GOOGLE_NEWS_BASE = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="

GOOGLE_QUERIES = {
    # Factors
    "Fundamentals": "earnings results quarterly revenue profit guidance",
    "Valuation":    "stock analyst rating price target upgrade downgrade valuation",
    "Interest Rates": "Federal Reserve interest rate yield treasury bond FOMC",
    "Policy":       "tariff trade policy government regulation fiscal Congress",
    "Behavioral / Trends": "stock market rally selloff volatility investors sentiment",
    # Asset classes
    "US Equities":  "S&P 500 Nasdaq Dow US stock market today",
    "International": "China Europe Japan global stocks emerging markets",
    "Fixed Income": "bond yield treasury fixed income credit high yield",
    "Commodities / Real Assets": "oil gold crude commodity energy metals real estate",
}

def _gn_url(topic):
    q = _up.quote(GOOGLE_QUERIES[topic] + " when:1d")
    return GOOGLE_NEWS_BASE + q

@st.cache_data(ttl=900)
def home_news_google(bucket, limit=5):
    """Fetch headlines from Google News RSS for a specific bucket topic.
    Returns [(title, source, link, when)] sorted newest-first.
    Google News handles the 24h filter via 'when:1d' in the query."""
    if bucket not in GOOGLE_QUERIES:
        return []
    try:
        url = _gn_url(bucket)
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; news-reader/1.0)"})
        with _urlreq.urlopen(req, timeout=15) as r:
            root = _ET.fromstring(r.read())
        items = []
        for it in root.iter():
            if not it.tag.lower().endswith(("item","entry")):
                continue
            title = link = pub = None
            for ch in it:
                ct = ch.tag.lower()
                if ct.endswith("title") and title is None:
                    title = (ch.text or "").strip()
                if ct.endswith("link") and link is None:
                    link = (ch.text or "").strip() or ch.get("href","")
                if (ct.endswith("pubdate") or ct.endswith("published")) and pub is None:
                    pub = ch.text
            if not title or len(title) < 12:
                continue
            low = title.lower()
            if any(f in low for f in NEWS_FLUFF):
                continue
            # Google titles are often "Headline - Source Name"; extract source.
            source = "Google News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title, source = parts[0].strip(), parts[1].strip()
            when = ""
            if pub:
                try:
                    import email.utils as _eu
                    dt = _eu.parsedate_to_datetime(pub.strip())
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    when = dt.astimezone(ZoneInfo("America/New_York")).strftime("%b %d %I:%M%p")
                except Exception:
                    pass
            items.append((title, source, link, when))
        return items[:limit]
    except Exception:
        return []

def home_news_categorized(lens, per_bucket=4, max_age_hours=24):
    """Return dict bucket→[(title,source,link,when)] using Google News as primary
    source, falling back to the pool+keyword approach if Google fails."""
    if lens == "Factors":
        buckets = ["Fundamentals", "Valuation", "Interest Rates", "Policy", "Behavioral / Trends"]
    else:
        buckets = ["US Equities", "International", "Fixed Income", "Commodities / Real Assets"]

    result = {}
    any_google = False
    for b in buckets:
        items = home_news_google(b, limit=per_bucket)
        result[b] = items
        if items:
            any_google = True

    if any_google:
        return result

    # Fallback: pool + keyword matching (broader keywords for better recall)
    pool = home_news_pool(max_age_hours)
    kw_broad = {
        "Fundamentals": ["earnings","profit","revenue","guidance","margin","results",
                         "beat","miss","outlook","quarter","eps","sales","forecast"],
        "Valuation":    ["valuation","p/e","price target","rating","upgrade","downgrade",
                         "overvalued","undervalued","expensive","multiple","bubble"],
        "Interest Rates":["fed","rate","yield","treasury","fomc","powell","monetary",
                          "rate cut","rate hike","central bank","basis point","bond yield"],
        "Policy":       ["tariff","trade","regulation","fiscal","tax","congress","policy",
                         "antitrust","legislation","government","sanction","stimulus"],
        "Behavioral / Trends":["rally","selloff","sell-off","volatility","fear","greed",
                               "momentum","bull","bear","record","high","low","gains",
                               "losses","falls","rises","surge","slide","tumble","drop",
                               "plunge","rebound","soar","stock market"],
        "US Equities":  ["s&p","nasdaq","dow","wall street","stocks","equities","shares",
                         "market","trading","stock market","tech stock","chip"],
        "International":["china","europe","japan","emerging","eurozone","global","india",
                         "overseas","eafe","asia","uk ","germany","geopolit"],
        "Fixed Income": ["bond","treasury","yield","credit","debt","fixed income",
                         "duration","coupon","high-yield","investment-grade","sovereign"],
        "Commodities / Real Assets":["oil","crude","gold","copper","commodity","energy",
                                     "metal","natural gas","silver","wheat","reit"],
    }
    out = {b: [] for b in buckets}
    if lens == "Factors":
        for title, source, link, when in pool:
            low = title.lower()
            for b in buckets:
                if len(out[b]) >= per_bucket: continue
                if any(w in low for w in kw_broad.get(b, [])):
                    out[b].append((title, source, link, when))
    else:
        for title, source, link, when in pool:
            low = title.lower()
            best, best_score = None, 0
            for b in buckets:
                score = sum(1 for w in kw_broad.get(b, []) if w in low)
                if score > best_score:
                    best, best_score = b, score
            if best and len(out[best]) < per_bucket:
                out[best].append((title, source, link, when))
    return out

# ===== Home market snapshot chart (added v9) =====
@st.cache_data(ttl=600)
def home_market_chart(symbol="SPY", period="6mo"):
    """Snapshot market chart data: closes + a 50-day moving average. Returns
    (dates, closes, ma50, last, chg_pct) or None. Less dynamic than Chart Analysis."""
    try:
        h = yf.Ticker(symbol).history(period=period, interval="1d")
        if h.empty or len(h) < 5:
            return None
        closes = h["Close"]
        ma50 = closes.rolling(50).mean()
        last = float(closes.iloc[-1])
        first = float(closes.iloc[0])
        chg = (last / first - 1) * 100 if first else None
        return {"dates": list(h.index), "closes": [float(x) for x in closes],
                "ma50": [None if pd.isna(x) else float(x) for x in ma50],
                "last": last, "chg_pct": chg, "symbol": symbol, "period": period}
    except Exception:
        return None

def home_market_chart_figure(mc):
    if not mc:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc["dates"], y=mc["closes"], mode="lines", name=mc["symbol"],
                             line=dict(color="#2dd4bf", width=2)))
    if any(v is not None for v in mc["ma50"]):
        fig.add_trace(go.Scatter(x=mc["dates"], y=mc["ma50"], mode="lines", name="50-day MA",
                                 line=dict(color="#94a3b8", width=1, dash="dot")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    fig.update_yaxes(gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)")
    return fig


# ===== Treasury yield curve data (embedded 1990-2025 + live current-year fetch, added v14) =====
TREASURY_HIST_B64 = (
    "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"
)

# ============================================================
# TREASURY YIELD CURVE DATA MODULE (added v14)
# Direct from the U.S. Treasury (Daily Par Yield Curve Rates), NOT FRED and NOT
# a third-party site. 1990-2025 is embedded (from the user's own archive +
# Treasury's official CSV export), so historical data works with zero network
# calls. The current year is fetched live from Treasury's own CSV export
# (no API key) and cached for 12 hours. A manual-upload fallback is provided
# in case the live fetch is ever blocked from this host.
# ============================================================
TREASURY_MATURITIES = ["1 Mo", "1.5 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr",
                       "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]

@st.cache_data(ttl=None, show_spinner=False)
def treasury_load_historical():
    """Decode the embedded 1990-2025 par yield curve archive. No network call."""
    raw = _gzip_std.decompress(_base64_std.b64decode(TREASURY_HIST_B64))
    df = pd.read_csv(_io_std.BytesIO(raw), parse_dates=["Date"])
    return df

@st.cache_data(ttl=43200, show_spinner=False)   # 12 hours — daily data, no need to refetch often
def treasury_fetch_year(year):
    """Fetch one year of the official Daily Treasury Par Yield Curve Rates
    directly from home.treasury.gov (the same dataset your colleague's tool used,
    just the correct 'par yield curve' view rather than 'bill rates'). No API key."""
    url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
           f"&field_tdr_date_value={year}&page=&_format=csv")
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=20) as r:
            raw = r.read()
        df = pd.read_csv(_io_std.BytesIO(raw))
        df.columns = [c.strip().strip('"') for c in df.columns]
        if "1.5 Month" in df.columns:
            df = df.rename(columns={"1.5 Month": "1.5 Mo"})
        for c in TREASURY_MATURITIES:
            if c not in df.columns:
                df[c] = None
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        return df[["Date"] + TREASURY_MATURITIES]
    except Exception:
        return pd.DataFrame(columns=["Date"] + TREASURY_MATURITIES)

def treasury_parse_uploaded_csv(uploaded_file):
    """Parse a user-uploaded CSV in Treasury's own export format (Date + maturity
    columns) into the canonical shape. Used as a fallback if the live fetch is
    ever blocked from this host."""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().strip('"') for c in df.columns]
        if "1.5 Month" in df.columns:
            df = df.rename(columns={"1.5 Month": "1.5 Mo"})
        for c in TREASURY_MATURITIES:
            if c not in df.columns:
                df[c] = None
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        return df[["Date"] + TREASURY_MATURITIES]
    except Exception:
        return None

def treasury_full_data():
    """Historical (embedded, 1990-2025) + live current-year fetch + any
    user-uploaded override, merged and deduplicated by date."""
    parts = [treasury_load_historical()]
    cur_year = datetime.now(ZoneInfo("America/New_York")).year
    live = treasury_fetch_year(cur_year)
    if not live.empty:
        parts.append(live)
    if cur_year > 2025:
        # in case a whole extra year has passed since this was built and the
        # embedded archive doesn't cover it — try the prior year too, harmless if empty
        prior_live = treasury_fetch_year(cur_year - 1)
        if not prior_live.empty:
            parts.append(prior_live)
    up = st.session_state.get("treasury_uploaded_df")
    if up is not None and not up.empty:
        parts.append(up)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)
    return combined

def treasury_nearest_on_or_before(df, target_date):
    """Row for the latest available date on or before target_date, or None."""
    sub = df[df["Date"] <= pd.Timestamp(target_date)]
    if sub.empty:
        return None
    return sub.iloc[-1]

def treasury_latest_curve_dict(df=None):
    """Adapter matching the shape the existing compact-chart helper
    (home_yield_curve_figure/home_curve_spreads) expects, so those render
    functions don't need to change. {'date','latest','prior','prior_date'}"""
    df = df if df is not None else treasury_full_data()
    if df.empty:
        return {}
    latest_row = df.iloc[-1]
    latest_date = latest_row["Date"]
    latest = [(m, float(latest_row[m])) for m in TREASURY_MATURITIES if pd.notna(latest_row[m])]
    prior_row = treasury_nearest_on_or_before(df[df["Date"] < latest_date], latest_date - pd.Timedelta(days=30))
    prior = None
    prior_date = None
    if prior_row is not None:
        prior = [(m, float(prior_row[m])) for m in TREASURY_MATURITIES if pd.notna(prior_row[m])]
        prior_date = prior_row["Date"].strftime("%Y-%m-%d")
    return {"date": latest_date.strftime("%Y-%m-%d"), "latest": latest,
            "prior": prior, "prior_date": prior_date}

def treasury_curve_comparison_figure(df, compare_dates):
    """compare_dates: list of (label, target_date). Draws one curve per entry,
    using the nearest available data on or before that date."""
    fig = go.Figure()
    colors = ["#2dd4bf", "#f59e0b", "#a78bfa", "#f87171", "#60a5fa", "#34d399", "#f472b6", "#94a3b8"]
    any_trace = False
    for i, (label, d) in enumerate(compare_dates):
        row = treasury_nearest_on_or_before(df, d)
        if row is None:
            continue
        xs, ys = [], []
        for m in TREASURY_MATURITIES:
            v = row[m]
            if pd.notna(v):
                xs.append(m); ys.append(float(v))
        if not xs:
            continue
        any_trace = True
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                 name=f"{label} ({row['Date'].strftime('%Y-%m-%d')})",
                                 line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=5)))
    if not any_trace:
        return None
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.16, x=0),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Yield (%)", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(title_text="Maturity", gridcolor="rgba(120,130,140,0.12)",
                     categoryorder="array", categoryarray=TREASURY_MATURITIES)
    return fig

def treasury_maturity_history_figure(df, maturity, start_date, end_date):
    sub = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    sub = sub.dropna(subset=[maturity])
    if sub.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub[maturity], mode="lines",
                             name=maturity, line=dict(color="#2dd4bf", width=1.5)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), showlegend=False, hovermode="x unified")
    fig.update_yaxes(title_text="Yield (%)", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(title_text="", gridcolor="rgba(120,130,140,0.12)")
    return fig

def treasury_spread_series(df, short_m, long_m):
    """Spread in basis points: long maturity yield minus short maturity yield.
    Uses explicit column extraction (not fancy [["Date",short_m,long_m]] indexing)
    so it stays correct even if short_m and long_m happen to be the same maturity
    (which would otherwise create a duplicate-named column and break the subtraction)."""
    s = df[["Date"]].copy()
    s["_short"] = df[short_m]
    s["_long"] = df[long_m]
    s = s.dropna(subset=["_short", "_long"])
    s["spread_bps"] = (s["_long"] - s["_short"]) * 100.0
    return s[["Date", "spread_bps"]]

def treasury_spread_figure(df, spread1, spread2=None, start_date=None, end_date=None):
    fig = go.Figure()
    any_trace = False

    def add(spec, color):
        nonlocal any_trace
        s = treasury_spread_series(df, spec["short"], spec["long"])
        if start_date:
            s = s[s["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            s = s[s["Date"] <= pd.Timestamp(end_date)]
        if s.empty:
            return
        any_trace = True
        fig.add_trace(go.Scatter(x=s["Date"], y=s["spread_bps"], mode="lines",
                                 name=spec["label"], line=dict(color=color, width=1.5)))

    add(spread1, "#2dd4bf")
    if spread2:
        add(spread2, "#f59e0b")
    if not any_trace:
        return None
    fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Spread (bps)", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(title_text="", gridcolor="rgba(120,130,140,0.12)")
    return fig

# ===== BLS labor/inflation data module (added v15) =====
BLS_SERIES = {
    "CPI (YoY)":               ("CUUR0000SA0", "yoy_pct"),
    "Core CPI (YoY)":          ("CUUR0000SA0L1E", "yoy_pct"),
    "PPI Final Demand (YoY)":  ("WPSFD4", "yoy_pct"),
    "Nonfarm Payrolls (MoM)":  ("CES0000000001", "mom_payrolls"),
    "Unemployment Rate":       ("LNS14000000", "level_pct_mom"),
    "Job Openings (JOLTS)":    ("JTS000000000000000JOL", "mom_level_x1000"),
    "Avg Hourly Earnings (YoY)": ("CES0500000003", "yoy_pct"),
}

def _bls_key():
    """Read a BLS API key from Streamlit secrets first, else a session-pasted key.
    Returns '' if none configured — the API still works without one, just at
    the lower unregistered rate limit (25 queries/day, 10-year lookback)."""
    try:
        k = st.secrets.get("BLS_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get("user_bls_key", "")

@st.cache_data(ttl=21600, show_spinner=False)   # 6 hours — BLS series update monthly
def bls_fetch_series(series_ids, start_year=None, end_year=None, _key=""):
    """POST to the BLS Public API v2 for a batch of series. Returns
    {series_id: [{'year':int,'month':int,'value':float}, ...]} sorted oldest→newest,
    monthly observations only (annual-average M13 rows are dropped)."""
    key = _key or _bls_key()
    end_year = end_year or datetime.now().year
    start_year = start_year or (end_year - 3)
    try:
        body = {"seriesid": list(series_ids), "startyear": str(start_year), "endyear": str(end_year)}
        if key:
            body["registrationkey"] = key
        data = json.dumps(body).encode("utf-8")
        req = _urlreq.Request("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                              data=data, headers={"Content-Type": "application/json"})
        with _urlreq.urlopen(req, timeout=20) as r:
            js = json.loads(r.read())
        out = {}
        if js.get("status") not in ("REQUEST_SUCCEEDED",):
            return out
        for s in js.get("Results", {}).get("series", []):
            sid = s.get("seriesID")
            rows = []
            for d in s.get("data", []):
                per = d.get("period", "")
                if not (per.startswith("M") and per != "M13"):
                    continue
                try:
                    rows.append({"year": int(d["year"]), "month": int(per[1:]), "value": float(d["value"])})
                except Exception:
                    pass
            rows.sort(key=lambda r: (r["year"], r["month"]))
            out[sid] = rows
        return out
    except Exception:
        return {}

def _bls_yoy(rows):
    """Year-over-year % change: latest month vs the same month one year prior."""
    if len(rows) < 13:
        return None, None
    latest = rows[-1]
    yr_ago = next((r for r in rows if r["year"] == latest["year"] - 1 and r["month"] == latest["month"]), None)
    if not yr_ago or yr_ago["value"] == 0:
        return None, None
    pct = (latest["value"] / yr_ago["value"] - 1) * 100
    return pct, f"{latest['year']}-{latest['month']:02d}"

def _bls_latest_prev(rows):
    if not rows:
        return None, None, None
    latest = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    return latest["value"], (prev["value"] if prev else None), f"{latest['year']}-{latest['month']:02d}"

@st.cache_data(ttl=21600, show_spinner=False)
def bls_labor_inflation_board():
    """Assemble the labor + inflation board entirely from BLS. Same output
    shape as the old fred_macro_board() so the UI rendering code barely changes."""
    ids = tuple(v[0] for v in BLS_SERIES.values())
    raw = bls_fetch_series(ids)
    board = {}
    for label, (sid, kind) in BLS_SERIES.items():
        rows = raw.get(sid, [])
        if not rows:
            continue
        if kind == "yoy_pct":
            pct, asof = _bls_yoy(rows)
            if pct is not None:
                board[label] = {"value": f"{pct:.1f}%", "asof": asof, "sub": "year-over-year"}
        elif kind == "mom_payrolls":
            cur, prev, asof = _bls_latest_prev(rows)
            if cur is not None:
                val_txt = f"{(cur - prev) * 1000:+,.0f}" if prev is not None else f"{cur * 1000:,.0f}"
                board[label] = {"value": val_txt, "asof": asof, "sub": "jobs added, monthly"}
        elif kind == "mom_level_x1000":
            cur, prev, asof = _bls_latest_prev(rows)
            if cur is not None:
                sub = f"{(cur - prev) * 1000:+,.0f} m/m" if prev is not None else "monthly"
                board[label] = {"value": f"{cur * 1000:,.0f}", "asof": asof, "sub": sub}
        elif kind == "level_pct_mom":
            cur, prev, asof = _bls_latest_prev(rows)
            if cur is not None:
                sub = f"{(cur - prev):+.1f}pp" if prev is not None else "monthly"
                board[label] = {"value": f"{cur:.1f}%", "asof": asof, "sub": sub}
    return board


# ===== Stock research module: overview/financials/ratios/peers/links (added v15) =====
def _fmt_num(v, kind="num", decimals=2):
    """Format a numeric .info value for display, or 'n/a' if missing."""
    if v is None:
        return "n/a"
    try:
        if kind == "pct":
            return f"{v*100:.{decimals}f}%"
        if kind == "pct_raw":       # already in percent units
            return f"{v:.{decimals}f}%"
        if kind == "usd":
            return f"${v:,.{decimals}f}"
        if kind == "big":
            av = abs(v)
            if av >= 1e12: return f"${v/1e12:.2f}T"
            if av >= 1e9:  return f"${v/1e9:.2f}B"
            if av >= 1e6:  return f"${v/1e6:.1f}M"
            return f"${v:,.0f}"
        if kind == "int":
            return f"{v:,.0f}"
        return f"{v:.{decimals}f}"
    except Exception:
        return "n/a"

@st.cache_data(ttl=1800, show_spinner=False)
def stock_overview_info(ticker):
    """Raw .info dict for a ticker, cached. Returns {} on failure."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

def stock_overview_grid(info):
    """Organize .info into a finviz-style categorized grid.
    Returns dict of category -> [(label, formatted_value), ...]."""
    g = info.get
    grid = {
        "Company": [
            ("Sector", g("sector") or "n/a"), ("Industry", g("industry") or "n/a"),
            ("Country", g("country") or "n/a"), ("Employees", _fmt_num(g("fullTimeEmployees"), "int")),
        ],
        "Valuation": [
            ("Market Cap", _fmt_num(g("marketCap"), "big")), ("Enterprise Value", _fmt_num(g("enterpriseValue"), "big")),
            ("P/E (trailing)", _fmt_num(g("trailingPE"))), ("P/E (forward)", _fmt_num(g("forwardPE"))),
            ("PEG Ratio", _fmt_num(g("trailingPegRatio") or g("pegRatio"))),
            ("P/S (ttm)", _fmt_num(g("priceToSalesTrailing12Months"))), ("P/B", _fmt_num(g("priceToBook"))),
            ("EV/Revenue", _fmt_num(g("enterpriseToRevenue"))), ("EV/EBITDA", _fmt_num(g("enterpriseToEbitda"))),
        ],
        "Profitability": [
            ("Gross Margin", _fmt_num(g("grossMargins"), "pct")), ("Operating Margin", _fmt_num(g("operatingMargins"), "pct")),
            ("Net Margin", _fmt_num(g("profitMargins"), "pct")), ("ROA", _fmt_num(g("returnOnAssets"), "pct")),
            ("ROE", _fmt_num(g("returnOnEquity"), "pct")),
        ],
        "Growth": [
            ("Revenue Growth (YoY)", _fmt_num(g("revenueGrowth"), "pct")),
            ("Earnings Growth (YoY)", _fmt_num(g("earningsGrowth"), "pct")),
            ("Qtrly Earnings Growth", _fmt_num(g("earningsQuarterlyGrowth"), "pct")),
        ],
        "Per Share": [
            ("EPS (ttm)", _fmt_num(g("trailingEps"), "usd")), ("EPS (fwd)", _fmt_num(g("forwardEps"), "usd")),
            ("Book Value/Share", _fmt_num(g("bookValue"), "usd")), ("Cash/Share", _fmt_num(g("totalCashPerShare"), "usd")),
            ("Revenue/Share", _fmt_num(g("revenuePerShare"), "usd")),
        ],
        "Dividends": [
            ("Div Yield", _fmt_num(g("dividendYield"), "pct_raw") if g("dividendYield") and g("dividendYield") > 1 else _fmt_num(g("dividendYield"), "pct")),
            ("Div Rate", _fmt_num(g("dividendRate"), "usd")), ("Payout Ratio", _fmt_num(g("payoutRatio"), "pct")),
            ("5Y Avg Div Yield", _fmt_num(g("fiveYearAvgDividendYield"), "pct_raw")),
        ],
        "Trading": [
            ("52W High", _fmt_num(g("fiftyTwoWeekHigh"), "usd")), ("52W Low", _fmt_num(g("fiftyTwoWeekLow"), "usd")),
            ("50-Day Avg", _fmt_num(g("fiftyDayAverage"), "usd")), ("200-Day Avg", _fmt_num(g("twoHundredDayAverage"), "usd")),
            ("Beta", _fmt_num(g("beta"))), ("Avg Volume", _fmt_num(g("averageVolume"), "int")),
            ("Shares Out.", _fmt_num(g("sharesOutstanding"), "big")), ("Short % of Float", _fmt_num(g("shortPercentOfFloat"), "pct")),
        ],
        "Ownership": [
            ("Insider Held", _fmt_num(g("heldPercentInsiders"), "pct")), ("Institutional Held", _fmt_num(g("heldPercentInstitutions"), "pct")),
        ],
        "Analyst": [
            ("Recommendation", (g("recommendationKey") or "n/a").replace("_", " ").title()),
            ("Target Mean", _fmt_num(g("targetMeanPrice"), "usd")), ("Target High", _fmt_num(g("targetHighPrice"), "usd")),
            ("Target Low", _fmt_num(g("targetLowPrice"), "usd")), ("# Analysts", _fmt_num(g("numberOfAnalystOpinions"), "int")),
        ],
    }
    return grid

@st.cache_data(ttl=1800, show_spinner=False)
def stock_financial_statements(ticker, statement, period):
    """statement: 'income'|'balance'|'cashflow'; period: 'annual'|'quarterly'.
    Returns a DataFrame (line items x period-end dates) or empty DataFrame."""
    try:
        t = yf.Ticker(ticker)
        if statement == "income":
            df = t.quarterly_financials if period == "quarterly" else t.financials
        elif statement == "balance":
            df = t.quarterly_balance_sheet if period == "quarterly" else t.balance_sheet
        else:
            df = t.quarterly_cashflow if period == "quarterly" else t.cashflow
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _find_row(df, candidates):
    """Find the first matching row (case-insensitive, substring-tolerant) among
    several possible yfinance label variants, and return its most recent value.
    yfinance's exact row naming has drifted across versions, so this is
    deliberately fuzzy rather than requiring an exact key match."""
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for cand in candidates:
        cl = cand.lower()
        if cl in idx_lower:
            try:
                return float(df.loc[idx_lower[cl]].iloc[0])
            except Exception:
                continue
        for k_low, k_orig in idx_lower.items():
            if cl in k_low:
                try:
                    return float(df.loc[k_orig].iloc[0])
                except Exception:
                    continue
    return None

def stock_compute_ratios(ticker, info, income_df, balance_df, cashflow_df):
    """Ratios not directly in .info, derived from the financial statements.
    Every lookup is defensive (returns None -> 'n/a' downstream) since exact
    yfinance row-label wording varies by version."""
    revenue = _find_row(income_df, ["Total Revenue", "TotalRevenue"])
    ebit = _find_row(income_df, ["EBIT", "Operating Income"])
    interest_exp = _find_row(income_df, ["Interest Expense", "InterestExpense"])
    net_income = _find_row(income_df, ["Net Income", "NetIncome"])

    cur_assets = _find_row(balance_df, ["Total Current Assets", "Current Assets"])
    cur_liab = _find_row(balance_df, ["Total Current Liabilities", "Current Liabilities"])
    inventory = _find_row(balance_df, ["Inventory"])
    total_debt = _find_row(balance_df, ["Total Debt"])
    equity = _find_row(balance_df, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])

    op_cf = _find_row(cashflow_df, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
    capex = _find_row(cashflow_df, ["Capital Expenditure", "CapitalExpenditures"])

    ratios = {}
    if cur_assets and cur_liab:
        ratios["Current Ratio"] = f"{cur_assets/cur_liab:.2f}"
    if cur_assets is not None and inventory is not None and cur_liab:
        ratios["Quick Ratio"] = f"{(cur_assets - inventory)/cur_liab:.2f}"
    if total_debt is not None and equity:
        ratios["Debt / Equity"] = f"{total_debt/equity:.2f}"
    if ebit is not None and interest_exp:
        try:
            ratios["Interest Coverage"] = f"{ebit/abs(interest_exp):.1f}x"
        except Exception:
            pass
    if op_cf is not None and capex is not None:
        fcf = op_cf - abs(capex)
        ratios["Free Cash Flow"] = _fmt_num(fcf, "big")
        if revenue:
            ratios["FCF Margin"] = f"{fcf/revenue*100:.1f}%"
    if net_income is not None and equity:
        ratios["ROE (computed)"] = f"{net_income/equity*100:.1f}%"
    # prefer info's own margins/ROE if statement-derived ones are unavailable
    if "ROE (computed)" not in ratios and info.get("returnOnEquity") is not None:
        ratios["ROE (computed)"] = f"{info['returnOnEquity']*100:.1f}%"
    return ratios

def stock_research_links(ticker):
    """SEC EDGAR + IR + Yahoo research links, matching the Earnings Monitor pattern."""
    edgar_full = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}&type=&dateb=&owner=include&count=20"
    edgar_10k = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}&type=10-K&dateb=&owner=include&count=10"
    yf_estimates = f"https://finance.yahoo.com/quote/{ticker}/analysis"
    yf_press = f"https://finance.yahoo.com/quote/{ticker}/press-releases"
    yf_financials = f"https://finance.yahoo.com/quote/{ticker}/financials"
    ir_search = f"https://www.google.com/search?q={ticker}+investor+relations"
    return {"edgar_full": edgar_full, "edgar_10k": edgar_10k, "yf_estimates": yf_estimates,
            "yf_press": yf_press, "yf_financials": yf_financials, "ir_search": ir_search}

PEER_METRICS = [
    ("Market Cap", "marketCap", "big"), ("P/E (fwd)", "forwardPE", "num"),
    ("PEG", "trailingPegRatio", "num"), ("P/S", "priceToSalesTrailing12Months", "num"),
    ("EV/EBITDA", "enterpriseToEbitda", "num"), ("Gross Margin", "grossMargins", "pct"),
    ("Operating Margin", "operatingMargins", "pct"), ("ROE", "returnOnEquity", "pct"),
    ("Revenue Growth", "revenueGrowth", "pct"), ("Div Yield", "dividendYield", "pct"),
    ("Beta", "beta", "num"),
]

def stock_peer_comparison(tickers):
    """Build a comparison table: rows = metrics, columns = tickers.
    Returns a DataFrame, or empty DataFrame if no tickers resolve."""
    infos = {}
    for tk in tickers:
        i = stock_overview_info(tk)
        if i:
            infos[tk] = i
    if not infos:
        return pd.DataFrame()
    rows = []
    for label, field, kind in PEER_METRICS:
        row = {"Metric": label}
        for tk, i in infos.items():
            row[tk] = _fmt_num(i.get(field), kind)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Metric")


# ===== BLS trend charts (added v16) =====
@st.cache_data(ttl=21600, show_spinner=False)
def bls_fetch_history(series_ids, years_back=11):
    """Fetch a longer history for trend charts. Cached separately (longer
    lookback) from the latest-point board fetch."""
    end_year = datetime.now().year
    start_year = end_year - years_back
    return bls_fetch_series(tuple(series_ids), start_year=start_year, end_year=end_year)

def _bls_row_date(r):
    return pd.Timestamp(year=r["year"], month=r["month"], day=1)

def bls_yoy_full_series(rows):
    """YoY % change at every month that has a same-month year-ago comparison.
    Returns a list of (Timestamp, pct)."""
    by_key = {(r["year"], r["month"]): r["value"] for r in rows}
    out = []
    for r in rows:
        prior = by_key.get((r["year"] - 1, r["month"]))
        if prior and prior != 0:
            out.append((_bls_row_date(r), (r["value"] / prior - 1) * 100))
    return out

def bls_level_full_series(rows):
    return [(_bls_row_date(r), r["value"]) for r in rows]

def bls_mom_change_series(rows, scale=1000):
    out = []
    for i in range(1, len(rows)):
        out.append((_bls_row_date(rows[i]), (rows[i]["value"] - rows[i-1]["value"]) * scale))
    return out

def _trim_to_window(series, months_back):
    if not series or months_back is None:
        return series
    cutoff = series[-1][0] - pd.DateOffset(months=months_back)
    return [(d, v) for d, v in series if d >= cutoff]

def bls_inflation_trend_figure(raw, months_back=60):
    """CPI / Core CPI / PPI YoY overlaid on one chart."""
    cpi = _trim_to_window(bls_yoy_full_series(raw.get("CUUR0000SA0", [])), months_back)
    core = _trim_to_window(bls_yoy_full_series(raw.get("CUUR0000SA0L1E", [])), months_back)
    ppi = _trim_to_window(bls_yoy_full_series(raw.get("WPSFD4", [])), months_back)
    fig = go.Figure()
    any_trace = False
    for series, name, color in [(cpi, "CPI (YoY)", "#2dd4bf"), (core, "Core CPI (YoY)", "#f59e0b"), (ppi, "PPI (YoY)", "#a78bfa")]:
        if series:
            any_trace = True
            xs = [d for d, v in series]; ys = [v for d, v in series]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(width=2, color=color)))
    if not any_trace:
        return None
    fig.add_hline(y=2.0, line_dash="dot", line_color="#6b7280", annotation_text="Fed 2% target", annotation_position="bottom right")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
    fig.update_yaxes(title_text="YoY % change", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)")
    return fig

def bls_labor_trend_figure(raw, months_back=60):
    """Unemployment rate (left axis) + JOLTS openings level (right axis)."""
    unemp = _trim_to_window(bls_level_full_series(raw.get("LNS14000000", [])), months_back)
    jolts = _trim_to_window(bls_level_full_series(raw.get("JTS000000000000000JOL", [])), months_back)
    if not unemp and not jolts:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if unemp:
        fig.add_trace(go.Scatter(x=[d for d,v in unemp], y=[v for d,v in unemp], mode="lines",
                                 name="Unemployment Rate (%)", line=dict(width=2, color="#2dd4bf")), secondary_y=False)
    if jolts:
        fig.add_trace(go.Scatter(x=[d for d,v in jolts], y=[v*1000 for d,v in jolts], mode="lines",
                                 name="JOLTS Openings", line=dict(width=2, color="#f59e0b", dash="dot")), secondary_y=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
    fig.update_yaxes(title_text="Unemployment Rate (%)", gridcolor="rgba(120,130,140,0.12)", secondary_y=False)
    fig.update_yaxes(title_text="Job Openings", secondary_y=True)
    fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)")
    return fig

def bls_payrolls_trend_figure(raw, months_back=36):
    """Monthly nonfarm payroll change, bar chart, color-coded pos/neg."""
    series = _trim_to_window(bls_mom_change_series(raw.get("CES0000000001", [])), months_back)
    if not series:
        return None
    xs = [d for d, v in series]; ys = [v for d, v in series]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in ys]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=ys, marker_color=colors, name="Jobs added"))
    fig.add_hline(y=0, line_color="#6b7280")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), showlegend=False)
    fig.update_yaxes(title_text="Jobs added (monthly)", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)")
    return fig

def bls_wages_trend_figure(raw, months_back=60):
    series = _trim_to_window(bls_yoy_full_series(raw.get("CES0500000003", [])), months_back)
    if not series:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d for d,v in series], y=[v for d,v in series], mode="lines",
                             name="Avg Hourly Earnings (YoY)", line=dict(width=2, color="#2dd4bf")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e0"), showlegend=False, hovermode="x unified")
    fig.update_yaxes(title_text="YoY % change", gridcolor="rgba(120,130,140,0.12)")
    fig.update_xaxes(gridcolor="rgba(120,130,140,0.12)")
    return fig


# ===== Actual NAV/drift + mandate alerts (added v16) =====
@st.cache_data(ttl=300, show_spinner=False)
def ai_nav_shares_and_weights(holdings_key):
    """holdings_key: tuple of (ticker, target_weight) for CURRENT holdings with
    weight > 0 (same convention as ai_equity_curve's cache key, so it busts
    correctly when Manage-tab weights change).

    Returns a dict with: shares_post_rebal, cash_residual, pv_at_rebalance,
    price_now, value_now, nav_now, weight_now (actual % of NAV per ticker),
    total_return_pct (true return since 2/10), asof."""
    orig = {tk: w for tk, nm, ti, w in AI_PORTFOLIO_ORIGINAL}
    cur = dict(holdings_key)
    notional = AI_PORTFOLIO_NOTIONAL

    # Step 1: shares bought 2/10 at the ORIGINAL model weights.
    px0 = ai_prices_on_date(tuple(orig.keys()), AI_INCEPTION_DATE)
    shares0 = {}
    for tk, w in orig.items():
        p = px0.get(tk)
        shares0[tk] = int((notional * w / 100.0) // p) if (p and p > 0) else 0

    # Step 2: value those ORIGINAL shares at 6/1 prices — this is the actual
    # portfolio value the rebalance traded from (not a fresh $100k).
    all_tk_reb = sorted(set(orig) | set(cur))
    px1 = ai_prices_on_date(tuple(all_tk_reb), AI_REBALANCE_DATE)
    pv_at_rebal = sum(shares0.get(tk, 0) * (px1.get(tk) or 0) for tk in orig)
    if pv_at_rebal <= 0:
        pv_at_rebal = notional  # fallback if price data is unavailable

    # Step 3: re-buy to the CURRENT target weights using the actual 6/1 value.
    shares_post = {}
    invested_at_rebal = 0.0
    for tk, w in cur.items():
        p = px1.get(tk)
        sh = int((pv_at_rebal * w / 100.0) // p) if (p and p > 0) else 0
        shares_post[tk] = sh
        invested_at_rebal += sh * (p or 0)
    cash_residual = pv_at_rebal - invested_at_rebal   # assumed non-interest-bearing

    # Step 4: value TODAY at live prices — these shares are held flat from 6/1,
    # so today's weight reflects real market drift, not the target weight.
    today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    px_now = ai_prices_on_date(tuple(shares_post.keys()), today_str)
    value_now = {}
    for tk, sh in shares_post.items():
        p = px_now.get(tk)
        value_now[tk] = (sh * p) if p else None

    known_val = sum(v for v in value_now.values() if v is not None)
    nav_now = cash_residual + known_val

    weight_now = {}
    for tk, v in value_now.items():
        weight_now[tk] = (v / nav_now * 100.0) if (v is not None and nav_now > 0) else None

    total_return_pct = (nav_now / notional - 1) * 100.0 if notional > 0 else None

    return {"shares_post_rebal": shares_post, "cash_residual": cash_residual,
            "pv_at_rebalance": pv_at_rebal, "price_now": px_now, "value_now": value_now,
            "nav_now": nav_now, "weight_now": weight_now,
            "total_return_pct": total_return_pct, "cash_pct": (cash_residual/nav_now*100 if nav_now else None),
            "asof": today_str}

# ---------- Mandate guardrails ----------
AI_MANDATE_DEFAULTS = {
    "max_single_name_pct": 10.0,   # hard cap: no position should exceed this % of NAV
    "max_drift_pp": 2.5,           # flag if |current - target| exceeds this many points
    "tier_band_pp": 5.0,           # tier-level actual vs target tolerance
    "max_cash_pct": 15.0,          # flag if too much of the book sits in un-invested cash
}

def ai_mandate_alerts(nav, holdings, thresholds=None):
    """Check the actual (drift) NAV weights against configurable mandate
    guardrails. Returns a list of {level:'error'|'warning', message:str}."""
    th = {**AI_MANDATE_DEFAULTS, **(thresholds or {})}
    weight_now = nav.get("weight_now", {})
    targets = {h["ticker"]: (h.get("target_weight") or 0) for h in holdings}
    tiers = {h["ticker"]: h.get("tier") for h in holdings}
    alerts = []

    for tk, w in weight_now.items():
        if w is None:
            continue
        if w > th["max_single_name_pct"]:
            alerts.append({"level": "error",
                           "message": f"**{tk}** is {w:.1f}% of NAV — exceeds the {th['max_single_name_pct']:.0f}% single-name cap."})

    for tk, w in weight_now.items():
        if w is None:
            continue
        t = targets.get(tk, 0)
        drift = w - t
        if abs(drift) > th["max_drift_pp"]:
            direction = "above" if drift > 0 else "below"
            action = "consider trimming back to target" if drift > 0 else "consider adding back to target"
            alerts.append({"level": "warning",
                           "message": f"**{tk}** has drifted {abs(drift):.1f}pp {direction} its {t:.1f}% target (now {w:.1f}%) — {action}."})

    tier_now, tier_target = {}, {}
    for tk, w in weight_now.items():
        ti = tiers.get(tk)
        if ti and w is not None:
            tier_now[ti] = tier_now.get(ti, 0) + w
    for tk, t in targets.items():
        ti = tiers.get(tk)
        if ti:
            tier_target[ti] = tier_target.get(ti, 0) + t
    for ti in tier_now:
        drift = tier_now[ti] - tier_target.get(ti, 0)
        if abs(drift) > th["tier_band_pp"]:
            alerts.append({"level": "warning",
                           "message": f"**{ti}** tier has drifted to {tier_now[ti]:.1f}% vs a ~{tier_target.get(ti,0):.1f}% target ({drift:+.1f}pp)."})

    cash_pct = nav.get("cash_pct")
    if cash_pct is not None and cash_pct > th["max_cash_pct"]:
        alerts.append({"level": "warning",
                       "message": f"Residual cash is {cash_pct:.1f}% of NAV — above the {th['max_cash_pct']:.0f}% guideline; consider redeploying."})

    return alerts


# ========================================
# HOME — Market Dashboard (landing page)
# ========================================
selected = st.session_state.nav_page
if selected == "Home":
    # ---- compact ticker tape ----
    tape = home_tape_quotes()
    chips = []
    for q in tape:
        if q["last"] is None:
            continue
        pct = q["pct"]
        color = "#9aa4b2" if pct is None else ("#34d399" if pct >= 0 else "#f87171")
        arrow = "" if pct is None else ("▲" if pct >= 0 else "▼")
        last_txt = f"{q['last']:,.2f}"
        pct_txt = "" if pct is None else f" {arrow}{abs(pct):.2f}%"
        chips.append(
            f"<span style='display:inline-block;padding:7px 18px;margin:2px;border-right:1px solid #2a2f3a;white-space:nowrap;'>"
            f"<b style='color:#cbd5e1;font-size:.92rem'>{q['label']}</b> "
            f"<span style='color:#e5e7eb;font-size:.92rem'>{last_txt}</span>"
            f"<span style='color:{color};font-size:.88rem'>{pct_txt}</span></span>"
        )
    mkt_label = ("<span style='background:#16a34a;color:#fff;font-size:.62rem;padding:2px 7px;"
                  "border-radius:3px;font-family:monospace;margin-right:8px;'>● LIVE</span>"
                  if (tape and tape[0].get("open")) else
                  "<span style='background:#374151;color:#9aa4b2;font-size:.62rem;padding:2px 7px;"
                  "border-radius:3px;font-family:monospace;margin-right:8px;'>FUTURES</span>")
    st.markdown(
        "<div style='background:#0e1117;border:1px solid #1f2530;border-radius:8px;"
        "padding:8px 8px;overflow-x:auto;white-space:nowrap;margin-bottom:16px'>"
        + mkt_label + "".join(chips) + "</div>", unsafe_allow_html=True)

    st.title("Market Dashboard")

    # ---- main grid: news (left, wide) + market chart / rates (right) ----
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("#### Headlines")
        news_view = st.radio("Lens", ["Factors", "Asset Classes"], horizontal=True, key="home_news_lens",
                             label_visibility="collapsed")
        if news_view == "Factors":
            buckets = ["Fundamentals", "Valuation", "Interest Rates", "Policy", "Behavioral / Trends"]
        else:
            buckets = ["US Equities", "International", "Fixed Income", "Commodities / Real Assets"]
        cats = home_news_categorized(news_view, per_bucket=4)
        any_news = False
        for b in buckets:
            items = cats.get(b, [])
            if not items:
                continue
            any_news = True
            st.markdown(f"**{b}**")
            for title, source, link, when in items:
                meta = f"· {source}" + (f" · {when}" if when else "")
                if link:
                    st.markdown(f"- [{title}]({link})  <span style='color:#8a93a3;font-size:.72rem'>{meta}</span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"- {title}  <span style='color:#8a93a3;font-size:.72rem'>{meta}</span>",
                                unsafe_allow_html=True)
        if not any_news:
            st.info("Headlines are temporarily unavailable (news feeds unreachable). Market data above is unaffected.")
        st.caption("Headlines from quality public RSS feeds (WSJ, The Economist, CNBC, AP), filtered to drop "
                   "advertorial/personal-finance fluff and bucketed by topic via keyword match. WSJ/Economist links "
                   "open in your browser, so your own logins handle the paywall. Asset-class items are assigned to their "
                   "best-matching class, not just the source section. Showing items from roughly the last 24 hours.")

    with right:
        st.markdown("#### Market Snapshot")
        _mc_sym = st.radio("Index", ["SPY", "QQQ", "DIA", "IWM"], horizontal=True, key="home_mkt_sym",
                           label_visibility="collapsed")
        _mc = home_market_chart(_mc_sym, "6mo")
        if _mc:
            _mcc1, _mcc2 = st.columns([2, 1])
            _mcc1.metric(f"{_mc_sym}", f"${_mc['last']:,.2f}",
                         f"{_mc['chg_pct']:+.1f}% (6mo)" if _mc['chg_pct'] is not None else None)
            mfig = home_market_chart_figure(_mc)
            if mfig is not None:
                st.plotly_chart(mfig, use_container_width=True)
        else:
            st.info("Market chart temporarily unavailable.")
        st.divider()

        st.markdown("#### Rates & the Fed")
        # FedWatch — CME methodology. Current rate editable since it can't be inferred.
        _fed_mid_input = st.number_input(
            "Current Fed Funds target midpoint (%)", min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("fed_current_mid", FED_CURRENT_MID_DEFAULT)),
            step=0.25, format="%.3f", key="fed_mid_home",
            help=f"Midpoint of the current target range (e.g. 3.625 = 350-375 bps). "
                 f"Update when the Fed moves. Default: {FED_CURRENT_MID_DEFAULT}%.")
        st.session_state["fed_current_mid"] = _fed_mid_input
        fed = home_fed_probabilities(current_mid=_fed_mid_input)
        if fed:
            st.markdown(f"**FOMC {fed.get('meeting','')} — implied**"
                        f" <span style='color:#8a93a3;font-size:.7rem'>(CME method, from Fed Funds futures)</span>",
                        unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Cut 25", f"{fed['p_cut']}%")
            fc2.metric("Hold", f"{fed['p_hold']}%")
            fc3.metric("Hike 25", f"{fed['p_hike']}%")
            if fed.get('p_cut50') or fed.get('p_hike50'):
                st.caption(f"50 bps cut {fed.get('p_cut50',0)}% · 50 bps hike {fed.get('p_hike50',0)}%")
            st.caption(f"ZQ=F implies ~{fed['implied_rate']:.2f}% post-meeting vs {fed['target_mid']:.3f}% now "
                       f"({fed['move_bps']:+.0f} bps). Meeting-date-weighted; derived, not CME's published number.")
        else:
            st.info("Fed-futures temporarily unavailable.")

        st.markdown("**Treasury Yield Curve**")
        yc = treasury_latest_curve_dict()
        fig = home_yield_curve_figure(yc)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            sp = home_curve_spreads(yc)
            if sp:
                s1, s2 = st.columns(2)
                if "2s10s" in sp:
                    s1.metric("2s10s", f"{sp['2s10s']:+d} bps",
                              help="10Y minus 2Y. Negative = inverted.")
                if "3m10y" in sp:
                    s2.metric("3m10y", f"{sp['3m10y']:+d} bps",
                              help="10Y minus 3-month. The Fed's preferred recession gauge.")
        else:
            st.info("Yield-curve data temporarily unavailable (Treasury feed unreachable).")

    st.divider()
    st.caption("Use the sidebar to navigate: Trading Hub · Markets · Research · My Portfolios. "
               "Data is delayed/real-time from public sources and is for information only.")

# ========================================
# TRADING HUB
# ========================================

elif selected == "Trading Hub":
    st.header("Trading Hub - Multi-Ticker Analysis")

    with st.expander("⚙️ Trading Settings", expanded=False):
        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            TRADING_MODE = st.radio("Mode", ["Paper Trading", "Live Trading"], key="th_mode")
            MIN_CONVICTION = st.slider("Min Conviction", 1, 10, 5, key="th_minconv")
            SIGNAL_EXPIRATION_MINUTES = st.slider("Signal Expiration (min)", 10, 120, DEFAULT_SIGNAL_EXPIRATION, key="th_sigexp")
        with _sc2:
            ENABLE_MACRO_FILTER = st.checkbox("Macro Signal Filter", value=True, key="th_macro")
            USE_DYNAMIC_STOPS = st.checkbox("Dynamic Position Sizing", value=True, key="th_dyn")
            STOP_LOSS_PCT = st.slider("Stop Loss %", -5.0, -0.5, -2.0, 0.5, key="th_stop")
            TRAILING_STOP_PCT = st.slider("Trailing Stop %", 0.5, 3.0, 1.0, 0.5, key="th_trail")
        with _sc3:
            ENABLE_CONTINUATION_SIGNALS = st.checkbox("Continuation Signals", value=True, key="th_cont")
            ENABLE_SUPPORT_RESISTANCE = st.checkbox("Support/Resistance", value=True, key="th_sr")
            ENABLE_OPTIONS_SIGNALS = st.checkbox("Options Signals", value=True, key="th_opt")
            ENABLE_TREND_FOLLOWING = st.checkbox("Trend Following", value=True, key="th_trend")
            ENABLE_MEAN_REVERSION = st.checkbox("Mean Reversion", value=True, key="th_mr")
        _dc1, _dc2 = st.columns(2)
        with _dc1:
            if st.button("💾 Force Save All Data", key="th_save"):
                save_all_data(); st.success("✅ All data saved!")
        with _dc2:
            st.caption(f"Mode: {TRADING_MODE} · Last save: {st.session_state.last_save.strftime('%H:%M:%S')}")
    
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
            "Value": "JVAL",
            "Momentum": "MTUM",
            "Quality": "QUAL",
            "Size (Small Cap)": "SIZE",
            "Low Volatility": "USMV",
            "Dividend": "DVY",
            "Growth": "IVW",
            "High Dividend": "HDV"
        },
        "International Factors": {
            "EAFE Value": "EFV",
            "EAFE Growth": "EFG",
            "EAFE Quality": "IQLT",
            "Intl Momentum": "IMTM",
            "Intl Low Volatility": "EFAV",
            "Intl Dividend": "IDV",
            "Intl High Dividend": "VYMI",
            "Intl Small Cap": "SCZ",
            "Intl Multi-Factor": "INTF"
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
                                # N-year window anchored to the latest close: start = anchor - N years.
                                # period_value is trading days (252/yr); convert to calendar years.
                                _years = period_value / 252.0
                                _start = anchor - pd.Timedelta(days=int(round(_years * 365.25)))
                                period_hist = hist[hist.index >= _start]
                            
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

    period = st.radio("History", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "Custom"], index=2, horizontal=True, key="chart_period")
    _cust_start = _cust_end = None
    if period == "Custom":
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            _cust_start = st.date_input("Start", value=datetime.now() - timedelta(days=180), key="chart_cust_start")
        with _cc2:
            _cust_end = st.date_input("End", value=datetime.now(), key="chart_cust_end")

    valid, info = validate_ticker(chosen)
    if not valid:
        st.error(f"Could not find data for '{chosen}'. Check the symbol (US-listed stocks and ETFs work; some ADRs/foreign listings may not).")
    else:
        nm = info.get("name", chosen)
        st.markdown(f"### {chosen} — {nm}")

        research_tabs = st.tabs(["📈 Chart & Technicals", "🏢 Overview", "📊 Financials", "🔍 Peer Comparison", "📄 Research Links"])

        # ===== Tab 1: Chart & Technicals (existing chart/scorecard content) =====
        with research_tabs[0]:
            try:
                t = yf.Ticker(chosen)
                if period == "Custom" and _cust_start and _cust_end:
                    hist = t.history(start=_cust_start, end=_cust_end, interval="1d")
                else:
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
                    try:
                        pret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                        s5.metric(f"{period} Return", f"{pret:+.1f}%")
                    except Exception:
                        s5.metric(f"{period} Return", "n/a")

                    sc = ta_scorecard(df)
                    if sc:
                        vcolor = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#eab308"}[sc["verdict"]]
                        st.markdown(
                            f"<div style='border:1px solid #2a2f3a;border-radius:8px;padding:10px 14px;margin:8px 0;'>"
                            f"<span style='font-size:.8rem;color:#9aa4b2'>Technical signal</span><br>"
                            f"<span style='font-size:1.5rem;font-weight:700;color:{vcolor}'>{sc['verdict']}</span> "
                            f"<span style='color:#9aa4b2;font-size:.85rem'>· {sc['bull']} bullish / {sc['bear']} bearish / {sc['neutral']} neutral "
                            f"(score {sc['score']:+d})</span></div>", unsafe_allow_html=True)
                        sig_map = {1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"}
                        srows = [{"Indicator": r["indicator"], "Signal": sig_map[r["signal"]], "Reading": r["detail"]}
                                 for r in sc["rows"]]
                        st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
                        st.caption("Each indicator votes bullish/bearish/neutral; the verdict is the net. This is a "
                                   "mechanical read of price/volume momentum and trend, not a recommendation — combine with "
                                   "fundamentals and your entry targets.")
                        with st.expander("📖 What do these indicators mean?"):
                            for term, desc in TA_GLOSSARY:
                                st.markdown(f"**{term}** — {desc}")

                    t_tgt = AI_ENTRY_TARGETS.get(chosen)
                    if t_tgt:
                        zone, _ = ai_entry_zone(chosen, cur)
                        st.info(f"**AI portfolio holding — entry targets:** optimal ${t_tgt['optimal']:,.2f} · "
                                f"secondary ${t_tgt['secondary']:,.2f} · do-not-exceed ${t_tgt['ceiling']:,.2f}  →  **{zone}**")
            except Exception as e:
                st.error(f"Error loading chart: {e}")

        # ===== Tab 2: Overview (finviz-style snapshot grid) =====
        with research_tabs[1]:
            with st.spinner(f"Fetching {chosen} overview data..."):
                ov_info = stock_overview_info(chosen)
            if not ov_info:
                st.warning("Overview data temporarily unavailable.")
            else:
                nm2 = ov_info.get("longName", chosen)
                st.markdown(f"**{nm2}**")
                summary = ov_info.get("longBusinessSummary", "")
                if summary:
                    with st.expander("Business summary", expanded=False):
                        st.write(summary)
                grid = stock_overview_grid(ov_info)
                grid_cols = st.columns(3)
                cats = list(grid.keys())
                for i, cat in enumerate(cats):
                    with grid_cols[i % 3]:
                        st.markdown(f"**{cat}**")
                        for label, val in grid[cat]:
                            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:.85rem;padding:2px 0;'>"
                                       f"<span style='color:#9aa4b2'>{label}</span><span>{val}</span></div>",
                                       unsafe_allow_html=True)
                        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.caption("Data from Yahoo Finance via yfinance. Fields show 'n/a' when the data provider doesn't report that metric for this security (common for ETFs, foreign listings, or recent IPOs).")

        # ===== Tab 3: Financials (full statements + ratios) =====
        with research_tabs[2]:
            fc1, fc2 = st.columns(2)
            with fc1:
                fin_statement = st.selectbox("Statement", ["Income Statement", "Balance Sheet", "Cash Flow"], key="fin_statement")
            with fc2:
                fin_period = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="fin_period")
            stmt_key = {"Income Statement": "income", "Balance Sheet": "balance", "Cash Flow": "cashflow"}[fin_statement]
            period_key = "quarterly" if fin_period == "Quarterly" else "annual"
            with st.spinner(f"Fetching {chosen} {fin_statement.lower()}..."):
                stmt_df = stock_financial_statements(chosen, stmt_key, period_key)
            if stmt_df.empty:
                st.info(f"{fin_statement} data unavailable for {chosen} (common for ETFs, foreign listings, or thinly-covered names).")
            else:
                disp = stmt_df.copy()
                disp.columns = [pd.Timestamp(c).strftime("%b %Y") for c in disp.columns]
                def _fmt_cell(v):
                    if pd.isna(v): return ""
                    av = abs(v)
                    if av >= 1e9: return f"{v/1e9:,.2f}B"
                    if av >= 1e6: return f"{v/1e6:,.1f}M"
                    return f"{v:,.0f}"
                disp = disp.apply(lambda col: col.map(_fmt_cell))
                st.dataframe(disp, use_container_width=True, height=460)
                st.caption("All figures in USD. Source: Yahoo Finance via yfinance.")

            st.divider()
            st.markdown("##### Key Ratios")
            with st.spinner("Computing ratios..."):
                inc_a = stock_financial_statements(chosen, "income", "annual")
                bal_a = stock_financial_statements(chosen, "balance", "annual")
                cf_a = stock_financial_statements(chosen, "cashflow", "annual")
                ratios = stock_compute_ratios(chosen, ov_info if 'ov_info' in dir() else stock_overview_info(chosen), inc_a, bal_a, cf_a)
            if ratios:
                rcols = st.columns(4)
                for i, (label, val) in enumerate(ratios.items()):
                    rcols[i % 4].metric(label, val)
                st.caption("Ratios computed from the most recent annual statements (or from Yahoo's reported margins where statement line items aren't available). "
                           "Row-label matching is best-effort since exact statement line-item names vary by company and data-provider version.")
            else:
                st.info("Insufficient statement data to compute ratios for this security.")

        # ===== Tab 4: Peer Comparison =====
        with research_tabs[3]:
            st.caption("Compare key metrics against other tickers. Enter comma-separated symbols.")
            default_peers = ""
            _ov_sector = (ov_info or {}).get("sector") if 'ov_info' in dir() else None
            peer_input = st.text_input("Peer tickers (comma-separated)", value=default_peers,
                                       placeholder="e.g. AMD, AVGO, TSM", key="peer_tickers_input")
            peer_list = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
            all_tickers = [chosen] + [p for p in peer_list if p != chosen]
            if len(all_tickers) < 2:
                st.info("Add at least one peer ticker above to see a comparison.")
            else:
                with st.spinner("Fetching peer data..."):
                    comp_df = stock_peer_comparison(all_tickers)
                if comp_df.empty:
                    st.warning("Could not fetch peer comparison data.")
                else:
                    st.dataframe(comp_df, use_container_width=True)
                    st.caption(f"**{chosen}** (first column) is the selected ticker. Sector/industry: "
                               f"{(ov_info or {}).get('sector','n/a') if 'ov_info' in dir() else 'n/a'} / "
                               f"{(ov_info or {}).get('industry','n/a') if 'ov_info' in dir() else 'n/a'}.")

        # ===== Tab 5: Research Links (EDGAR, IR, earnings) =====
        with research_tabs[4]:
            links = stock_research_links(chosen)
            ei = em_earnings_info(chosen)
            lc1, lc2 = st.columns([1, 2])
            with lc1:
                if ei.get("next"):
                    st.metric("Next earnings", ei["next"], (f"in {ei['days_until']}d" if ei.get("days_until") is not None else None), delta_color="off")
                elif ei.get("last"):
                    st.metric("Last earnings", ei["last"], delta_color="off")
            with lc2:
                st.markdown(
                    f"**Deeper dive:** "
                    f"[SEC filings (EDGAR)]({links['edgar_full']}) · "
                    f"[Latest 10-K/10-Q]({links['edgar_10k']}) · "
                    f"[Yahoo estimates]({links['yf_estimates']}) · "
                    f"[Yahoo financials]({links['yf_financials']}) · "
                    f"[Press releases]({links['yf_press']}) · "
                    f"[Investor Relations ↗]({links['ir_search']})")
            st.caption("EDGAR links go straight to the company's filings (10-K annual, 10-Q quarterly, 8-K for earnings). "
                       "The IR link is a search since IR URLs aren't standardized.")
            if 'ov_info' in dir() and ov_info and ov_info.get("website"):
                st.markdown(f"**Website:** [{ov_info['website']}]({ov_info['website']})")


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

        _ocf1, _ocf2, _ocf3 = st.columns(3)
        with _ocf1:
            option_type = st.radio("Type", ["Calls", "Puts", "Both"], horizontal=True, key="opt_type_radio")
        with _ocf2:
            _exps = ["All"] + sorted(chain_df['expiration'].dropna().unique().tolist())
            _exp_pick = st.selectbox("Expiration", _exps, key="opt_exp_pick")
        with _ocf3:
            _money = st.radio("Moneyness", ["All", "ITM", "ATM", "OTM"], horizontal=True, key="opt_money",
                              help="Relative to the current spot price. ATM = nearest ~6% band around spot.")

        if option_type == "Calls":
            filtered = chain_df[chain_df['type'] == 'Call']
        elif option_type == "Puts":
            filtered = chain_df[chain_df['type'] == 'Put']
        else:
            filtered = chain_df

        if _exp_pick != "All":
            filtered = filtered[filtered['expiration'] == _exp_pick]

        # moneyness filter relative to live spot
        if _money != "All":
            try:
                _spot = float(yf.Ticker(st.session_state.get('opt_chain_ticker', ticker_choice)).history(period="1d")['Close'].iloc[-1])
            except Exception:
                _spot = float(filtered['strike'].median()) if not filtered.empty else None
            if _spot:
                _band = _spot * 0.03
                def _is_money(row):
                    k = row['strike']; typ = row['type']
                    if _money == "ATM":
                        return abs(k - _spot) <= _band
                    if _money == "ITM":
                        return (k < _spot) if typ == 'Call' else (k > _spot)
                    if _money == "OTM":
                        return (k > _spot) if typ == 'Call' else (k < _spot)
                    return True
                if not filtered.empty:
                    filtered = filtered[filtered.apply(_is_money, axis=1)]
                st.caption(f"Spot ≈ ${_spot:,.2f} · ATM band ±${_band:,.2f}")

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

    ai_tabs = st.tabs(["📊 Portfolio", "📈 Performance", "✏️ Manage", "👁️ Bench", "📜 Mandate", "🎯 Conviction", "📄 Fact Sheet", "📰 Earnings Monitor", "🧾 Transactions"])

    # ---------------- PORTFOLIO ----------------
    with ai_tabs[0]:
        holdings = ai_load_portfolio()
        st.subheader("Model Portfolio: Live Holdings")
        with st.spinner("Fetching live prices..."):
            strat_ret, bench_ret, per = ai_compute_performance(holdings, AI_STRATEGY_INCEPTION, AI_STRATEGY_BENCHMARK)
        hk = tuple((h['ticker'], h.get('target_weight') or 0) for h in holdings if (h.get('target_weight') or 0) > 0)
        with st.spinner("Computing actual position drift..."):
            nav = ai_nav_shares_and_weights(hk)

        # headline metrics — ACTUAL share-based return (fixes "sticking with model weights")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Holdings", f"{len(holdings)}")
        tw = sum(h.get('target_weight') or 0 for h in holdings)
        mc2.metric("Invested Weight (target)", f"{tw:.0f}%", help="Sum of target weights. Remainder is cash.")
        actual_ret = nav.get("total_return_pct")
        mc3.metric("Strategy (actual, since inception)", f"{actual_ret:+.1f}%" if actual_ret is not None else "n/a",
                   help="Share-based return: shares bought 2/10/2026 at original weights, revalued and re-bought at the "
                        "6/1/2026 rebalance, held flat since. Reflects what the portfolio actually earned, not a "
                        "continuously-rebalanced-to-target approximation.")
        delta_v = (actual_ret - bench_ret) if (actual_ret is not None and bench_ret is not None) else None
        mc4.metric("NDX (benchmark)", f"{bench_ret:+.1f}%" if bench_ret is not None else "n/a",
                   f"{delta_v:+.1f}% active" if delta_v is not None else None)
        st.caption(f"Inception date: **{AI_STRATEGY_INCEPTION}** · Rebalanced **{AI_REBALANCE_DATE}** · Benchmark: Nasdaq-100 (NDX) · "
                   f"Showing current holdings only (target weight > 0). NAV as of {nav.get('asof','today')}: ${nav.get('nav_now',0):,.0f} "
                   f"(cash ${nav.get('cash_residual',0):,.0f}, {nav.get('cash_pct',0):.1f}%).")

        # tier allocation summary — now shows ACTUAL drifted tier weight vs target
        st.markdown("##### Tier Allocation: Actual (Drift) vs Target")
        weight_now = nav.get("weight_now", {})
        tier_actual = {}
        for h in holdings:
            w = weight_now.get(h['ticker'])
            if w is not None:
                tier_actual[h['tier']] = tier_actual.get(h['tier'], 0) + w
        tcols = st.columns(len(AI_TIER_TARGETS))
        for i, (tier, target) in enumerate(AI_TIER_TARGETS.items()):
            if tier == "CASH":
                cur = nav.get("cash_pct", max(0, 100 - tw))
            else:
                cur = tier_actual.get(tier, 0)
            tcols[i].metric(tier.title(), f"{cur:.1f}%", f"target {target}%", delta_color="off")
        st.caption("Actual reflects real price drift since the 6/1 rebalance (shares held flat, revalued at live prices) — not a re-derivation of target weights.")

        # ----- Mandate Alerts -----
        st.markdown("##### Mandate Alerts")
        with st.expander("⚙️ Adjust mandate thresholds", expanded=False):
            th1, th2, th3, th4 = st.columns(4)
            with th1:
                _max_single = st.number_input("Max single-name %", min_value=1.0, max_value=30.0,
                                              value=AI_MANDATE_DEFAULTS["max_single_name_pct"], step=0.5, key="mandate_max_single")
            with th2:
                _max_drift = st.number_input("Max drift (pp)", min_value=0.5, max_value=10.0,
                                             value=AI_MANDATE_DEFAULTS["max_drift_pp"], step=0.5, key="mandate_max_drift")
            with th3:
                _tier_band = st.number_input("Tier band (pp)", min_value=1.0, max_value=15.0,
                                             value=AI_MANDATE_DEFAULTS["tier_band_pp"], step=0.5, key="mandate_tier_band")
            with th4:
                _max_cash = st.number_input("Max cash %", min_value=1.0, max_value=30.0,
                                            value=AI_MANDATE_DEFAULTS["max_cash_pct"], step=0.5, key="mandate_max_cash")
        _mandate_th = {"max_single_name_pct": _max_single, "max_drift_pp": _max_drift,
                       "tier_band_pp": _tier_band, "max_cash_pct": _max_cash}
        alerts = ai_mandate_alerts(nav, holdings, _mandate_th)
        if alerts:
            for a in alerts:
                if a["level"] == "error":
                    st.error(a["message"])
                else:
                    st.warning(a["message"])
        else:
            st.success("✅ Portfolio is within all mandate guidelines — no single-name, tier, drift, or cash breaches.")

        # ----- entry-target view toggle -----
        st.markdown("##### Holdings")
        show_entries = st.checkbox("Show entry-target columns (optimal / secondary / do-not-exceed)", value=True,
                                   help="Limit-order anchors from the buildout plan. Zone shows where the live price sits now.")

        per_current = [p for p in per if (p.get('target_weight') or 0) > 0]
        df = pd.DataFrame(per_current)
        if not df.empty:
            df['Price'] = df['price_now'].apply(lambda x: f"${x:,.2f}" if x is not None else "n/a")
            df['Since Incept'] = df['ret_pct'].apply(lambda x: f"{x:+.1f}%" if x is not None else "n/a")
            df['Contribution'] = df['contribution'].apply(lambda x: f"{x:+.2f}%" if x is not None else "n/a")
            df['Target Wt'] = df['target_weight'].apply(lambda x: f"{x:.1f}%" if x is not None else "")
            df['Current Wt'] = df['ticker'].apply(lambda tk: f"{weight_now.get(tk):.1f}%" if weight_now.get(tk) is not None else "n/a")
            def _drift_fmt(row):
                w = weight_now.get(row['ticker']); t = row.get('target_weight') or 0
                if w is None: return "n/a"
                d = w - t
                flag = " 🔴" if abs(d) > _max_drift else (" 🟡" if abs(d) > _max_drift/2 else "")
                return f"{d:+.1f}pp{flag}"
            df['Drift'] = df.apply(_drift_fmt, axis=1)
            df['Score'] = df['score'].apply(lambda x: f"{x:.1f}" if x is not None else "")

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
                cols = ['tier', 'ticker', 'name', 'Score', 'Target Wt', 'Current Wt', 'Drift', 'Price', 'Zone',
                        'Optimal', 'Secondary', 'Do-Not-Exceed', 'Since Incept']
            else:
                cols = ['tier', 'ticker', 'name', 'Score', 'Target Wt', 'Current Wt', 'Drift', 'Price', 'Since Incept', 'Contribution']
            show = df[cols].rename(columns={'tier': 'Tier', 'ticker': 'Ticker', 'name': 'Name'})
            st.dataframe(show, use_container_width=True, hide_index=True, height=560)
            st.caption(f"Current Wt / Drift reflect actual share-based drift since the {AI_REBALANCE_DATE} rebalance (live prices), "
                       f"not a re-derivation of the target weight. 🔴 = drift exceeds {_max_drift:.1f}pp · 🟡 = over half that threshold.")
            if show_entries:
                st.caption("Zone: 🟢 at/below optimal or in buy zone · 🟡 below the do-not-exceed ceiling · 🔴 above ceiling · "
                           "🔄 RE-RATE/Stale = a recent earnings or guidance event (or a gap >12% past the ceiling) has made the static levels obsolete — re-derive off new guidance before acting. "
                           "Ceilings are anchored to a forward-P/E multiple (shown in the thesis view), so they travel with estimates rather than staying fixed. "
                           "Optimal = the pullback worth waiting for; Secondary = scale-in; Do-Not-Exceed = ceiling. Limit-order guides, not forecasts.")
                _reval = [h['ticker'] for h in holdings if (AI_ENTRY_TARGETS.get(h['ticker']) or {}).get('reval')]
                if _reval:
                    st.warning("🔄 Targets need refreshing after a recent re-rating event: " + ", ".join(_reval)
                               + ". These names reported or gapped on news; re-derive their optimal/secondary/ceiling off the updated guidance.")

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

        # ===== Actual Portfolio Return (share-based, accounts for the 6/1 rebalance) =====
        hk = tuple((h['ticker'], h.get('target_weight') or 0) for h in holdings if (h.get('target_weight') or 0) > 0)
        with st.spinner("Computing actual share-based return..."):
            nav = ai_nav_shares_and_weights(hk)
        st.markdown("##### Actual Portfolio Return (share-based)")
        st.success(
            f"**{nav.get('total_return_pct', 0):+.2f}%** since {AI_STRATEGY_INCEPTION} — NAV **${nav.get('nav_now', 0):,.0f}** "
            f"(cash ${nav.get('cash_residual', 0):,.0f}, {nav.get('cash_pct', 0):.1f}%). "
            f"This tracks actual share counts through the {AI_REBALANCE_DATE} rebalance and current live prices — "
            f"the number to trust for \"how has this portfolio actually done.\""
        )
        st.caption("Methodology: shares bought 2/10/2026 at original model weights → revalued at 6/1/2026 close → "
                   "re-bought at 6/1 to current target weights using that actual (not a fresh $100k) value → held flat "
                   "since, drifting with live prices. This fixes the prior approximation below, which continuously "
                   "re-weights every holding to its *current* target for the whole period — that silently gives sold "
                   "names (BABA, HPE) zero credit for the P&L they actually generated, and gives new adds (PANW, ARM, SO) "
                   "credit for price moves before the portfolio owned them.")
        st.divider()
        st.markdown("##### Model Breakdown (continuously target-weighted — illustrative, see note above)")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Strategy (Model)", f"{strat_ret:+.2f}%" if strat_ret is not None else "n/a")
        pc2.metric("Nasdaq-100", f"{bench_ret:+.2f}%" if bench_ret is not None else "n/a")
        active = (strat_ret - bench_ret) if (strat_ret is not None and bench_ret is not None) else None
        pc3.metric("Active Return", f"{active:+.2f}%" if active is not None else "n/a",
                   help="Strategy minus benchmark. Positive = outperforming NDX.")

        st.info("This breakdown below is **model** performance: current target weights × each holding's actual price change since inception, "
                "continuously applied for the whole period (a simplification used for per-name contribution/attribution below — it does not "
                "reflect the actual 6/1 rebalance timing the way the Actual Portfolio Return above does). "
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
        # ===== Batting Average =====
        st.divider()
        st.markdown("##### Batting Average")
        st.caption("Share of holdings that beat the benchmark on since-inception return. NDX is the primary measure; S&P 500 (VOO) is secondary.")
        ndx_tot = ec.get("ndx_total") if ec else bench_ret
        voo_tot = ec.get("voo_total") if ec else None
        ba = ai_batting_average(per, ndx_tot, voo_tot)
        if ba["total"]:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Holdings scored", f"{ba['total']}")
            bc2.metric("Batting avg vs NDX", f"{ba['ba_ndx']:.0f}%" if ba['ba_ndx'] is not None else "n/a",
                       f"{ba['beat_ndx']} of {ba['total']} beat NDX" if ba['ba_ndx'] is not None else None, delta_color="off")
            bc3.metric("Batting avg vs S&P 500", f"{ba['ba_voo']:.0f}%" if ba['ba_voo'] is not None else "n/a",
                       f"{ba['beat_voo']} of {ba['total']} beat VOO" if ba['ba_voo'] is not None else None, delta_color="off")
            # per-holding beat/miss table
            tier_order = {"HYPER": 0, "TIER 1": 1, "TIER 2": 2, "TIER 3": 3}
            brow = []
            for r in sorted(ba["rows"], key=lambda x: (x['vs_ndx'] is None, -(x['vs_ndx'] or -999))):
                brow.append({
                    "Ticker": r['ticker'], "Tier": r['tier'],
                    "Return": f"{r['ret']:+.1f}%",
                    "vs NDX": f"{r['vs_ndx']:+.1f}" if r['vs_ndx'] is not None else "n/a",
                    "Beat NDX": "✅" if r['beat_ndx'] else "—",
                    "vs S&P": f"{r['vs_voo']:+.1f}" if r['vs_voo'] is not None else "n/a",
                    "Beat S&P": "✅" if r['beat_voo'] else "—",
                })
            st.dataframe(pd.DataFrame(brow), use_container_width=True, hide_index=True, height=420)
            st.caption(f"Benchmarks since inception: NDX {ndx_tot:+.1f}%" + (f" · S&P 500 (VOO) {voo_tot:+.1f}%" if voo_tot is not None else "")
                       + ". A high batting average with lower active return (or vice versa) tells you whether performance is broad-based or driven by a few names.")
        else:
            st.warning("Not enough return data to compute batting average.")

        # ===== Attribution (allocation + selection) =====
        st.divider()
        st.markdown("##### Attribution: Allocation vs Selection")
        st.caption("Brinson-Fachler decomposition by tier, portfolio vs the equal-weighted bench (watchlist) universe. "
                   "Allocation = did tier tilts help; Selection = did the names held beat the bench names in the same tier; "
                   "Interaction = the combined effect. The three sum to active return vs the bench universe.")
        with st.spinner("Fetching bench returns for attribution..."):
            _bt = [b['ticker'] for b in ai_load_bench()]
            _bp = ai_fetch_prices(tuple(_bt), AI_STRATEGY_INCEPTION)
            bench_per = [{"tier": b['tier'], "ret_pct": _bp.get(b['ticker'], {}).get('ret_pct')} for b in ai_load_bench()]
        attr_rows, attr_tot = ai_attribution_by_tier(per, bench_per)
        ar = []
        for r in attr_rows:
            ar.append({
                "Tier": r['tier'].replace("TIER ", "T").replace("HYPER", "Hyper"),
                "Port Wt": f"{r['wp']:.1f}%", "Bench Wt": f"{r['wb']:.1f}%",
                "Port Ret": f"{r['Rp']:+.1f}%", "Bench Ret": f"{r['Rb']:+.1f}%",
                "Allocation": f"{r['alloc']:+.2f}", "Selection": f"{r['sel']:+.2f}",
                "Interaction": f"{r['inter']:+.2f}", "Total": f"{r['total']:+.2f}",
            })
        ar.append({"Tier": "TOTAL", "Port Wt": "", "Bench Wt": "", "Port Ret": f"{attr_tot['Rp']:+.1f}%",
                   "Bench Ret": f"{attr_tot['Rb']:+.1f}%", "Allocation": f"{attr_tot['alloc']:+.2f}",
                   "Selection": f"{attr_tot['sel']:+.2f}", "Interaction": f"{attr_tot['inter']:+.2f}",
                   "Total": f"{attr_tot['active']:+.2f}"})
        st.dataframe(pd.DataFrame(ar), use_container_width=True, hide_index=True)
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Allocation effect", f"{attr_tot['alloc']:+.2f}%", help="Value from over/underweighting tiers vs the bench universe.")
        ac2.metric("Selection effect", f"{attr_tot['sel']:+.2f}%", help="Value from the specific names held vs bench names in the same tier.")
        ac3.metric("Active vs bench", f"{attr_tot['active']:+.2f}%", help="Total active return vs the equal-weight bench universe (alloc+selection+interaction).")
        st.caption("Note: this attribution is measured against the equal-weighted bench/watchlist universe (the names we track), "
                   "not against NDX, because the bench can be cleanly segmented into the four tiers. NDX and VOO remain the "
                   "return benchmarks for batting average and headline performance above.")

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

            # ----- Entry-target reference (read-only) -----
            st.divider()
            st.markdown("##### Entry Targets (optimal / secondary / do-not-exceed)")
            st.caption("Limit-order guides per holding. Ceilings are forward-P/E anchored; 🔄 names re-rated on a recent event and need re-derivation.")
            et_rows = []
            tier_order_m = {"HYPER": 0, "TIER 1": 1, "TIER 2": 2, "TIER 3": 3}
            for h in sorted([hh for hh in holdings if (hh.get('target_weight') or 0) > 0],
                            key=lambda x: (tier_order_m.get(x.get('tier'), 9), -(x.get('target_weight') or 0))):
                t = AI_ENTRY_TARGETS.get(h['ticker'])
                if not t:
                    continue
                et_rows.append({
                    "Ticker": h['ticker'], "Tier": h.get('tier', ''),
                    "Optimal": f"${t['optimal']:,.2f}", "Secondary": f"${t['secondary']:,.2f}",
                    "Do-Not-Exceed": f"${t['ceiling']:,.2f}", "P/E Anchor": t.get('pe_anchor', ''),
                    "Re-rate?": "🔄 refresh" if t.get('reval') else "—",
                })
            if et_rows:
                st.dataframe(pd.DataFrame(et_rows), use_container_width=True, hide_index=True, height=420)

            # ----- Share calculator -----
            st.divider()
            st.markdown("##### Share Calculator")
            st.caption("Whole-share counts to reach each target weight, using live prices. This is a buy list for a fresh book; "
                       "to compute buy/sell deltas against positions you already hold, enter your current shares in the right column.")
            calc_notional = st.number_input("Portfolio size ($)", min_value=1000.0, value=float(AI_PORTFOLIO_NOTIONAL),
                                            step=5000.0, key="ai_calc_notional")
            show_deltas = st.checkbox("I already hold some of these (enter current shares to get buy/sell deltas)", value=False, key="ai_calc_deltas")
            cur_holds = sorted([hh for hh in holdings if (hh.get('target_weight') or 0) > 0],
                               key=lambda x: (tier_order_m.get(x.get('tier'), 9), -(x.get('target_weight') or 0)))
            with st.spinner("Fetching live prices for the calculator..."):
                calc_px = ai_fetch_prices(tuple(h['ticker'] for h in cur_holds), AI_STRATEGY_INCEPTION)
            calc_seed = []
            for h in cur_holds:
                p = calc_px.get(h['ticker'], {}).get('price_now')
                tgt_dollars = calc_notional * ((h.get('target_weight') or 0) / 100.0)
                tgt_shares = int(tgt_dollars // p) if (p and p > 0) else 0
                calc_seed.append({"Ticker": h['ticker'], "Tier": h.get('tier', ''),
                                  "Target %": h.get('target_weight') or 0,
                                  "Price": round(p, 2) if p else None,
                                  "Target $": round(tgt_dollars, 0),
                                  "Target Shares": tgt_shares,
                                  "Current Shares": 0})
            calc_df = pd.DataFrame(calc_seed)
            if show_deltas:
                edited_calc = st.data_editor(
                    calc_df, use_container_width=True, hide_index=True, height=460, key="ai_calc_editor",
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                        "Tier": st.column_config.TextColumn("Tier", disabled=True),
                        "Target %": st.column_config.NumberColumn("Target %", disabled=True, format="%.1f"),
                        "Price": st.column_config.NumberColumn("Price", disabled=True, format="$%.2f"),
                        "Target $": st.column_config.NumberColumn("Target $", disabled=True, format="$%.0f"),
                        "Target Shares": st.column_config.NumberColumn("Target Shares", disabled=True),
                        "Current Shares": st.column_config.NumberColumn("Current Shares", min_value=0, step=1),
                    })
                deltas = []
                for _, r in edited_calc.iterrows():
                    tgt = int(r["Target Shares"]) if pd.notna(r["Target Shares"]) else 0
                    cur = int(r["Current Shares"]) if pd.notna(r["Current Shares"]) else 0
                    d = tgt - cur
                    deltas.append({"Ticker": r["Ticker"], "Tier": r["Tier"],
                                   "Target Shares": tgt, "Current Shares": cur,
                                   "Action": "BUY" if d > 0 else ("SELL" if d < 0 else "HOLD"),
                                   "Shares to Trade": abs(d),
                                   "Trade $": f"${abs(d) * (r['Price'] or 0):,.0f}"})
                st.markdown("**Buy / Sell to reach targets**")
                st.dataframe(pd.DataFrame(deltas), use_container_width=True, hide_index=True, height=460)
                st.download_button("⬇️ Download buy/sell list (CSV)", pd.DataFrame(deltas).to_csv(index=False),
                                   "ai_buy_sell_list.csv", "text/csv")
            else:
                disp_calc = calc_df.drop(columns=["Current Shares"]).copy()
                disp_calc["Price"] = disp_calc["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "n/a")
                disp_calc["Target $"] = disp_calc["Target $"].apply(lambda x: f"${x:,.0f}")
                disp_calc["Target %"] = disp_calc["Target %"].apply(lambda x: f"{x:.1f}%")
                disp_calc["Target Shares"] = disp_calc["Target Shares"].apply(lambda x: f"{int(x):,}")
                st.dataframe(disp_calc, use_container_width=True, hide_index=True, height=460)
                _tot = sum((r["Target Shares"] * (r["Price"] or 0)) for r in calc_seed)
                st.caption(f"Total deployed at target: ${_tot:,.0f} of ${calc_notional:,.0f} "
                           f"(${calc_notional - _tot:,.0f} residual cash). Prices are live, so counts shift intraday.")

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

    # ---------------- EARNINGS & NEWS MONITOR ----------------
    with ai_tabs[7]:
        st.subheader("Earnings & News Monitor")
        st.caption("Flags which holdings to review: upcoming/recent earnings, large price gaps, a P/E-anchored ceiling recompute from live forward EPS, and recent headlines. This surfaces what to look at; it does not change recommendations automatically.")

        mon_holdings = ai_load_portfolio()
        st.markdown("##### Review Flags")
        st.caption("Pure rules, no AI. 🔴 = something changed worth a look; 🟢 = nothing flagged.")

        with st.spinner("Scanning earnings calendar, price gaps, and estimates..."):
            mon_rows = []
            for h in mon_holdings:
                tk = h['ticker']
                pr = ai_fetch_prices((tk,), AI_STRATEGY_INCEPTION).get(tk, {})
                price_now = pr.get('price_now')
                status, reasons = em_review_flag(tk, price_now)
                ei = em_earnings_info(tk)
                new_ceil = em_recomputed_ceiling(tk)
                old = AI_ENTRY_TARGETS.get(tk) or {}
                old_ceil = old.get("ceiling")
                ceil_drift = None
                if new_ceil and old_ceil:
                    ceil_drift = (new_ceil / old_ceil - 1) * 100
                mon_rows.append({
                    "Ticker": tk,
                    "Status": status,
                    "Why": "; ".join(reasons) if reasons else "—",
                    "Next Earnings": ei.get("next") or "—",
                    "Anchor Ceiling": f"${old_ceil:,.2f}" if old_ceil else "—",
                    "Live-EPS Ceiling": f"${new_ceil:,.2f}" if new_ceil else "n/a",
                    "Ceiling Drift": f"{ceil_drift:+.0f}%" if ceil_drift is not None else "—",
                })
            mon_df = pd.DataFrame(mon_rows)
            # sort: review first
            mon_df["_o"] = mon_df["Status"].apply(lambda s: 0 if "REVIEW" in s else 1)
            mon_df = mon_df.sort_values(["_o", "Ticker"]).drop(columns="_o")
        st.dataframe(mon_df, use_container_width=True, hide_index=True, height=560)
        st.caption("Live-EPS Ceiling = (forward P/E anchor) × (live forward EPS from the data feed). "
                   "When it drifts far from the static Anchor Ceiling, the static target is stale; re-derive. "
                   "Forward EPS is best-effort from free data and can lag a fresh print by a day.")

        # per-name headlines + optional Claude interpretation
        st.divider()
        st.markdown("##### Headlines, Earnings & Research Links")
        pick = st.selectbox("Holding", [h['ticker'] for h in mon_holdings], key="em_pick")

        # next earnings + deeper-dive links
        _ei = em_earnings_info(pick)
        _el1, _el2 = st.columns([1, 2])
        with _el1:
            if _ei.get("next"):
                _du = _ei.get("days_until")
                st.metric("Next earnings", _ei["next"], (f"in {_du}d" if _du is not None else None), delta_color="off")
            elif _ei.get("last"):
                st.metric("Last earnings", _ei["last"], delta_color="off")
            else:
                st.metric("Next earnings", "n/a")
        with _el2:
            _edgar = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={pick}&type=10-K&dateb=&owner=include&count=10"
            _edgarq = f"https://efts.sec.gov/LATEST/search-index?q=%22{pick}%22"
            _edgar_full = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={pick}&type=&dateb=&owner=include&count=20"
            _yf_earn = f"https://finance.yahoo.com/quote/{pick}/analysis"
            _yf_press = f"https://finance.yahoo.com/quote/{pick}/press-releases"
            _ir_search = f"https://www.google.com/search?q={pick}+investor+relations"
            st.markdown(
                f"**Deeper dive:** "
                f"[SEC filings (EDGAR)]({_edgar_full}) · "
                f"[Latest 10-K/10-Q]({_edgar}) · "
                f"[Yahoo estimates]({_yf_earn}) · "
                f"[Press releases]({_yf_press}) · "
                f"[Investor Relations ↗]({_ir_search})")
            st.caption("EDGAR links go straight to the company's filings (10-K annual, 10-Q quarterly, 8-K for earnings). "
                       "The IR link is a search since IR URLs aren't standardized.")

        heads = em_headlines(pick, n=6)
        if heads:
            for title, pub, url in heads:
                if url:
                    st.markdown(f"- [{title}]({url})  \n  <span style='color:#8a93a3;font-size:.8rem'>{pub}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {title}  \n  <span style='color:#8a93a3;font-size:.8rem'>{pub}</span>", unsafe_allow_html=True)
        else:
            st.info("No recent headlines returned for this name.")

        with st.expander("🤖 Earnings Analysis — use your own Claude API key", expanded=False):
            st.caption(
                "Uses **your own** Anthropic API key — never stored, never charges anyone else's credits. "
                "Gathers quarterly EPS beat/miss history, revenue, margins, analyst recs, and recent headlines, "
                "then asks Claude for a structured analysis: last-quarter results, guidance, thesis validation, "
                "and whether conviction should be raised, maintained, or lowered."
            )
            _col_key, _col_model = st.columns([3, 1])
            with _col_key:
                user_key = st.text_input("Anthropic API key (sk-ant-...)", type="password", key="em_api_key",
                                         help="Your key from console.anthropic.com. Never stored server-side.")
                if user_key:
                    _k = user_key.strip()
                    _ok = _k.startswith("sk-ant-") and len(_k) >= 90
                    st.caption(
                        f"{'✅' if _ok else '⚠️'} {len(_k)} chars pasted — "
                        + ("looks complete." if _ok else
                           "key looks short or wrong format. Anthropic keys are ~108 chars starting with `sk-ant-`.")
                    )
            with _col_model:
                _model_choice = st.selectbox("Model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
                                             key="em_model", help="Haiku = faster/cheaper; Sonnet = deeper analysis")
            if st.button(f"📊 Analyse {pick} earnings", key="em_analyse_btn", use_container_width=True):
                if not user_key or not user_key.strip():
                    st.warning("Paste your API key above to run the analysis.")
                else:
                    with st.spinner(f"Gathering {pick} data and running analysis..."):
                        edata, analysis_text = em_earnings_analysis(user_key.strip(), pick, model=_model_choice)
                    if edata:
                        # show the data package that was sent to Claude
                        with st.expander("📋 Data used in analysis", expanded=False):
                            _d1, _d2, _d3 = st.columns(3)
                            _d1.metric("Fwd P/E", f"{edata['pe_fwd']:.1f}" if edata['pe_fwd'] else "n/a")
                            _d2.metric("EPS (fwd)", f"${edata['eps_fwd']:.2f}" if edata['eps_fwd'] else "n/a")
                            _d3.metric("Analyst rec", edata['analyst_rec'] or "n/a",
                                       f"{edata['analyst_count']} analysts" if edata['analyst_count'] else None,
                                       delta_color="off")
                            if edata["eps_history"]:
                                ep = pd.DataFrame(edata["eps_history"][:4])
                                ep.columns = [c.title() for c in ep.columns]
                                st.dataframe(ep, use_container_width=True, hide_index=True)
                    st.markdown(analysis_text)
                    st.caption(f"Analysis by {_model_choice} using your API key. For information only — "
                               "fundamental research and conviction changes remain your own judgment.")

    # ---------------- TRANSACTIONS ----------------
    with ai_tabs[8]:
        st.subheader("Transactions")
        st.caption(f"Initial book established {AI_INCEPTION_DATE} (30 names); rebalanced {AI_REBALANCE_DATE}. "
                   f"Modeled on a ${AI_PORTFOLIO_NOTIONAL:,.0f} book at each date's close, whole shares.")

        with st.spinner("Fetching 2/10 and 6/1 close prices..."):
            inc_rows, reb_rows, summ = ai_build_two_date_transactions(AI_PORTFOLIO_NOTIONAL)

        tier_order = {"HYPER": 0, "TIER 1": 1, "TIER 2": 2, "TIER 3": 3}

        # ===== Section 1: inception buys =====
        st.markdown(f"##### Initial Buys — {AI_INCEPTION_DATE}")
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Names bought", f"{summ['n_inception']}")
        ic2.metric("Invested", f"${summ['inception_invested']:,.0f}",
                   f"{summ['inception_invested']/summ['notional']*100:.1f}% of book")
        ic3.metric("Residual cash", f"${summ['notional']-summ['inception_invested']:,.0f}", delta_color="off")
        inc_disp = []
        for r in sorted(inc_rows, key=lambda x: (tier_order.get(x['tier'], 9), -(x['weight'] or 0))):
            inc_disp.append({"Date": r['date'], "Action": r['action'], "Ticker": r['ticker'],
                             "Name": r['name'], "Tier": r['tier'], "Target Wt": f"{r['weight']:.1f}%",
                             "Fill Price": f"${r['price']:,.2f}" if r['price'] else "n/a",
                             "Shares": f"{r['shares']:,}" if r['shares'] is not None else "n/a",
                             "Value": f"${r['value']:,.0f}" if r['value'] is not None else "n/a"})
        st.dataframe(pd.DataFrame(inc_disp), use_container_width=True, hide_index=True, height=420)

        # ===== Section 2: 6/1 rebalance =====
        st.divider()
        st.markdown(f"##### Rebalance Actions — {AI_REBALANCE_DATE}")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Sells", f"{summ['n_sell']}")
        rc2.metric("Trims", f"{summ['n_trim']}")
        rc3.metric("New buys", f"{summ['n_buy']}")
        rc4.metric("Adds", f"{summ['n_add']}")
        if reb_rows:
            action_order = {"SELL (exit)": 0, "TRIM": 1, "ADD": 2, "BUY (new)": 3}
            reb_disp = []
            for r in sorted(reb_rows, key=lambda x: (action_order.get(x['action'], 9), x['ticker'])):
                reb_disp.append({"Date": r['date'], "Action": r['action'], "Ticker": r['ticker'],
                                 "Name": r['name'], "Tier": r['tier'],
                                 "Drifted Wt": f"{r['from_w']:.1f}%", "Target Wt": f"{r['target_w']:.1f}%",
                                 "Fill Price": f"${r['price']:,.2f}" if r['price'] else "n/a",
                                 "Δ Shares": (f"{r['shares']:+,}" if r['shares'] is not None else "n/a"),
                                 "Δ Value": (f"${r['value']:+,.0f}" if r['value'] is not None else "n/a")})
            st.dataframe(pd.DataFrame(reb_disp), use_container_width=True, hide_index=True, height=320)
            st.caption("Drifted Wt = each position's weight at the 6/1 close after price moves since 2/10; "
                       "Target Wt = its current target. TRIM = sell back to target (e.g. winners like Dell that drifted "
                       "above weight), ADD = buy up to target, SELL = full exit, BUY (new) = new position. "
                       "Δ Shares/Value are the trade to return to target at the 6/1 close. BABA and HPE exited; "
                       "NEE trimmed on the Dominion-deal affordability risk; PANW, ARM, SO added.")
        else:
            st.info("No rebalance changes detected between the original book and the current holdings.")

        # combined downloadable blotter
        blot = []
        for r in inc_rows:
            blot.append({"date": r['date'], "action": r['action'], "ticker": r['ticker'],
                         "tier": r['tier'], "target_weight": r['weight'], "fill_price": r['price'],
                         "shares": r['shares'], "value": r['value']})
        for r in reb_rows:
            blot.append({"date": r['date'], "action": r['action'], "ticker": r['ticker'],
                         "tier": r['tier'], "drifted_weight": r['from_w'], "target_weight": r['target_w'],
                         "fill_price": r['price'], "share_delta": r['shares'], "value_delta": r['value']})
        st.download_button("⬇️ Download full transaction blotter (CSV)", pd.DataFrame(blot).to_csv(index=False),
                           "ai_transactions_full.csv", "text/csv")

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

    # ===== Rates, Curve & Fed (added) =====
    st.divider()
    st.subheader("Rates, Yield Curve & the Fed")

    msnap = home_macro_snapshot()
    if msnap:
        rorder = [k for k in ["13W TBill", "2Y", "5Y", "10Y", "30Y", "VIX", "DXY"] if k in msnap]
        if rorder:
            rc = st.columns(len(rorder))
            for i, k in enumerate(rorder):
                v = msnap[k]["value"]; ch = msnap[k].get("change")
                is_rate = k in ("13W TBill", "2Y", "5Y", "10Y", "30Y")
                rc[i].metric(k, f"{v:.2f}%" if is_rate else f"{v:,.2f}",
                             (f"{ch:+.02f}" + ("pp" if is_rate else "")) if ch is not None else None,
                             delta_color="inverse" if k == "VIX" else "normal")

    mc_left, mc_right = st.columns([1, 1])
    with mc_left:
        st.markdown("**Treasury Yield Curve**")
        yc = treasury_latest_curve_dict()
        fig = home_yield_curve_figure(yc)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            sp = home_curve_spreads(yc)
            if sp:
                cc1, cc2 = st.columns(2)
                if "2s10s" in sp:
                    cc1.metric("2s10s spread", f"{sp['2s10s']:+d} bps", help="10Y minus 2Y. Negative = inverted.")
                if "3m10y" in sp:
                    cc2.metric("3m10y spread", f"{sp['3m10y']:+d} bps", help="10Y minus 3-month T-bill.")
        else:
            st.info("Yield-curve data temporarily unavailable (Treasury feed unreachable).")

    with mc_right:
        st.markdown("**Fed: implied next move** <span style='color:#8a93a3;font-size:.72rem'>(derived from Fed Funds futures)</span>", unsafe_allow_html=True)
        _fed_mid_macro = st.number_input(
            "Current target midpoint (%)", min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("fed_current_mid", FED_CURRENT_MID_DEFAULT)),
            step=0.25, format="%.3f", key="fed_mid_macro",
            help="Update whenever the Fed moves rates. Shared with the Home page.")
        st.session_state["fed_current_mid"] = _fed_mid_macro
        fed = home_fed_probabilities(current_mid=_fed_mid_macro)
        if fed:
            st.caption(f"Next meeting: **{fed.get('meeting','')}** · ZQ=F @ {fed.get('front_price',0):.4f}")
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Cut 25", f"{fed['p_cut']}%")
            fc2.metric("Hold", f"{fed['p_hold']}%")
            fc3.metric("Hike 25", f"{fed['p_hike']}%")
            if fed.get('p_cut50') or fed.get('p_hike50'):
                st.caption(f"50 bps cut {fed.get('p_cut50',0)}% · 50 bps hike {fed.get('p_hike50',0)}%")
            st.caption(f"ZQ=F implies ~{fed['implied_rate']:.2f}% post-meeting vs {fed['target_mid']:.3f}% now "
                       f"({fed['move_bps']:+.0f} bps). CME FedWatch methodology (meeting-date-weighted), "
                       "derived here from the futures curve — CME exposes no free API.")
            st.markdown("[CME FedWatch (official)](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html) · "
                        "[Treasury yield curve](https://www.ustreasuryyieldcurve.com/)")
        else:
            st.info("Fed-futures data temporarily unavailable.")


    # ===== Treasury Yield Curve Explorer (added v14) =====
    st.divider()
    st.subheader("Treasury Yield Curve Explorer (1990–present)")
    st.caption("Sourced directly from the U.S. Treasury's Daily Par Yield Curve Rates — 1990–2025 is embedded in "
               "this app (no network call needed for history); the current year refreshes from Treasury's own CSV "
               "export every 12 hours. This replaces the FRED/third-party yield-curve source used previously.")

    with st.expander("📤 Manual data upload (use if live data looks stale or unreachable)"):
        st.caption("If the live Treasury fetch is ever blocked from this host, download the current-year CSV from "
                   "[Treasury's Daily Treasury Par Yield Curve Rates page](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) "
                   "— select the current year, click **Download CSV** — and upload it below. It merges into the data automatically.")
        _tup = st.file_uploader("Upload Treasury par yield curve CSV", type="csv", key="treasury_csv_upload")
        if _tup is not None:
            _parsed = treasury_parse_uploaded_csv(_tup)
            if _parsed is not None and not _parsed.empty:
                st.session_state["treasury_uploaded_df"] = _parsed
                st.success(f"Loaded {len(_parsed)} rows, {_parsed['Date'].min().date()} to {_parsed['Date'].max().date()}. Merged below.")
            else:
                st.error("Could not parse that file. Use the Treasury 'Daily Treasury Par Yield Curve Rates' CSV export.")

    tdf = treasury_full_data()
    if tdf.empty:
        st.warning("Treasury yield curve data unavailable.")
    else:
        st.caption(f"Data through **{tdf['Date'].max().strftime('%B %d, %Y')}** · {len(tdf):,} trading days on file (1990–present).")
        tc_tabs = st.tabs(["📈 Yield Curve", "📊 Single Maturity History", "📐 Spread Tracking"])

        # ---- Tab 1: Yield Curve with multi-period comparison ----
        with tc_tabs[0]:
            st.caption("Compare the current par yield curve against one or more prior periods.")
            latest_date = tdf["Date"].max()
            preset_options = {
                "1 week ago": latest_date - pd.Timedelta(weeks=1),
                "1 month ago": latest_date - pd.DateOffset(months=1),
                "3 months ago": latest_date - pd.DateOffset(months=3),
                "6 months ago": latest_date - pd.DateOffset(months=6),
                "1 year ago": latest_date - pd.DateOffset(years=1),
                "2 years ago": latest_date - pd.DateOffset(years=2),
                "5 years ago": latest_date - pd.DateOffset(years=5),
                "10 years ago": latest_date - pd.DateOffset(years=10),
            }
            picked = st.multiselect("Compare against", list(preset_options.keys()),
                                    default=["1 year ago"], key="yc_compare_presets")
            add_custom = st.checkbox("Add a custom comparison date", key="yc_custom_toggle")
            custom_date = None
            if add_custom:
                custom_date = st.date_input("Custom date", value=(latest_date - pd.DateOffset(years=3)).date(),
                                            min_value=tdf["Date"].min().date(), max_value=latest_date.date(),
                                            key="yc_custom_date")
            compare_dates = [("Today", latest_date)]
            for label in picked:
                compare_dates.append((label, preset_options[label]))
            if add_custom and custom_date:
                compare_dates.append((f"Custom ({custom_date})", pd.Timestamp(custom_date)))
            fig_yc = treasury_curve_comparison_figure(tdf, compare_dates)
            if fig_yc is not None:
                st.plotly_chart(fig_yc, use_container_width=True)
            else:
                st.info("No data available for the selected comparison dates.")

        # ---- Tab 2: Single Maturity History ----
        with tc_tabs[1]:
            st.caption("Track one maturity's yield across any date range — e.g. the 10-year from 2010 to present.")
            mc1, mc2, mc3 = st.columns([1, 1, 1])
            with mc1:
                sm_mat = st.selectbox("Maturity", TREASURY_MATURITIES,
                                      index=TREASURY_MATURITIES.index("10 Yr"), key="sm_maturity")
            with mc2:
                sm_start = st.date_input("Start", value=datetime(2010, 1, 1).date(),
                                         min_value=tdf["Date"].min().date(), max_value=tdf["Date"].max().date(),
                                         key="sm_start")
            with mc3:
                sm_end = st.date_input("End", value=tdf["Date"].max().date(),
                                       min_value=tdf["Date"].min().date(), max_value=tdf["Date"].max().date(),
                                       key="sm_end")
            fig_sm = treasury_maturity_history_figure(tdf, sm_mat, sm_start, sm_end)
            if fig_sm is not None:
                st.plotly_chart(fig_sm, use_container_width=True)
                sub = tdf[(tdf["Date"] >= pd.Timestamp(sm_start)) & (tdf["Date"] <= pd.Timestamp(sm_end))].dropna(subset=[sm_mat])
                if not sub.empty:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Latest", f"{sub[sm_mat].iloc[-1]:.2f}%")
                    m2.metric("Period High", f"{sub[sm_mat].max():.2f}%")
                    m3.metric("Period Low", f"{sub[sm_mat].min():.2f}%")
                    m4.metric("Period Avg", f"{sub[sm_mat].mean():.2f}%")
            else:
                st.info(f"No data available for {sm_mat} in the selected range (this maturity may not have existed yet).")

        # ---- Tab 3: Spread Tracking (up to two spreads) ----
        with tc_tabs[2]:
            st.caption("Track the spread between two maturities over time (long minus short), in basis points. "
                       "Add a second spread to compare, e.g. 2s10s vs 3m10y.")
            s1c1, s1c2 = st.columns(2)
            with s1c1:
                sp1_short = st.selectbox("Spread 1 — short leg", TREASURY_MATURITIES,
                                         index=TREASURY_MATURITIES.index("2 Yr"), key="sp1_short")
            with s1c2:
                sp1_long = st.selectbox("Spread 1 — long leg", TREASURY_MATURITIES,
                                        index=TREASURY_MATURITIES.index("10 Yr"), key="sp1_long")
            sdc1, sdc2 = st.columns(2)
            with sdc1:
                sp_start = st.date_input("Start", value=(tdf["Date"].max() - pd.DateOffset(years=6)).date(),
                                         min_value=tdf["Date"].min().date(), max_value=tdf["Date"].max().date(),
                                         key="sp_start")
            with sdc2:
                sp_end = st.date_input("End", value=tdf["Date"].max().date(),
                                       min_value=tdf["Date"].min().date(), max_value=tdf["Date"].max().date(),
                                       key="sp_end")
            add_sp2 = st.checkbox("Add a second spread to compare", key="sp2_toggle")
            spread2_spec = None
            if add_sp2:
                s2c1, s2c2 = st.columns(2)
                with s2c1:
                    sp2_short = st.selectbox("Spread 2 — short leg", TREASURY_MATURITIES,
                                             index=TREASURY_MATURITIES.index("3 Mo"), key="sp2_short")
                with s2c2:
                    sp2_long = st.selectbox("Spread 2 — long leg", TREASURY_MATURITIES,
                                            index=TREASURY_MATURITIES.index("10 Yr"), key="sp2_long")
                spread2_spec = {"short": sp2_short, "long": sp2_long, "label": f"{sp2_long} - {sp2_short}"}
            spread1_spec = {"short": sp1_short, "long": sp1_long, "label": f"{sp1_long} - {sp1_short}"}
            fig_sp = treasury_spread_figure(tdf, spread1_spec, spread2_spec, sp_start, sp_end)
            if fig_sp is not None:
                st.plotly_chart(fig_sp, use_container_width=True)
                st.caption("Negative = inverted (short yield above long yield). Zero line shown for reference.")
            else:
                st.info("No data available for the selected spread(s) and date range.")

    # ===== Economic Data (BLS — replaces FRED, added v15) =====
    st.divider()
    st.subheader("Economic Data")
    st.caption("Sourced directly from the U.S. Bureau of Labor Statistics — not FRED. Inflation shown year-over-year; "
               "payrolls/JOLTS/wages monthly. Jobless Claims and PCE are dropped here since neither is a BLS series "
               "(Claims is DOL/ETA data; PCE is BEA data) — this board is BLS-only.")
    if not _bls_key():
        with st.expander("⚙️ Add your BLS API key (optional — raises the daily query limit)"):
            st.markdown("The BLS Public API works without a key at a lower rate limit (25 queries/day, 10-year lookback). "
                        "A free registered key raises this to 500 queries/day and 20 years. Register at "
                        "[bls.gov/developers](https://www.bls.gov/developers/) and either add it to Streamlit secrets as "
                        "`BLS_API_KEY` (recommended — never exposed in chat or source), or paste it below for this session only.")
            _bk = st.text_input("BLS API key (this session only)", type="password", key="bls_key_input")
            if _bk:
                st.session_state["user_bls_key"] = _bk
                st.success("Key set for this session. Refresh the data below.")
    with st.spinner("Fetching BLS economic series..."):
        eco = bls_labor_inflation_board()
    if eco:
        labor = ["Nonfarm Payrolls (MoM)", "Unemployment Rate", "Job Openings (JOLTS)", "Avg Hourly Earnings (YoY)"]
        infl = ["CPI (YoY)", "Core CPI (YoY)", "PPI Final Demand (YoY)"]
        st.markdown("**Labor Market**")
        lp = [k for k in labor if k in eco]
        if lp:
            lc = st.columns(len(lp))
            for i, k in enumerate(lp):
                lc[i].metric(k.replace(" (MoM)", "").replace(" (YoY)", ""), eco[k]["value"], eco[k]["sub"], delta_color="off",
                             help=f"As of {eco[k]['asof']}")
        st.markdown("**Inflation**")
        ip = [k for k in infl if k in eco]
        if ip:
            ic = st.columns(len(ip))
            for i, k in enumerate(ip):
                ic[i].metric(k.replace(" (YoY)", ""), eco[k]["value"], eco[k]["sub"], delta_color="off",
                             help=f"As of {eco[k]['asof']}")
    else:
        st.info("BLS economic data temporarily unavailable.")

    st.divider()
    st.markdown("##### Trend Charts")
    st.caption("Historical trends for the metrics above, so direction and turning points are visible, not just the latest print.")
    trend_lookback = st.select_slider("Lookback", options=["1yr", "3yr", "5yr", "10yr"], value="5yr", key="bls_trend_lookback")
    lookback_months = {"1yr": 12, "3yr": 36, "5yr": 60, "10yr": 120}[trend_lookback]
    with st.spinner("Fetching BLS history..."):
        _hist_ids = [v[0] for v in BLS_SERIES.values()]
        _hist_raw = bls_fetch_history(tuple(_hist_ids), years_back=min(11, lookback_months // 12 + 1))
    trend_tabs = st.tabs(["Inflation", "Labor Market", "Payrolls", "Wages"])
    with trend_tabs[0]:
        fig_i = bls_inflation_trend_figure(_hist_raw, lookback_months)
        if fig_i is not None:
            st.plotly_chart(fig_i, use_container_width=True)
        else:
            st.info("Inflation trend data unavailable.")
    with trend_tabs[1]:
        fig_l = bls_labor_trend_figure(_hist_raw, lookback_months)
        if fig_l is not None:
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.info("Labor market trend data unavailable.")
    with trend_tabs[2]:
        fig_p = bls_payrolls_trend_figure(_hist_raw, lookback_months)
        if fig_p is not None:
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Payrolls trend data unavailable.")
    with trend_tabs[3]:
        fig_w = bls_wages_trend_figure(_hist_raw, lookback_months)
        if fig_w is not None:
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Wages trend data unavailable.")

    # ===== Sentiment =====
    st.divider()
    st.subheader("Sentiment Indicators")
    with st.spinner("Fetching sentiment..."):
        senti = fred_sentiment_board()
    if senti:
        sk = list(senti.keys())
        scol = st.columns(len(sk))
        for i, k in enumerate(sk):
            scol[i].metric(k, senti[k]["value"], senti[k].get("sub", ""), delta_color="off",
                           help=f"As of {senti[k].get('asof','')}" if senti[k].get('asof') else None)
        st.caption("CNN Fear & Greed is live (0=extreme fear, 100=extreme greed). Michigan sentiment from FRED. "
                   "Business/Consumer Confidence are OECD US indices (100=long-run average) used as proxies. "
                   "AAII and the Conference Board's official index are omitted — no reliable free feed.")
    else:
        st.info("Sentiment data temporarily unavailable.")

st.divider()
st.caption("DJR Trading System - Full Trading System with Persistent Storage")
