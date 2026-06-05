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
    try:
        import urllib.request as _ur, json as _js
        body = _js.dumps({
            "model": model,
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = _ur.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json", "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"})
        with _ur.urlopen(req, timeout=45) as r:
            resp = _js.loads(r.read())
        text = "".join(p.get("text", "") for p in resp.get("content", []) if p.get("type") == "text")
        return data, text or "No response."
    except Exception as e:
        return data, f"Claude call failed: {e}"
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
HOME_TAPE = [
    ("ES=F", "S&P Fut"), ("YM=F", "Dow Fut"), ("NQ=F", "Nasdaq Fut"),
    ("RTY=F", "Rus 2K Fut"), ("^VIX", "VIX"), ("GC=F", "Gold"),
    ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("^TNX", "10Y Yield"),
    ("DX-Y.NYB", "Dollar"),
]

@st.cache_data(ttl=120)
def home_tape_quotes():
    """Compact last/change/%-change for the tape. Returns list of dicts."""
    out = []
    syms = [s for s, _ in HOME_TAPE]
    try:
        data = yf.download(syms, period="2d", interval="1d", progress=False, auto_adjust=False, group_by="ticker")
    except Exception:
        data = None
    for sym, label in HOME_TAPE:
        last = chg = pct = None
        try:
            if data is not None:
                if isinstance(data.columns, pd.MultiIndex):
                    s = data[sym]["Close"].dropna()
                else:
                    s = data["Close"].dropna()
                if len(s) >= 2:
                    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                    chg = last - prev
                    pct = chg / prev * 100 if prev else None
                elif len(s) == 1:
                    last = float(s.iloc[-1])
        except Exception:
            pass
        # ^TNX is yield*10 on some feeds; normalize to a yield-looking number
        disp = last
        if sym == "^TNX" and last is not None and last > 100:
            disp = last / 10.0
        out.append({"label": label, "last": disp, "chg": chg, "pct": pct})
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

# ---- Derived Fed rate-move probabilities from Fed Funds futures ----
# ---------- Meeting-weighted Fed probabilities (CME FedWatch methodology) ----------
# 2026 FOMC decision dates (second day of each meeting). Update annually.
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16"]

def _next_fomc(today=None):
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    for d in FOMC_2026:
        dd = datetime.strptime(d, "%Y-%m-%d").date()
        if dd >= today:
            return dd
    return None

@st.cache_data(ttl=3600)
def home_fed_probabilities():
    """Probabilities for the NEXT FOMC meeting using the CME FedWatch methodology:
    a 30-day Fed Funds future settles to the month's average daily effective rate,
    so the meeting-month contract blends the pre- and post-meeting rate by the
    number of days each is in effect. We solve that blend for the implied
    post-meeting rate, compare to the current target midpoint, and convert the
    expected change into probabilities across the nearest 25 bps increments.

    This is the same math CME publishes; it is computed here from the futures
    curve (CME exposes no free API). The current target midpoint is inferred from
    the front contract when no meeting falls earlier in the current month."""
    try:
        import calendar as _cal
        meeting = _next_fomc()
        if meeting is None:
            return {}
        zq = yf.Ticker("ZQ=F")
        h = zq.history(period="10d", interval="1d")
        if h.empty:
            return {}
        front_price = float(h["Close"].iloc[-1])
        front_implied = 100.0 - front_price  # avg effective rate implied for the front month

        # Infer current target midpoint: round the front implied to the nearest 25 bps.
        cur_mid = round(front_implied * 4) / 4

        # Intra-month weighting for the meeting month.
        N = _cal.monthrange(meeting.year, meeting.month)[1]
        d_eff = meeting.day + 1          # new rate effective the day after the decision
        days_old = max(0, d_eff - 1)
        days_new = max(1, N - days_old)

        # If the next meeting is in the current (front) month, solve the front
        # contract's monthly average for the post-meeting rate.
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if meeting.month == today.month and meeting.year == today.year:
            month_avg = front_implied
            post_rate = (month_avg * N - days_old * cur_mid) / days_new
        else:
            # Meeting is in a later month: the front contract reflects the current
            # rate; use the implied forward as the post-meeting estimate.
            post_rate = front_implied

        move = (post_rate - cur_mid) * 100.0  # bps, signed (negative = cut)

        # Convert expected move into probabilities across 25 bps increments.
        step = 25.0
        steps = move / step  # e.g. -0.6 => 60% of a 25bp cut priced
        probs = {"cut25": 0.0, "hold": 0.0, "hike25": 0.0, "cut50": 0.0, "hike50": 0.0}
        if steps <= -1:
            # between a 25 and 50 bps cut
            frac = min(1.0, -steps - 1.0)
            probs["cut50"] = round(frac * 100)
            probs["cut25"] = round((1 - frac) * 100)
        elif steps < 0:
            probs["cut25"] = round(-steps * 100)
            probs["hold"] = round((1 + steps) * 100)
        elif steps == 0:
            probs["hold"] = 100
        elif steps < 1:
            probs["hike25"] = round(steps * 100)
            probs["hold"] = round((1 - steps) * 100)
        else:
            frac = min(1.0, steps - 1.0)
            probs["hike50"] = round(frac * 100)
            probs["hike25"] = round((1 - frac) * 100)

        return {"meeting": meeting.strftime("%b %d, %Y"),
                "implied_rate": round(post_rate, 3), "target_mid": round(cur_mid, 3),
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
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ Markets"),
    ("https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "WSJ Business"),
    ("https://feeds.a.dj.com/rss/RSSWorldNews.xml", "WSJ World"),
    ("https://feeds.a.dj.com/rss/RSSWSJD.xml", "WSJ Tech"),
    ("https://www.economist.com/finance-and-economics/rss.xml", "Economist"),
    ("https://www.economist.com/business/rss.xml", "Economist Business"),
    ("https://www.cnbc.com/id/15839135/device/rss/rss.html", "CNBC Markets"),
    ("https://www.cnbc.com/id/20910258/device/rss/rss.html", "CNBC Economy"),
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

def home_news_categorized(lens, per_bucket=4, max_age_hours=24):
    """Bucket fresh (<=24h) headlines by keyword. Factors allow multi-bucket;
    Asset Classes assign each headline to its single best class."""
    pool = home_news_pool(max_age_hours)
    kw = NEWS_FACTOR_KW if lens == "Factors" else NEWS_ASSET_KW
    out = {b: [] for b in kw}
    if lens == "Factors":
        for title, source, link, when in pool:
            low = title.lower()
            for b, words in kw.items():
                if len(out[b]) >= per_bucket:
                    continue
                if any(w in low for w in words):
                    out[b].append((title, source, link, when))
    else:
        for title, source, link, when in pool:
            low = title.lower()
            best, best_score = None, 0
            for b, words in kw.items():
                score = sum(1 for w in words if w in low)
                if score > best_score:
                    best, best_score = b, score
            if best and len(out[best]) < per_bucket:
                out[best].append((title, source, link, when))
    return out

# ---------- FRED: official API (key) with CSV fallback ----------
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
    st.markdown(
        "<div style='background:#0e1117;border:1px solid #1f2530;border-radius:8px;"
        "padding:8px 8px;overflow-x:auto;white-space:nowrap;margin-bottom:16px'>"
        + "".join(chips) + "</div>", unsafe_allow_html=True)

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
        # Fed probabilities (CME FedWatch methodology, computed from futures)
        fed = home_fed_probabilities()
        if fed:
            st.markdown(f"**FOMC {fed.get('meeting','')} — implied** <span style='color:#8a93a3;font-size:.7rem'>(CME method, from Fed Funds futures)</span>", unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Cut 25", f"{fed['p_cut']}%")
            fc2.metric("Hold", f"{fed['p_hold']}%")
            fc3.metric("Hike 25", f"{fed['p_hike']}%")
            if fed.get('p_cut50') or fed.get('p_hike50'):
                st.caption(f"Tail: 50 bps cut {fed.get('p_cut50',0)}% · 50 bps hike {fed.get('p_hike50',0)}%")
            st.caption(f"Implies ~{fed['implied_rate']:.2f}% post-meeting vs ~{fed['target_mid']:.2f}% now "
                       f"({fed['move_bps']:+.0f} bps). Meeting-date-weighted; derived, not CME's published figure.")
        else:
            st.info("Fed-futures probabilities temporarily unavailable.")

        st.markdown("**Treasury Yield Curve**")
        yc = fred_yield_curve()
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

                # ----- Technical scorecard (buy/sell/hold) -----
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
        st.caption(f"Inception date: **{AI_STRATEGY_INCEPTION}** · Benchmark: Nasdaq-100 (NDX) · Showing current holdings only (target weight > 0).")

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

        per_current = [p for p in per if (p.get('target_weight') or 0) > 0]
        df = pd.DataFrame(per_current)
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
                user_key = st.text_input("Anthropic API key (sk-ant-...)", type="password", key="em_api_key")
            with _col_model:
                _model_choice = st.selectbox("Model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
                                             key="em_model", help="Haiku = faster/cheaper; Sonnet = deeper analysis")
            if st.button(f"📊 Analyse {pick} earnings", key="em_analyse_btn", use_container_width=True):
                if not user_key:
                    st.warning("Paste your API key above to run the analysis.")
                else:
                    with st.spinner(f"Gathering {pick} data and running analysis..."):
                        edata, analysis_text = em_earnings_analysis(user_key, pick, model=_model_choice)
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
        yc = fred_yield_curve()
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
        fed = home_fed_probabilities()
        if fed:
            st.caption(f"Next meeting: {fed.get('meeting','')}")
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Cut 25", f"{fed['p_cut']}%")
            fc2.metric("Hold", f"{fed['p_hold']}%")
            fc3.metric("Hike 25", f"{fed['p_hike']}%")
            if fed.get('p_cut50') or fed.get('p_hike50'):
                st.caption(f"Tail risk: 50 bps cut {fed.get('p_cut50',0)}% · 50 bps hike {fed.get('p_hike50',0)}%")
            st.caption(f"The meeting-month Fed Funds future implies ~{fed['implied_rate']:.2f}% post-meeting effective rate "
                       f"vs ~{fed['target_mid']:.2f}% now ({fed['move_bps']:+.0f} bps). Computed with the CME FedWatch "
                       "methodology (meeting-date-weighted futures), but derived here from the futures curve, not CME's "
                       "published figure (CME exposes no free API).")
            st.markdown("[CME FedWatch (official)](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html) · "
                        "[Treasury yield curve](https://www.ustreasuryyieldcurve.com/)")
        else:
            st.info("Fed-futures data temporarily unavailable.")

    # ===== Economic Data (FRED) =====
    st.divider()
    st.subheader("Economic Data")
    st.caption("Latest releases from FRED (Federal Reserve Bank of St. Louis). Inflation shown year-over-year; claims weekly; payrolls/JOLTS monthly.")
    if not _fred_key():
        with st.expander("⚙️ FRED data not loading? Add a free API key"):
            st.markdown("FRED's no-key CSV endpoint can be throttled on shared cloud IPs. For reliable data, get a "
                        "free key at [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys) and either "
                        "add it to Streamlit secrets as `FRED_API_KEY`, or paste it below for this session.")
            _k = st.text_input("FRED API key (this session only)", type="password", key="fred_key_input")
            if _k:
                st.session_state["user_fred_key"] = _k
                st.success("Key set for this session. Refresh the data below.")
    with st.spinner("Fetching FRED economic series..."):
        eco = fred_macro_board()
    if eco:
        labor = ["Initial Jobless Claims", "Continuing Claims", "Nonfarm Payrolls (MoM)", "Unemployment Rate", "Job Openings (JOLTS)"]
        infl = ["CPI (YoY)", "Core CPI (YoY)", "PPI Final Demand (YoY)", "PCE (YoY)", "Core PCE (YoY)"]
        st.markdown("**Labor Market**")
        lp = [k for k in labor if k in eco]
        if lp:
            lc = st.columns(len(lp))
            for i, k in enumerate(lp):
                lc[i].metric(k.replace(" (MoM)", ""), eco[k]["value"], eco[k]["sub"], delta_color="off",
                             help=f"As of {eco[k]['asof']}")
        st.markdown("**Inflation**")
        ip = [k for k in infl if k in eco]
        if ip:
            ic = st.columns(len(ip))
            for i, k in enumerate(ip):
                ic[i].metric(k.replace(" (YoY)", ""), eco[k]["value"], eco[k]["sub"], delta_color="off",
                             help=f"As of {eco[k]['asof']}")
    else:
        st.info("FRED economic data temporarily unavailable.")

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
