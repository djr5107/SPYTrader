# streamlit_app_ENHANCED.py
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

# V3.0 ENHANCEMENTS:
# 1. More liberal signal generation (less restrictive conditions)
# 2. Persistent signal timestamps (not reset on refresh)
# 3. Signal history tracking with expiration
# 4. Options signals generation
# 5. Continuation/scale-in signals for existing trends
# 6. Support/resistance signals
# 7. Trend following signals

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

st.set_page_config(page_title="SPY Pro v3.0-ENHANCED", layout="wide")
st.title("SPY Pro v3.0 - ENHANCED 🚀")
st.caption("✨ Active Signal Generation | Options Signals | Signal History | Persistent Timestamps")

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

# Signal expiration time (minutes)
SIGNAL_EXPIRATION_MINUTES = 30

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
        'P&L Numeric', 'DTE', 'Strategy', 'Thesis', 'Max Hold Minutes', 'Actual Hold Minutes',
        'Conviction', 'Signal Type'
    ])

if 'active_trades' not in st.session_state:
    st.session_state.active_trades = load_json(ACTIVE_TRADES_FILE, [])
    for trade in st.session_state.active_trades:
        if 'entry_time' in trade and isinstance(trade['entry_time'], str):
            trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])

if 'signal_queue' not in st.session_state:
    signal_queue_data = load_json(SIGNAL_QUEUE_FILE, [])
    # Convert string timestamps back to datetime objects
    for sig in signal_queue_data:
        if 'timestamp' in sig and isinstance(sig['timestamp'], str):
            sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
    st.session_state.signal_queue = signal_queue_data

if 'signal_history' not in st.session_state:
    history_data = load_json(SIGNAL_HISTORY_FILE, [])
    # Convert timestamps
    for sig in history_data:
        if 'timestamp' in sig and isinstance(sig['timestamp'], str):
            sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
        if 'expiration' in sig and isinstance(sig['expiration'], str):
            sig['expiration'] = datetime.fromisoformat(sig['expiration'])
    st.session_state.signal_history = history_data

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

# V3.0: Initialize macro analyzer
if 'macro_analyzer' not in st.session_state:
    st.session_state.macro_analyzer = MacroAnalyzer()
    try:
        st.session_state.macro_analyzer.fetch_macro_data()
    except:
        pass

# Settings (Sidebar)
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Signal Generation Settings
    st.subheader("Signal Generation")
    MIN_CONVICTION = st.slider("Min Conviction", 1, 10, 5, help="Minimum conviction level to show signals")
    ENABLE_MACRO_FILTER = st.checkbox("Macro Signal Filter", value=True, help="Use macro conditions to filter signals")
    USE_DYNAMIC_STOPS = st.checkbox("Dynamic Position Sizing", value=True, help="Adjust position size based on conviction")
    SIGNAL_EXPIRATION_MINUTES = st.slider("Signal Expiration (min)", 10, 120, 30, help="Minutes before signal expires")
    
    # V3.0: New signal types toggles
    st.subheader("Signal Types")
    ENABLE_CONTINUATION_SIGNALS = st.checkbox("Continuation Signals", value=True, help="Add to existing trends")
    ENABLE_SUPPORT_RESISTANCE = st.checkbox("Support/Resistance", value=True, help="Bounce signals at key levels")
    ENABLE_OPTIONS_SIGNALS = st.checkbox("Options Signals", value=True, help="Options trade opportunities")
    ENABLE_TREND_FOLLOWING = st.checkbox("Trend Following", value=True, help="Follow strong trends")
    
    # Stop Loss Settings
    st.subheader("Risk Management")
    STOP_LOSS_PCT = st.slider("Stop Loss %", -5.0, -0.5, -2.0, 0.5) / 100
    TRAILING_STOP_PCT = st.slider("Trailing Stop %", 0.5, 3.0, 1.0, 0.5) / 100
    
    st.divider()
    st.caption("SPY Pro v3.0 Enhanced")

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
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    
    # SMAs
    for period in [10, 20, 50, 100, 200]:
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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20
    
    # Support and Resistance levels (pivot points)
    if len(df) >= 5:
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    
    return df

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

def save_signal_queue():
    """Save signal queue with proper datetime serialization"""
    signals_to_save = []
    for sig in st.session_state.signal_queue:
        sig_copy = sig.copy()
        if isinstance(sig_copy.get('timestamp'), datetime):
            sig_copy['timestamp'] = sig_copy['timestamp'].isoformat()
        signals_to_save.append(sig_copy)
    save_json(SIGNAL_QUEUE_FILE, signals_to_save)

def save_signal_history():
    """Save signal history with proper datetime serialization"""
    history_to_save = []
    for sig in st.session_state.signal_history:
        sig_copy = sig.copy()
        if isinstance(sig_copy.get('timestamp'), datetime):
            sig_copy['timestamp'] = sig_copy['timestamp'].isoformat()
        if isinstance(sig_copy.get('expiration'), datetime):
            sig_copy['expiration'] = sig_copy['expiration'].isoformat()
        history_to_save.append(sig_copy)
    save_json(SIGNAL_HISTORY_FILE, history_to_save)

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
        save_signal_queue()
        save_signal_history()
    
    return len(expired)

# V3.0: ENHANCED SIGNAL GENERATION
def generate_signal():
    """
    Enhanced signal generation with:
    - More liberal conditions (not requiring TODAY's crossover)
    - Lower volume thresholds
    - Continuation signals
    - Support/resistance signals
    - Options signals
    - Trend following signals
    """
    now = datetime.now(ZoneInfo("US/Eastern"))
    now_str = now.strftime("%m/%d %H:%M")
    
    # Expire old signals first
    expire_old_signals()
    
    # V3.0: Fetch current macro regime
    try:
        regime = st.session_state.macro_analyzer.detect_regime()
        
        # Check VIX for volatility signals
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
            
            # ========================================
            # CATEGORY 1: SVXY VOLATILITY SIGNALS
            # ========================================
            
            if ticker == "SVXY" and len(df) >= 5:
                five_day_drop = ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
                
                # Signal 1: Vol Spike Recovery (8/10)
                if (five_day_drop < -8 and
                    volume_ratio > 1.1 and  # LOWERED from 1.3
                    price_change_pct > -1.0):
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
                        'thesis': f"VOL SPIKE RECOVERY: SVXY dropped {five_day_drop:.1f}% in 5 days (VIX spike). Mean reversion at ${current_price:.2f}. Vol {volume_ratio:.1f}x. VIX: {vix_current:.1f}",
                        'conviction': 8,
                        'signal_type': 'SVXY Vol Spike Recovery'
                    }
                
                # Signal 2: Sharp Drop Bounce (7/10)
                elif (len(df) >= 2 and
                      df['Close'].pct_change().iloc[-2] < -0.03 and
                      price_change_pct > 0.5 and
                      volume_ratio > 1.2):  # LOWERED from 1.5
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
                        'thesis': f"SHARP DROP RECOVERY: SVXY bouncing +{price_change_pct:.1f}% after yesterday's {df['Close'].pct_change().iloc[-2]*100:.1f}% drop. Vol {volume_ratio:.1f}x.",
                        'conviction': 7,
                        'signal_type': 'SVXY Sharp Drop Bounce'
                    }
            
            # ========================================
            # CATEGORY 2: SMA SIGNALS (MORE LIBERAL)
            # ========================================
            
            # V3.0: Changed to check if ALREADY crossed (not just TODAY)
            if not signal and (10 in sma_values and 20 in sma_values):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                
                # Check if 10 > 20 (already crossed) and price above both
                if (sma_10 > sma_20 and
                    current_price > sma_10 and
                    volume_ratio > 1.1):  # LOWERED from 1.2
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
                        'thesis': f"UPTREND CONFIRMED: {ticker} trading above SMA10 (${sma_10:.2f}) and SMA20 (${sma_20:.2f}). Price ${current_price:.2f}, Vol {volume_ratio:.1f}x, RSI {rsi:.0f}.",
                        'conviction': 7,
                        'signal_type': 'SMA Uptrend'
                    }
            
            # Golden Cross (already crossed)
            if not signal and (50 in sma_values and 200 in sma_values):
                sma_50 = sma_values[50]
                sma_200 = sma_values[200]
                
                if (sma_50 > sma_200 and
                    current_price > sma_50 and
                    macd > 0 and
                    adx > 15):  # LOWERED from 20
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
                        'thesis': f"GOLDEN CROSS TREND: {ticker} in golden cross (50>200). Price ${current_price:.2f} above SMA50 (${sma_50:.2f}), MACD bullish, ADX {adx:.1f}.",
                        'conviction': 9,
                        'signal_type': 'Golden Cross'
                    }
            
            # ========================================
            # CATEGORY 3: VOLUME & MOMENTUM SIGNALS
            # ========================================
            
            # V3.0: Lower volume threshold
            if not signal and (20 in sma_values):
                sma_20 = sma_values[20]
                
                if (volume_ratio > 1.5 and  # LOWERED from 2.0
                    price_change_pct > 0.5 and
                    current_price > sma_20 and
                    rsi > 50 and rsi < 75):
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
                        'thesis': f"VOLUME BREAKOUT: {ticker} ${current_price:.2f} with strong volume ({volume_ratio:.1f}x), +{price_change_pct:.1f}% move. RSI {rsi:.0f}, above SMA20.",
                        'conviction': 7,
                        'signal_type': 'Volume Breakout'
                    }
            
            # ========================================
            # CATEGORY 4: NEW - CONTINUATION SIGNALS
            # ========================================
            
            if not signal and ENABLE_CONTINUATION_SIGNALS and (10 in sma_values and 20 in sma_values and 50 in sma_values):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
                
                # Strong uptrend: 10 > 20 > 50, price pulled back to 10-day
                price_near_10_sma = abs(current_price - sma_10) / sma_10 < 0.02  # Within 2%
                
                if (sma_10 > sma_20 > sma_50 and
                    price_near_10_sma and
                    rsi > 40 and rsi < 60):
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
                        'thesis': f"TREND CONTINUATION: {ticker} in strong uptrend (10>20>50), pullback to 10-day SMA (${sma_10:.2f}). Add at ${current_price:.2f}, RSI {rsi:.0f}.",
                        'conviction': 8,
                        'signal_type': 'Continuation'
                    }
            
            # ========================================
            # CATEGORY 5: NEW - SUPPORT/RESISTANCE
            # ========================================
            
            if not signal and ENABLE_SUPPORT_RESISTANCE and 'S1' in df.columns:
                s1 = df['S1'].iloc[-1]
                r1 = df['R1'].iloc[-1]
                
                # Bounce off support
                price_near_support = abs(current_price - s1) / s1 < 0.005  # Within 0.5%
                
                if (price_near_support and
                    price_change_pct > 0.3 and
                    rsi > 30 and rsi < 50):
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
                        'thesis': f"SUPPORT BOUNCE: {ticker} bouncing at S1 support (${s1:.2f}). Price ${current_price:.2f}, RSI {rsi:.0f}, +{price_change_pct:.1f}% reversal.",
                        'conviction': 6,
                        'signal_type': 'Support Bounce'
                    }
            
            # ========================================
            # CATEGORY 6: NEW - TREND FOLLOWING
            # ========================================
            
            if not signal and ENABLE_TREND_FOLLOWING and (10 in sma_values and 20 in sma_values and 50 in sma_values and 200 in sma_values):
                sma_10 = sma_values[10]
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
                sma_200 = sma_values[200]
                
                # All SMAs aligned + price above all
                all_aligned = (current_price > sma_10 > sma_20 > sma_50 > sma_200)
                
                if (all_aligned and
                    adx > 20 and
                    rsi > 50 and rsi < 70):
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
                        'thesis': f"STRONG TREND: {ticker} all SMAs aligned (Price>${sma_10:.2f}>${sma_20:.2f}>${sma_50:.2f}>${sma_200:.2f}). ADX {adx:.1f}, RSI {rsi:.0f}. Momentum continues.",
                        'conviction': 9,
                        'signal_type': 'Strong Trend'
                    }
            
            # ========================================
            # CATEGORY 7: MEAN REVERSION
            # ========================================
            
            if not signal and (rsi < 30 and
                  price_change_pct > 0.2 and
                  20 in sma_values):
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
                    'thesis': f"OVERSOLD BOUNCE: {ticker} ${current_price:.2f} reversing from oversold (RSI {rsi:.0f}). +{price_change_pct:.1f}% bounce. SMA20 at ${sma_20:.2f}.",
                    'conviction': 6,
                    'signal_type': 'Oversold Bounce'
                }
            
            # ========================================
            # CATEGORY 8: NEW - OPTIONS SIGNALS
            # ========================================
            
            if not signal and ENABLE_OPTIONS_SIGNALS:
                # Get options chain
                try:
                    options_df = get_options_chain(ticker, dte_min=14, dte_max=45)
                    
                    if not options_df.empty and 20 in sma_values:
                        sma_20 = sma_values[20]
                        
                        # High IV + Support = Sell put spreads
                        if 'impliedVolatility' in options_df.columns:
                            avg_iv = options_df[options_df['type'] == 'Put']['impliedVolatility'].mean()
                            
                            if avg_iv > 0.25 and current_price > sma_20 and rsi > 45:
                                # Find ATM put
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
                                    'thesis': f"HIGH IV OPPORTUNITY: {ticker} IV={avg_iv:.1%} elevated, price ${current_price:.2f} above support (SMA20 ${sma_20:.2f}). Sell ${atm_put['strike']:.0f}P DTE {int(atm_put['dte'])}.",
                                    'conviction': 7,
                                    'signal_type': 'Options Put Spread'
                                }
                except:
                    pass
            
            # Apply macro filter and conviction threshold
            if signal:
                if ENABLE_MACRO_FILTER:
                    signal = adjust_signal_for_macro(signal, regime, df)
                
                if signal and should_take_signal(signal, MIN_CONVICTION):
                    if USE_DYNAMIC_STOPS:
                        signal['size'] = calculate_position_size(signal)
                        if 'shares' in signal['action']:
                            signal['action'] = f"BUY {signal['size']} shares @ ${signal['entry_price']:.2f}"
                    
                    st.session_state.signal_queue.append(signal)
                    save_signal_queue()
                    
                    # Also add to history immediately (as active)
                    sig_history = signal.copy()
                    sig_history['status'] = 'Active'
                    st.session_state.signal_history.append(sig_history)
                    save_signal_history()
                    
                    break  # Only one signal per cycle
                    
        except Exception as e:
            continue

def save_active_trades():
    trades_to_save = []
    for trade in st.session_state.active_trades:
        trade_copy = trade.copy()
        if isinstance(trade_copy.get('entry_time'), datetime):
            trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
        trades_to_save.append(trade_copy)
    save_json(ACTIVE_TRADES_FILE, trades_to_save)

# Auto-Exit Logic
def simulate_exit():
    """Exit trades based on stop loss, trailing stop, momentum reversal, or SMA break"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    
    for trade in st.session_state.active_trades[:]:
        current_price = market_data[trade['symbol']]['price']
        entry_price = trade['entry_price']
        
        # Calculate P&L percentage
        if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        exit_triggered = False
        exit_reason = ""
        
        # Exit Rule 1: Stop Loss
        if pnl_pct <= STOP_LOSS_PCT * 100:
            exit_triggered = True
            exit_reason = f"Stop Loss ({STOP_LOSS_PCT*100:.1f}%)"
        
        # Exit Rule 2: Trailing Stop
        elif pnl_pct >= 4.0:
            max_pnl_reached = trade.get('max_pnl_reached', pnl_pct)
            if pnl_pct > max_pnl_reached:
                trade['max_pnl_reached'] = pnl_pct
            elif (max_pnl_reached - pnl_pct) >= TRAILING_STOP_PCT * 100:
                exit_triggered = True
                exit_reason = f"Trailing Stop (from +{max_pnl_reached:.1f}% to +{pnl_pct:.1f}%)"
        
        # Execute exit if triggered
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
    
    save_active_trades()

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

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Trading Hub", "Signal History", "Trade Log", "Performance", "Chart Analysis", "Options Chain"],
    icons=["activity", "clock-history", "list-ul", "graph-up", "bar-chart", "currency-exchange"],
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
    
    # Market Status
    market_open = is_market_open()
    st.info(f"""
    **Market Status:** {'🟢 OPEN' if market_open else '🔴 CLOSED'}  
    **Current Time (ET):** {datetime.now(ZoneInfo("US/Eastern")).strftime('%I:%M:%S %p')}  
    **Active Signals:** {len(st.session_state.signal_queue)} signals  
    **Signal Generation:** {'✅ Active' if market_open else '❌ Paused (market closed)'}  
    **Signal Expiration:** {SIGNAL_EXPIRATION_MINUTES} minutes
    """)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Generate Signals", use_container_width=True):
            generate_signal()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Signals", use_container_width=True):
            st.session_state.signal_queue = []
            save_signal_queue()
            st.rerun()
    with col3:
        expired_count = expire_old_signals()
        if expired_count > 0:
            st.success(f"Expired {expired_count} old signals")
    
    st.divider()
    
    # Auto-generate signals if market open
    if market_open:
        generate_signal()
        simulate_exit()
    
    # Display Signals
    st.subheader(f"📊 Trading Signals ({len(st.session_state.signal_queue)} Active)")
    
    if len(st.session_state.signal_queue) == 0:
        st.info(f"""
        **No signals currently active.**
        
        Signals will appear when market conditions align:
        - ✅ SMA trends and crossovers
        - ✅ Volume spikes and breakouts
        - ✅ Support/resistance bounces
        - ✅ Trend continuation setups
        - ✅ Options opportunities (high IV)
        
        **Try:**
        - Click "🔄 Generate Signals" button
        - Lower "Min Conviction" in sidebar (try 4-5)
        - Enable all signal types in sidebar
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
            if st.button(f"✅ Take Trade: {sig['id']}", key=f"take_{sig['id']}", use_container_width=True):
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
                save_active_trades()
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
                save_signal_history()
                
                st.session_state.signal_queue.remove(sig)
                save_signal_queue()
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
                save_signal_history()
                
                st.session_state.signal_queue.remove(sig)
                save_signal_queue()
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
                    save_active_trades()
                    st.success("Trade closed!")
                    st.rerun()

# ========================================
# SIGNAL HISTORY PAGE
# ========================================

elif selected == "Signal History":
    st.header("📜 Signal History")
    
    if not st.session_state.signal_history:
        st.info("No signal history yet. Signals will be tracked here once generated.")
    else:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", ["Active", "Taken", "Skipped", "Expired"], default=["Active", "Taken", "Skipped", "Expired"])
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
        
        st.write(f"**Showing {len(filtered_signals)} signals from last {days_back} days**")
        
        # Display as table
        if filtered_signals:
            history_df = pd.DataFrame([{
                'Time': sig['time'],
                'Symbol': sig['symbol'],
                'Type': sig['signal_type'],
                'Action': sig['action'],
                'Conviction': sig['conviction'],
                'Status': sig.get('status', 'Unknown'),
                'Age (min)': int((datetime.now(ZoneInfo("US/Eastern")) - sig['timestamp']).total_seconds() / 60) if 'timestamp' in sig else 0
            } for sig in filtered_signals])
            
            st.dataframe(history_df, use_container_width=True)
            
            # Export button
            if st.button("📥 Export to CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"signal_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ========================================
# TRADE LOG PAGE
# ========================================

elif selected == "Trade Log":
    st.header("📋 Trade Log")
    
    if st.session_state.trade_log.empty:
        st.info("No trades logged yet.")
    else:
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show All Trades", value=True)
        with col2:
            if st.button("📥 Export Trade Log"):
                csv = st.session_state.trade_log.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        # Display trade log
        if show_all:
            st.dataframe(st.session_state.trade_log, use_container_width=True)
        else:
            st.dataframe(st.session_state.trade_log.tail(50), use_container_width=True)

# ========================================
# PERFORMANCE PAGE
# ========================================

elif selected == "Performance":
    st.header("📊 Performance Metrics")
    
    if st.session_state.trade_log.empty:
        st.info("No trades to analyze yet.")
    else:
        # Calculate metrics from closed trades
        closed_trades = st.session_state.trade_log[st.session_state.trade_log['Type'] == 'Close']
        
        if not closed_trades.empty:
            total_pnl = closed_trades['P&L Numeric'].sum()
            winning_trades = len(closed_trades[closed_trades['P&L Numeric'] > 0])
            losing_trades = len(closed_trades[closed_trades['P&L Numeric'] <= 0])
            total_trades = len(closed_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = closed_trades[closed_trades['P&L Numeric'] > 0]['P&L Numeric'].mean() if winning_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['P&L Numeric'] <= 0]['P&L Numeric'].mean() if losing_trades > 0 else 0
            
            # Display metrics
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
                name='Cumulative P&L',
                line=dict(color='green' if total_pnl > 0 else 'red', width=2)
            ))
            fig.update_layout(
                title="Cumulative P&L Over Time",
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
# CHART ANALYSIS PAGE
# ========================================

elif selected == "Chart Analysis":
    st.header("📈 Chart Analysis")
    
    ticker_choice = st.selectbox("Select Ticker", TICKERS)
    
    try:
        t = yf.Ticker(ticker_choice)
        hist = t.history(period="6mo", interval="1d")
        
        if not hist.empty:
            df = calculate_technical_indicators(hist)
            
            # Create candlestick chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{ticker_choice} Price', 'RSI', 'Volume')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ), row=1, col=1)
            
            # Add SMAs
            for period, color in [(20, 'orange'), (50, 'blue'), (200, 'purple')]:
                if f'SMA_{period}' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[f'SMA_{period}'],
                        name=f'SMA {period}',
                        line=dict(color=color, width=1)
                    ), row=1, col=1)
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ), row=2, col=1)
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ), row=3, col=1)
            
            fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Current stats
            st.subheader("Current Technical Stats")
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1.0
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
            
            col1.metric("Price", f"${current_price:.2f}")
            col2.metric("RSI", f"{rsi:.1f}")
            col3.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            col4.metric("ADX", f"{adx:.1f}")
            
    except Exception as e:
        st.error(f"Error loading chart: {e}")

# ========================================
# OPTIONS CHAIN PAGE
# ========================================

elif selected == "Options Chain":
    st.header("💱 Options Chain")
    
    ticker_choice = st.selectbox("Select Ticker", TICKERS, key="opt_ticker")
    
    col1, col2 = st.columns(2)
    with col1:
        dte_min = st.number_input("Min DTE", value=14, min_value=1, max_value=365)
    with col2:
        dte_max = st.number_input("Max DTE", value=45, min_value=1, max_value=365)
    
    if st.button("Load Options Chain"):
        with st.spinner("Loading options..."):
            options_df = get_options_chain(ticker_choice, dte_min, dte_max)
            
            if not options_df.empty:
                st.success(f"Loaded {len(options_df)} options contracts")
                
                # Filter
                option_type = st.radio("Type", ["Calls", "Puts", "Both"], horizontal=True)
                
                if option_type == "Calls":
                    filtered = options_df[options_df['type'] == 'Call']
                elif option_type == "Puts":
                    filtered = options_df[options_df['type'] == 'Put']
                else:
                    filtered = options_df
                
                # Display key columns
                display_cols = ['strike', 'type', 'dte', 'expiration', 'mid', 'impliedVolatility', 'volume', 'openInterest', 'delta', 'gamma', 'theta', 'vega']
                available_cols = [col for col in display_cols if col in filtered.columns]
                
                st.dataframe(filtered[available_cols].sort_values('dte'), use_container_width=True)
            else:
                st.warning("No options data available for this ticker")

st.divider()
st.caption("SPY Pro v3.0 Enhanced - Active Signal Generation with History Tracking")
