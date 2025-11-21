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

st.set_page_config(page_title="SPY Pro v3.0-COMPLETE", layout="wide")
st.title("SPY Pro v3.0 - COMPLETE 🚀")
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
    st.caption(f"SPY Pro v3.0 | {TRADING_MODE}")
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
    options=["Trading Hub", "Market Dashboard", "Signal History", "Backtest", "Trade Log", "Performance", "Chart Analysis", "Options Chain", "Macro Dashboard"],
    icons=["activity", "speedometer2", "clock-history", "graph-up-arrow", "list-ul", "trophy", "bar-chart", "currency-exchange", "globe"],
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
        
        # Get the most recent trading day
        def get_last_trading_day():
            """Get the last valid trading day accounting for market hours, weekends, and holidays"""
            now = datetime.now(ZoneInfo("US/Eastern"))
            check_date = now
            
            # If before 4pm today, use yesterday. If after 4pm, can use today if it's a trading day
            if now.hour < 16:
                check_date = now - timedelta(days=1)
            
            # Now go back to find last valid trading day
            for _ in range(10):  # Check up to 10 days back
                # Skip weekends
                if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    check_date = check_date - timedelta(days=1)
                    continue
                
                # Check if it's a US market holiday
                us_holidays = [
                    datetime(2024, 1, 1),   # New Year's Day
                    datetime(2024, 1, 15),  # MLK Day
                    datetime(2024, 2, 19),  # Presidents Day
                    datetime(2024, 3, 29),  # Good Friday
                    datetime(2024, 5, 27),  # Memorial Day
                    datetime(2024, 6, 19),  # Juneteenth
                    datetime(2024, 7, 4),   # Independence Day
                    datetime(2024, 9, 2),   # Labor Day
                    datetime(2024, 11, 28), # Thanksgiving
                    datetime(2024, 12, 25), # Christmas
                    datetime(2025, 1, 1),   # New Year's Day
                    datetime(2025, 1, 20),  # MLK Day
                    datetime(2025, 2, 17),  # Presidents Day
                    datetime(2025, 4, 18),  # Good Friday
                    datetime(2025, 5, 26),  # Memorial Day
                    datetime(2025, 6, 19),  # Juneteenth
                    datetime(2025, 7, 4),   # Independence Day
                    datetime(2025, 9, 1),   # Labor Day
                    datetime(2025, 11, 27), # Thanksgiving
                    datetime(2025, 12, 25)  # Christmas
                ]
                
                if check_date.date() in [h.date() for h in us_holidays]:
                    check_date = check_date - timedelta(days=1)
                    continue
                
                # Found a valid trading day
                return check_date
            
            return now
        
        last_trading_day = get_last_trading_day()
        
        for category, tickers in tickers_dict.items():
            category_data = []
            
            for name, ticker in tickers.items():
                try:
                    # Fetch enough data for longest period
                    t = yf.Ticker(ticker)
                    # Get more history to ensure we have enough data
                    hist = t.history(period="max")
                    
                    if hist.empty:
                        continue
                    
                    # Filter to only data up to last trading day
                    hist = hist[hist.index.date <= last_trading_day.date()]
                    
                    if len(hist) < 2:
                        continue
                    
                    row_data = {'Name': name, 'ETF': ticker}
                    
                    # Calculate returns for each period
                    for period_name, period_value in STANDARD_PERIODS.items():
                        try:
                            if period_value == "mtd":
                                # Month to date - first day of month containing last_trading_day
                                period_start = last_trading_day.replace(day=1)
                                period_hist = hist[hist.index >= period_start]
                            elif period_value == "ytd":
                                # Year to date - first day of year containing last_trading_day
                                period_start = last_trading_day.replace(month=1, day=1)
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
        """Create table showing all periods with clickable ETF tickers"""
        st.markdown(f"### {title}")
        
        # Create HTML table
        periods = ["Today", "MTD", "YTD", "1yr", "3yr", "5yr", "10yr"]
        period_years = {"Today": 0, "MTD": 0, "YTD": 0, "1yr": 1, "3yr": 3, "5yr": 5, "10yr": 10}
        
        html = '<table style="width:100%; border-collapse: collapse; font-size:14px;">'
        
        # Header row
        html += '<tr style="background:#1a1a1a; border-bottom: 2px solid #404040;">'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">Name</th>'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">ETF</th>'
        for period in periods:
            html += f'<th style="padding:12px; text-align:right; color:#FFFFFF; {"border-right:1px solid #404040;" if period != periods[-1] else ""}">{period}</th>'
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
        
        # Add clickable ETF selector below table
        st.write("")
        etf_list = df['ETF'].tolist()
        selected_etf = st.selectbox(
            f"Analyze {title.split()[1]} ETF in detail:",
            ["Select an ETF..."] + etf_list,
            key=f"select_{title}"
        )
        
        if selected_etf != "Select an ETF...":
            if st.button(f"📊 View {selected_etf} Chart Analysis", key=f"btn_{title}_{selected_etf}"):
                st.session_state['chart_ticker'] = selected_etf
                st.session_state['nav_to_chart'] = True
                st.rerun()
        
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
    st.header("📈 Chart Analysis & Technical Assessment")
    
    # Check if navigated from Market Dashboard
    if 'nav_to_chart' in st.session_state and st.session_state['nav_to_chart']:
        default_ticker = st.session_state.get('chart_ticker', 'SPY')
        st.session_state['nav_to_chart'] = False
    else:
        default_ticker = 'SPY'
    
    # Build ticker list from all Market Dashboard categories
    all_etfs = ['SPY']  # Start with SPY
    for category_etfs in MARKET_ETFS.values():
        all_etfs.extend(list(category_etfs.values()))
    all_etfs = sorted(list(set(all_etfs)))  # Remove duplicates and sort
    
    # Find index of default ticker
    try:
        default_index = all_etfs.index(default_ticker)
    except:
        default_index = 0
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker_choice = st.selectbox("Select Ticker", all_etfs, index=default_index)
    with col2:
        period_type = st.radio("Period Type", ["Standard", "Custom"], horizontal=True)
    
    if period_type == "Standard":
        with col3:
            time_period = st.selectbox("Time Period", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "Max"], index=5)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            custom_start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=180))
        with col_b:
            custom_end_date = st.date_input("End Date", value=datetime.now())
    
    try:
        t = yf.Ticker(ticker_choice)
        
        # Fetch data based on period selection
        if period_type == "Standard":
            period_map = {
                "1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo",
                "6M": "6mo", "YTD": "ytd", "1Y": "1y", "2Y": "2y",
                "5Y": "5y", "Max": "max"
            }
            hist = t.history(period=period_map[time_period], interval="1d")
        else:
            hist = t.history(start=custom_start_date, end=custom_end_date, interval="1d")
        
        if not hist.empty:
            df = calculate_technical_indicators(hist)
            
            # Current values
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
            
            # Display current price prominently
            st.markdown(f"""
            <div style="background:#1a1a1a; padding:20px; border-radius:10px; margin-bottom:20px;">
                <h2 style="color:#FFFFFF; margin:0;">{ticker_choice}</h2>
                <h1 style="color:{'#00ff00' if price_change >= 0 else '#ff0000'}; margin:10px 0;">${current_price:.2f}</h1>
                <h3 style="color:{'#00ff00' if price_change >= 0 else '#ff0000'}; margin:0;">{price_change:+.2f} ({price_change_pct:+.2f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create candlestick chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{ticker_choice} Price & Moving Averages', 'RSI (14)', 'Volume')
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
            sma_colors = {20: '#FFA500', 50: '#0080FF', 200: '#FF00FF'}
            for period, color in sma_colors.items():
                if f'SMA_{period}' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[f'SMA_{period}'],
                        name=f'SMA {period}',
                        line=dict(color=color, width=2)
                    ), row=1, col=1)
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=1.5)
                ), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1, opacity=0.3)
            
            # Volume
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ), row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Technical Analysis Statistics & Assessment
            st.subheader("📊 Technical Analysis & Signals")
            
            # Get technical values
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else 0
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 20
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            
            # SMA positions
            sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
            sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
            sma_200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns else current_price
            
            # Bollinger Bands
            bb_position = None
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                bb_upper = df['BB_Upper'].iloc[-1]
                bb_lower = df['BB_Lower'].iloc[-1]
                bb_range = bb_upper - bb_lower
                bb_position = ((current_price - bb_lower) / bb_range) * 100 if bb_range > 0 else 50
            
            # Calculate signals
            def get_signal_color(value, bullish_threshold, bearish_threshold, reverse=False):
                """Return color and assessment based on thresholds"""
                if not reverse:
                    if value >= bullish_threshold:
                        return "#00ff00", "BULLISH"
                    elif value <= bearish_threshold:
                        return "#ff0000", "BEARISH"
                    else:
                        return "#ffff00", "NEUTRAL"
                else:
                    if value <= bullish_threshold:
                        return "#00ff00", "BULLISH"
                    elif value >= bearish_threshold:
                        return "#ff0000", "BEARISH"
                    else:
                        return "#ffff00", "NEUTRAL"
            
            # RSI Assessment
            rsi_color, rsi_signal = get_signal_color(rsi, 70, 30, reverse=True)
            
            # MACD Assessment
            macd_bullish = macd > macd_signal
            macd_color = "#00ff00" if macd_bullish else "#ff0000"
            macd_signal_text = "BULLISH" if macd_bullish else "BEARISH"
            
            # Trend Assessment (SMA alignment)
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            price_above_sma200 = current_price > sma_200
            
            trend_score = sum([price_above_sma20, price_above_sma50, price_above_sma200])
            if trend_score == 3:
                trend_color, trend_signal = "#00ff00", "STRONG UPTREND"
            elif trend_score == 2:
                trend_color, trend_signal = "#7FFF00", "UPTREND"
            elif trend_score == 1:
                trend_color, trend_signal = "#ffff00", "NEUTRAL"
            else:
                trend_color, trend_signal = "#ff0000", "DOWNTREND"
            
            # ADX Assessment (trend strength)
            adx_color, adx_signal = get_signal_color(adx, 25, 15, reverse=False)
            if adx > 25:
                adx_signal = "STRONG TREND"
            elif adx > 15:
                adx_signal = "MODERATE"
            else:
                adx_signal = "WEAK/RANGING"
            
            # Volume Assessment
            vol_color, vol_signal = get_signal_color(volume_ratio, 1.5, 0.7, reverse=False)
            
            # Speedometer visualizations
            def create_speedometer(value, title, min_val=0, max_val=100, thresholds=[30, 70]):
                """Create a speedometer gauge"""
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': title, 'font': {'size': 16, 'color': '#FFFFFF'}},
                    number={'font': {'size': 24, 'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [min_val, max_val], 'tickcolor': '#FFFFFF'},
                        'bar': {'color': "#1f77b4"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#404040",
                        'steps': [
                            {'range': [min_val, thresholds[0]], 'color': '#ff0000'},
                            {'range': [thresholds[0], thresholds[1]], 'color': '#ffff00'},
                            {'range': [thresholds[1], max_val], 'color': '#00ff00'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': value
                        }
                    }
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#FFFFFF'}
                )
                return fig
            
            # Display speedometers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(create_speedometer(rsi, "RSI (14)", 0, 100, [30, 70]), use_container_width=True)
                st.markdown(f"<div style='text-align:center; color:{rsi_color}; font-size:20px; font-weight:bold;'>{rsi_signal}</div>", unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(create_speedometer(adx, "ADX (Trend Strength)", 0, 50, [15, 25]), use_container_width=True)
                st.markdown(f"<div style='text-align:center; color:{adx_color}; font-size:20px; font-weight:bold;'>{adx_signal}</div>", unsafe_allow_html=True)
            
            with col3:
                volume_gauge_value = min(volume_ratio * 50, 100)  # Scale to 0-100
                st.plotly_chart(create_speedometer(volume_gauge_value, "Volume (vs 20D Avg)", 0, 100, [50, 75]), use_container_width=True)
                st.markdown(f"<div style='text-align:center; color:{vol_color}; font-size:20px; font-weight:bold;'>{vol_signal} ({volume_ratio:.2f}x)</div>", unsafe_allow_html=True)
            
            st.divider()
            
            # Technical Statistics Table
            st.subheader("📋 Technical Statistics")
            
            stats_data = {
                "Indicator": ["RSI (14)", "MACD", "Trend (SMA)", "ADX", "Volume Ratio", "Bollinger Band Position"],
                "Value": [
                    f"{rsi:.1f}",
                    f"{macd:.2f}",
                    f"{trend_score}/3 SMAs",
                    f"{adx:.1f}",
                    f"{volume_ratio:.2f}x",
                    f"{bb_position:.1f}%" if bb_position else "N/A"
                ],
                "Signal": [
                    rsi_signal,
                    macd_signal_text,
                    trend_signal,
                    adx_signal,
                    vol_signal,
                    "Overbought" if bb_position and bb_position > 80 else ("Oversold" if bb_position and bb_position < 20 else "Normal") if bb_position else "N/A"
                ],
                "Color": [rsi_color, macd_color, trend_color, adx_color, vol_color, "#ffff00"]
            }
            
            stats_df = pd.DataFrame(stats_data)
            
            # Create HTML table with colored signals
            html_stats = '<table style="width:100%; border-collapse: collapse; font-size:14px;">'
            html_stats += '<tr style="background:#1a1a1a; border-bottom:2px solid #404040;">'
            html_stats += '<th style="padding:12px; text-align:left; color:#FFFFFF;">Indicator</th>'
            html_stats += '<th style="padding:12px; text-align:center; color:#FFFFFF;">Value</th>'
            html_stats += '<th style="padding:12px; text-align:center; color:#FFFFFF;">Signal</th>'
            html_stats += '</tr>'
            
            for idx, row in stats_df.iterrows():
                html_stats += '<tr style="border-bottom:1px solid #404040;">'
                html_stats += f'<td style="padding:10px; color:#FFFFFF;">{row["Indicator"]}</td>'
                html_stats += f'<td style="padding:10px; text-align:center; color:#FFFFFF;">{row["Value"]}</td>'
                html_stats += f'<td style="padding:10px; text-align:center; background:{row["Color"]}; color:#000000; font-weight:bold;">{row["Signal"]}</td>'
                html_stats += '</tr>'
            
            html_stats += '</table>'
            st.markdown(html_stats, unsafe_allow_html=True)
            
            st.divider()
            
            # Overall Assessment
            st.subheader("🎯 Overall Assessment")
            
            # Calculate overall score
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
            
            # MACD
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Trend
            if trend_score >= 2:
                bullish_signals += trend_score - 1
            else:
                bearish_signals += (2 - trend_score)
            
            # ADX (only if strong)
            if adx > 25:
                if trend_score >= 2:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Volume
            if volume_ratio > 1.5 and price_change_pct > 0:
                bullish_signals += 1
            elif volume_ratio > 1.5 and price_change_pct < 0:
                bearish_signals += 1
            
            # Total signals
            total_signals = bullish_signals + bearish_signals
            bullish_pct = (bullish_signals / total_signals * 100) if total_signals > 0 else 50
            
            # Determine recommendation
            if bullish_pct >= 70:
                recommendation = "🟢 BUY"
                rec_color = "#00ff00"
                rec_text = "Strong bullish signals suggest buying opportunity"
            elif bullish_pct >= 55:
                recommendation = "🟢 ACCUMULATE"
                rec_color = "#7FFF00"
                rec_text = "Moderately bullish, consider gradual accumulation"
            elif bullish_pct >= 45:
                recommendation = "🟡 HOLD"
                rec_color = "#ffff00"
                rec_text = "Mixed signals, maintain current position"
            elif bullish_pct >= 30:
                recommendation = "🟠 REDUCE"
                rec_color = "#FFA500"
                rec_text = "Moderately bearish, consider reducing exposure"
            else:
                recommendation = "🔴 SELL"
                rec_color = "#ff0000"
                rec_text = "Strong bearish signals suggest selling"
            
            # Display recommendation
            st.markdown(f"""
            <div style="background:#1a1a1a; padding:30px; border-radius:10px; text-align:center; border: 3px solid {rec_color};">
                <h1 style="color:{rec_color}; margin:0;">{recommendation}</h1>
                <h3 style="color:#FFFFFF; margin:20px 0;">{rec_text}</h3>
                <p style="color:#CCCCCC; font-size:16px;">
                    Bullish Signals: {bullish_signals} | Bearish Signals: {bearish_signals} | Score: {bullish_pct:.0f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("⚠️ This is an automated technical analysis. Always conduct your own research and consider fundamental factors before making investment decisions.")
            
        else:
            st.warning("No data available for the selected period.")
    
    except Exception as e:
        st.error(f"Error loading chart data: {str(e)}")

# ========================================
# OPTIONS CHAIN
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
        with st.spinner("Loading..."):
            options_df = get_options_chain(ticker_choice, dte_min, dte_max)
            
            if not options_df.empty:
                st.success(f"Loaded {len(options_df)} contracts")
                
                option_type = st.radio("Type", ["Calls", "Puts", "Both"], horizontal=True)
                
                if option_type == "Calls":
                    filtered = options_df[options_df['type'] == 'Call']
                elif option_type == "Puts":
                    filtered = options_df[options_df['type'] == 'Put']
                else:
                    filtered = options_df
                
                display_cols = ['strike', 'type', 'dte', 'expiration', 'mid', 'impliedVolatility', 
                               'volume', 'openInterest', 'delta', 'gamma', 'theta', 'vega']
                available_cols = [col for col in display_cols if col in filtered.columns]
                
                st.dataframe(filtered[available_cols].sort_values('dte'), use_container_width=True)
            else:
                st.warning("No options data available")

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
st.caption("SPY Pro v3.0 COMPLETE - Full Trading System with Persistent Storage")
