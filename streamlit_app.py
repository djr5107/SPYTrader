# streamlit_app_FINAL.py - SPY Pro v3.0 COMPLETE & FIXED
# Full featured trading system with backtest, live trading, signal history, and persistent storage
# FIXES:
# 1. Removed duplicate MARKET_ETFS definition (NameError fixed)
# 2. Added click-to-navigate from Market Dashboard to Chart Analysis  
# 3. Integrated all 80+ ETFs into Trading Hub for signal generation
# 4. Fixed syntax errors in Signal History section
# 5. Added proper total return calculation including dividends

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

st.set_page_config(page_title="SPY Pro v3.0-FINAL", layout="wide")
st.title("SPY Pro v3.0 - FINAL 🚀")
st.caption("✨ Full Trading System | Total Return w/ Dividends | All ETFs | Click Navigation")

# Persistent Storage Paths
DATA_DIR = Path("trading_data")
DATA_DIR.mkdir(exist_ok=True)
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ACTIVE_TRADES_FILE = DATA_DIR / "active_trades.json"
SIGNAL_QUEUE_FILE = DATA_DIR / "signal_queue.json"
SIGNAL_HISTORY_FILE = DATA_DIR / "signal_history.json"
PERFORMANCE_FILE = DATA_DIR / "performance_metrics.json"

# Legacy tickers for display
LEGACY_TICKERS = ["SPY", "SVXY", "QQQ", "EFA", "EEM", "AGG", "TLT"]

# Signal expiration time (minutes)
DEFAULT_SIGNAL_EXPIRATION = 30

# Load/Save Functions
def load_json(filepath, default):
    """Load JSON file with robust error handling"""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
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
    """Save JSON file with atomic write"""
    try:
        temp_file = filepath.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(filepath)
    except Exception as e:
        st.error(f"Error saving {filepath.name}: {e}")

# Market Dashboard ETFs - GLOBAL DEFINITION (ONLY ONE)
# Using Vanguard ETFs where available for better total return tracking with dividends
MARKET_ETFS = {
    "Equities": {
        "Large Cap": "VOO",  # Vanguard S&P 500 (better dividend tracking than IVV)
        "Mid Cap": "VO",     # Vanguard Mid-Cap
        "Small Cap": "VB",   # Vanguard Small-Cap
        "SMID": "SMMD",
        "All Cap": "VTI",    # Vanguard Total Stock Market
        "Developed Markets": "VEA",  # Vanguard Developed Markets
        "Emerging Markets": "VWO",   # Vanguard Emerging Markets
        "World ex-US": "VEU",        # Vanguard All-World ex-US
        "World": "VT"        # Vanguard Total World Stock
    },
    "Fixed Income": {
        "Aggregate": "BND",   # Vanguard Total Bond Market
        "Short-Term Treasury": "VGSH",  # Vanguard Short-Term Treasury
        "Intermediate Treasury": "VGIT", # Vanguard Intermediate-Term Treasury
        "Long-Term Treasury": "VGLT",    # Vanguard Long-Term Treasury
        "TIPS": "VTIP",       # Vanguard Short-Term Inflation-Protected
        "Investment Grade Corp": "VCIT", # Vanguard Intermediate-Term Corporate
        "High Yield Corporate": "HYG",
        "Emerging Market Bonds": "VWOB", # Vanguard Emerging Markets Bond
        "Municipals": "VTEB", # Vanguard Tax-Exempt Bond
        "Mortgage-Backed": "VMBS", # Vanguard Mortgage-Backed Securities
        "Floating Rate": "FLOT"
    },
    "Real Assets": {
        "Bitcoin": "IBIT",
        "Gold": "GLD",     # SPDR Gold (better liquidity)
        "Silver": "SLV",
        "Commodity Basket": "GSG",
        "Natural Resources": "IGE",
        "Oil": "USO",      # United States Oil Fund (more liquid)
        "Real Estate": "VNQ",  # Vanguard Real Estate
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
        "Value": "VTV",    # Vanguard Value
        "Momentum": "MTUM",
        "Quality": "QUAL",
        "Size (Small Cap)": "VB",  # Vanguard Small-Cap
        "Low Volatility": "USMV",
        "Dividend": "VYM",  # Vanguard High Dividend Yield
        "Growth": "VUG",    # Vanguard Growth
        "High Dividend": "VYM"
    }
}

# Build expanded ticker list
ALL_TICKERS = list(set(LEGACY_TICKERS + [etf for category in MARKET_ETFS.values() for etf in category.values()]))
ALL_TICKERS.sort()
TICKERS = ALL_TICKERS

# Initialize Session State
def init_session_state():
    """Initialize all session state variables with persistent data"""
    
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
        st.session_state.signal_queue = load_json(SIGNAL_QUEUE_FILE, [])
        for sig in st.session_state.signal_queue:
            if 'timestamp' in sig and isinstance(sig['timestamp'], str):
                sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
    
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = load_json(SIGNAL_HISTORY_FILE, [])
    
    if 'performance' not in st.session_state:
        st.session_state.performance = load_json(PERFORMANCE_FILE, {
            'total_pnl': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        })
    
    if 'macro_analyzer' not in st.session_state:
        st.session_state.macro_analyzer = MacroAnalyzer()
    
    if 'last_save' not in st.session_state:
        st.session_state.last_save = datetime.now()
    
    if 'nav_to_chart' not in st.session_state:
        st.session_state.nav_to_chart = False
    if 'chart_ticker' not in st.session_state:
        st.session_state.chart_ticker = 'SPY'

init_session_state()

def save_all_data():
    """Save all persistent data"""
    trade_log_data = st.session_state.trade_log.to_dict('records')
    save_json(TRADE_LOG_FILE, trade_log_data)
    
    active_trades = []
    for trade in st.session_state.active_trades:
        trade_copy = trade.copy()
        if 'entry_time' in trade_copy and isinstance(trade_copy['entry_time'], datetime):
            trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
        active_trades.append(trade_copy)
    save_json(ACTIVE_TRADES_FILE, active_trades)
    
    signal_queue = []
    for sig in st.session_state.signal_queue:
        sig_copy = sig.copy()
        if 'timestamp' in sig_copy and isinstance(sig_copy['timestamp'], datetime):
            sig_copy['timestamp'] = sig_copy['timestamp'].isoformat()
        signal_queue.append(sig_copy)
    save_json(SIGNAL_QUEUE_FILE, signal_queue)
    
    save_json(SIGNAL_HISTORY_FILE, st.session_state.signal_history)
    save_json(PERFORMANCE_FILE, st.session_state.performance)
    
    st.session_state.last_save = datetime.now()

# Auto-save every 5 minutes
if (datetime.now() - st.session_state.last_save).total_seconds() > 300:
    save_all_data()

def is_market_open():
    """Check if market is open"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

@st.cache_data(ttl=60)
def fetch_market_data():
    data = {}
    for ticker in LEGACY_TICKERS:
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
    
    return df

def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id, 
               entry_numeric=None, exit_numeric=None, pnl_numeric=None, dte=None,
               strategy=None, thesis=None, max_hold=None, actual_hold=None, 
               conviction=None, signal_type=None):
    """Log a trade"""
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
            sig['expiration'] = now
            sig['status'] = 'Expired'
            st.session_state.signal_history.append(sig)
            expired.append(sig)
            st.session_state.signal_queue.remove(sig)
    
    if expired:
        save_all_data()
    
    return len(expired)

def generate_signal():
    """Enhanced signal generation across all 80+ ETFs"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    now_str = now.strftime("%m/%d %H:%M")
    
    expire_old_signals()
    
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
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else 0
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            stoch_k = df['Stoch_%K'].iloc[-1] if 'Stoch_%K' in df.columns and not pd.isna(df['Stoch_%K'].iloc[-1]) else 50
            
            sma_values = {}
            for period in [10, 20, 50, 200]:
                col_name = f'SMA_{period}'
                if col_name in df.columns and len(df) >= period:
                    sma_values[period] = df[col_name].iloc[-1]
            
            signal = None
            
            # SVXY Volatility Signals
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
                        'signal_type': 'Mean Reversion'
                    }
            
            # Golden Cross
            if not signal and 50 in sma_values and 200 in sma_values:
                prev_50 = df['SMA_50'].iloc[-2]
                prev_200 = df['SMA_200'].iloc[-2]
                
                if (sma_values[50] > sma_values[200] and 
                    prev_50 <= prev_200 and 
                    current_price > sma_values[50]):
                    
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'Golden Cross',
                        'symbol': ticker,
                        'action': f"BUY 20 shares @ ${current_price:.2f}",
                        'size': 20,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Golden Cross',
                        'thesis': f"GOLDEN CROSS: SMA50 crossed above SMA200. Strong bullish signal at ${current_price:.2f}",
                        'conviction': 9,
                        'signal_type': 'Golden Cross'
                    }
            
            # SMA 10/20 Cross
            if not signal and 10 in sma_values and 20 in sma_values:
                prev_10 = df['SMA_10'].iloc[-2]
                prev_20 = df['SMA_20'].iloc[-2]
                
                if (sma_values[10] > sma_values[20] and 
                    prev_10 <= prev_20 and 
                    current_price > sma_values[10] and 
                    volume_ratio > 1.2):
                    
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'timestamp': now,
                        'time': now_str,
                        'type': 'SMA 10/20 Cross',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Short-term Momentum',
                        'thesis': f"SMA CROSS: SMA10 crossed above SMA20 with volume {volume_ratio:.1f}x",
                        'conviction': 7,
                        'signal_type': 'SMA Cross'
                    }
            
            # Volume Breakout
            if not signal and volume_ratio > 1.5 and price_change_pct > 0.3:
                if 20 in sma_values and current_price > sma_values[20]:
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
                        'strategy': f'{ticker} Volume Breakout',
                        'thesis': f"VOLUME BREAKOUT: {volume_ratio:.1f}x avg volume with +{price_change_pct:.2f}% move",
                        'conviction': 7,
                        'signal_type': 'Volume Breakout'
                    }
            
            # Oversold Bounce
            if not signal and rsi < 35 and stoch_k < 30:
                if rsi > df['RSI'].iloc[-2]:
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
                        'strategy': f'{ticker} Mean Reversion',
                        'thesis': f"OVERSOLD BOUNCE: RSI {rsi:.0f}, Stoch {stoch_k:.0f}. Both oversold.",
                        'conviction': 7,
                        'signal_type': 'Oversold'
                    }
            
            # Mean Reversion (Bollinger Bands)
            if not signal and 'BB_Lower' in df.columns:
                bb_lower = df['BB_Lower'].iloc[-1]
                if current_price <= bb_lower * 1.01 and rsi < 40 and price_change_pct > 0:
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
                        'strategy': f'{ticker} BB Bounce',
                        'thesis': f"BB BOUNCE: Price at lower band (${bb_lower:.2f}), RSI {rsi:.0f}",
                        'conviction': 7,
                        'signal_type': 'Mean Reversion'
                    }
            
            if signal and signal['conviction'] >= MIN_CONVICTION:
                if not any(s['symbol'] == ticker for s in st.session_state.signal_queue):
                    st.session_state.signal_queue.append(signal)
                    st.session_state.signal_history.append(signal)
                    save_all_data()
                    return True
                    
        except Exception as e:
            continue
    
    return False

def simulate_exit():
    """Check and execute exit conditions for active trades"""
    if not st.session_state.active_trades:
        return
    
    now = datetime.now(ZoneInfo("US/Eastern"))
    
    for trade in st.session_state.active_trades[:]:
        try:
            ticker = yf.Ticker(trade['symbol'])
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            entry_price = trade.get('entry_price', trade.get('entry', 0))
            minutes_held = (now - trade['entry_time']).total_seconds() / 60
            gain_pct = ((current_price - entry_price) / entry_price) * 100
            
            exit_triggered = False
            exit_reason = ""
            
            max_hold = trade.get('max_hold', 240)
            if minutes_held >= max_hold:
                exit_triggered = True
                exit_reason = f"Time limit ({max_hold} min)"
            elif gain_pct >= 5.0:
                exit_triggered = True
                exit_reason = "Profit target (5%)"
            elif gain_pct <= -2.0:
                exit_triggered = True
                exit_reason = "Stop loss (2%)"
            
            if exit_triggered:
                exit_price = current_price
                minutes_held = (now - trade['entry_time']).total_seconds() / 60
                
                if 'SELL SHORT' in trade.get('action', '') or 'Short' in trade.get('strategy', ''):
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
                    sig_id=trade.get('signal_id'),
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
        
        except Exception as e:
            continue
    
    save_all_data()

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

# NEW: Improved total return calculation including dividends
@st.cache_data(ttl=300)
def fetch_multi_period_performance_with_dividends(market_etfs):
    """Fetch performance INCLUDING DIVIDENDS for multiple time periods"""
    periods_config = {
        '1D': {'days': 1},
        '5D': {'days': 5},
        '1M': {'days': 30},
        'YTD': {'ytd': True},
        '1Y': {'days': 365},
        '3Y': {'days': 1095}
    }
    
    results = {}
    
    for category, etf_dict in market_etfs.items():
        category_data = []
        
        for name, ticker in etf_dict.items():
            try:
                t = yf.Ticker(ticker)
                row_data = {'Name': name, 'ETF': ticker}
                
                for period_name, config in periods_config.items():
                    try:
                        # Fetch historical data with dividends
                        if 'ytd' in config:
                            start_date = datetime(datetime.now().year, 1, 1)
                            hist = t.history(start=start_date, auto_adjust=False)
                        else:
                            hist = t.history(period=f"{config['days']+5}d", auto_adjust=False)
                        
                        if not hist.empty and len(hist) >= 2:
                            # Calculate total return including dividends
                            # Total Return = (End Price + Dividends) / Start Price - 1
                            
                            # Get start and end prices
                            if 'ytd' in config:
                                start_price = hist['Close'].iloc[0]
                                end_price = hist['Close'].iloc[-1]
                                dividends = hist['Dividends'].sum() if 'Dividends' in hist.columns else 0
                            else:
                                # Get data for exact period
                                target_days = config['days']
                                if len(hist) > target_days:
                                    start_price = hist['Close'].iloc[-(target_days+1)]
                                    end_price = hist['Close'].iloc[-1]
                                    dividends = hist['Dividends'].iloc[-(target_days+1):].sum() if 'Dividends' in hist.columns else 0
                                else:
                                    start_price = hist['Close'].iloc[0]
                                    end_price = hist['Close'].iloc[-1]
                                    dividends = hist['Dividends'].sum() if 'Dividends' in hist.columns else 0
                            
                            # Calculate total return percentage
                            total_return = ((end_price - start_price + dividends) / start_price) * 100
                            row_data[period_name] = total_return
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
def fetch_custom_period_performance_with_dividends(market_etfs, start_date, end_date):
    """Fetch performance INCLUDING DIVIDENDS for custom date range"""
    results = {}
    
    for category, etf_dict in market_etfs.items():
        category_data = []
        
        for name, ticker in etf_dict.items():
            try:
                t = yf.Ticker(ticker)
                # Fetch with auto_adjust=False to get dividends separately
                hist = t.history(start=start_date, end=end_date, auto_adjust=False)
                
                if not hist.empty and len(hist) >= 2:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    
                    # Sum all dividends in the period
                    dividends = hist['Dividends'].sum() if 'Dividends' in hist.columns else 0
                    
                    # Total return = (End Price + Dividends) / Start Price - 1
                    total_return = ((end_price - start_price + dividends) / start_price) * 100
                    
                    category_data.append({
                        'Name': name,
                        'ETF': ticker,
                        'Return': total_return
                    })
            except Exception as e:
                continue
        
        if category_data:
            results[category] = pd.DataFrame(category_data)
    
    return results

def get_color_from_return(value):
    """Return background and text color based on return value"""
    if pd.isna(value):
        return '#2a2a2a', '#888888'
    
    if value >= 10:
        return '#006400', '#FFFFFF'
    elif value >= 5:
        return '#228B22', '#FFFFFF'
    elif value >= 0:
        return '#90EE90', '#000000'
    elif value >= -5:
        return '#FFB6C1', '#000000'
    elif value >= -10:
        return '#DC143C', '#FFFFFF'
    else:
        return '#8B0000', '#FFFFFF'

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Trading Hub", "Market Dashboard", "Signal History", "Backtest", "Trade Log", "Performance", "Chart Analysis", "Options Chain", "Macro Dashboard"],
    icons=["bar-chart", "globe", "clock-history", "graph-up", "list-task", "trophy", "graph-up-arrow", "cash-stack", "speedometer"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Trading Settings")
    TRADING_MODE = st.radio("Trading Mode", ["PAPER", "REAL"], index=0)
    SIGNAL_EXPIRATION_MINUTES = st.number_input("Signal Expiration (min)", value=DEFAULT_SIGNAL_EXPIRATION, min_value=5, max_value=180, step=5)
    MIN_CONVICTION = st.slider("Min Conviction", 0.0, 10.0, 6.0, 0.5)
    USE_DYNAMIC_STOPS = st.checkbox("Dynamic Position Sizing", value=True)
    
    st.divider()
    
    st.subheader("Signal Generation")
    SIGNAL_FREQUENCY = st.slider("Signal Check Frequency", 1, 10, 3)
    USE_MACRO_FILTER = st.checkbox("Use Macro Filter", value=True)
    
    st.divider()
    
    st.subheader("Backtest Settings")
    BACKTEST_CAPITAL = st.number_input("Starting Capital ($)", value=100000, min_value=10000, step=10000)
    BACKTEST_POSITION_SIZE = st.slider("Position Size (%)", 1, 100, 20, 1)
    
    st.divider()
    
    st.subheader("Data Management")
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.button("⚠️ Confirm Clear All"):
            st.session_state.trade_log = pd.DataFrame(columns=[
                'Timestamp', 'Type', 'Symbol', 'Action', 'Size', 'Entry', 'Exit', 'P&L', 
                'Status', 'Signal ID', 'Entry Price Numeric', 'Exit Price Numeric', 
                'P&L Numeric', 'DTE', 'Strategy', 'Thesis', 'Max Hold Minutes', 'Actual Hold Minutes',
                'Conviction', 'Signal Type'
            ])
            st.session_state.active_trades = []
            st.session_state.signal_queue = []
            st.session_state.signal_history = []
            st.session_state.performance = {
                'total_pnl': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
            save_all_data()
            st.success("All data cleared!")
            st.rerun()

# Continue in next message due to length...

# ========================================
# TRADING HUB
# ========================================

if selected == "Trading Hub":
    st.header("Trading Hub - Multi-Asset Analysis")
    st.caption(f"🎯 Monitoring {len(TICKERS)} ETFs across all asset classes")
    
    st.subheader("Core Tickers Overview")
    cols = st.columns(7)
    for i, ticker in enumerate(LEGACY_TICKERS):
        with cols[i]:
            data = market_data.get(ticker, {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0})
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
    **Total Tickers:** {len(TICKERS)} ETFs
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Generate Signals", use_container_width=True):
            with st.spinner(f"Scanning {len(TICKERS)} ETFs..."):
                found = False
                for _ in range(SIGNAL_FREQUENCY):
                    if generate_signal():
                        found = True
                if found:
                    st.success("New signals generated!")
                else:
                    st.info("No signals met criteria")
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
    
    if market_open:
        generate_signal()
        simulate_exit()
    
    st.subheader(f"📊 Trading Signals ({len(st.session_state.signal_queue)} Active)")
    
    if len(st.session_state.signal_queue) == 0:
        st.info(f"""
        **No active signals.**
        
        Signals generate when:
        - ✅ Market conditions align across {len(TICKERS)} ETFs
        - ✅ Conviction threshold met (≥{MIN_CONVICTION:.0f})
        - ✅ Macro filter passed (if enabled)
        
        Try: Generate Signals button or lower Min Conviction in sidebar
        """)
    
    for sig in st.session_state.signal_queue:
        signal_age = (datetime.now(ZoneInfo("US/Eastern")) - sig['timestamp']).total_seconds() / 60
        time_left = SIGNAL_EXPIRATION_MINUTES - signal_age
        
        st.markdown(f"""
        <div style="background:#1a1a1a;padding:20px;border-radius:10px;border-left:5px solid #4CAF50;margin:10px 0;">
            <h3 style="margin:0;color:white;">{sig['symbol']} - {sig['type']}</h3>
            <p style="margin:5px 0;"><strong>Action:</strong> {sig['action']} | <strong>Conviction:</strong> {sig['conviction']}/10</p>
            <p style="margin:5px 0;"><strong>Entry:</strong> ${sig['entry_price']:.2f}</p>
            <p style="margin:5px 0;"><strong>Thesis:</strong> {sig['thesis']}</p>
            <p style="margin:5px 0;font-size:12px;color:#888;">Time Left: {time_left:.1f} min | Generated: {sig['time']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("✅ Execute Trade", key=f"exec_{sig['id']}", use_container_width=True):
                log_trade(
                    ts=datetime.now(ZoneInfo("US/Eastern")).strftime("%m/%d %H:%M"),
                    typ='Open',
                    sym=sig['symbol'],
                    action=sig['action'].split()[0],
                    size=sig['size'],
                    entry=f"${sig['entry_price']:.2f}",
                    exit=None,
                    pnl=None,
                    status='OPEN',
                    sig_id=sig['id'],
                    entry_numeric=sig['entry_price'],
                    strategy=sig['strategy'],
                    thesis=sig['thesis'],
                    max_hold=sig.get('max_hold', 240),
                    conviction=sig['conviction'],
                    signal_type=sig['type']
                )
                
                st.session_state.active_trades.append({
                    'symbol': sig['symbol'],
                    'entry_price': sig['entry_price'],
                    'size': sig['size'],
                    'entry_time': datetime.now(ZoneInfo("US/Eastern")),
                    'signal_id': sig['id'],
                    'strategy': sig['strategy'],
                    'thesis': sig['thesis'],
                    'max_hold': sig.get('max_hold', 240),
                    'conviction': sig['conviction'],
                    'signal_type': sig['type'],
                    'action': sig['action']
                })
                
                st.session_state.signal_queue.remove(sig)
                save_all_data()
                st.success(f"Trade executed: {sig['action']}")
                st.rerun()
        
        with col_b:
            if st.button("❌ Dismiss Signal", key=f"dismiss_{sig['id']}", use_container_width=True):
                st.session_state.signal_queue.remove(sig)
                save_all_data()
                st.rerun()
    
    st.divider()
    st.subheader(f"🎯 Active Trades ({len(st.session_state.active_trades)})")
    
    if len(st.session_state.active_trades) == 0:
        st.info("No active trades. Execute signals above to open positions.")
    
    for trade in st.session_state.active_trades:
        try:
            ticker = yf.Ticker(trade['symbol'])
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            entry_price = trade.get('entry_price', trade.get('entry', 0))
            pnl = (current_price - entry_price) * trade['size']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
            
            minutes_held = (datetime.now(ZoneInfo("US/Eastern")) - trade['entry_time']).total_seconds() / 60
            
            pnl_color = 'green' if pnl >= 0 else 'red'
            
            st.markdown(f"""
            <div style="background:#1a1a1a;padding:20px;border-radius:10px;border-left:5px solid {'#4CAF50' if pnl >= 0 else '#f44336'};margin:10px 0;">
                <h3 style="margin:0;color:white;">{trade['symbol']} - {trade['size']} shares</h3>
                <p style="margin:5px 0;"><strong>Entry:</strong> ${entry_price:.2f} | <strong>Current:</strong> ${current_price:.2f}</p>
                <p style="margin:5px 0;"><strong>P&L:</strong> <span style="color:{pnl_color};font-weight:bold;">${pnl:+.2f} ({pnl_pct:+.2f}%)</span></p>
                <p style="margin:5px 0;"><strong>Thesis:</strong> {trade.get('thesis', 'N/A')}</p>
                <p style="margin:5px 0;font-size:12px;color:#888;">Held: {minutes_held:.1f} min | Max Hold: {trade.get('max_hold', 240)} min</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔴 Close Position", key=f"close_{trade['signal_id']}", use_container_width=True):
                exit_price = current_price
                final_pnl = (exit_price - entry_price) * trade['size']
                
                log_trade(
                    ts=datetime.now(ZoneInfo("US/Eastern")).strftime("%m/%d %H:%M"),
                    typ='Close',
                    sym=trade['symbol'],
                    action='SELL',
                    size=trade['size'],
                    entry=f"${entry_price:.2f}",
                    exit=f"${exit_price:.2f}",
                    pnl=f"${final_pnl:+.2f}",
                    status='CLOSED',
                    sig_id=trade.get('signal_id'),
                    entry_numeric=entry_price,
                    exit_numeric=exit_price,
                    pnl_numeric=final_pnl,
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
        
        except Exception as e:
            st.error(f"Error fetching data for {trade['symbol']}: {str(e)}")


# ========================================
# MARKET DASHBOARD - WITH DIVIDENDS & CLICK NAVIGATION
# ========================================

elif selected == "Market Dashboard":
    st.header("📊 Market Dashboard")
    st.caption("💰 Total Return including dividends | 📍 Click any ETF to view its chart")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        view_mode = st.radio("View Mode", ["Standard Periods", "Custom Period"], horizontal=True)
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if view_mode == "Custom Period":
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col2:
            custom_end = st.date_input("End Date", value=datetime.now())
    
    def create_multi_period_table(df, title):
        """Create HTML table with navigation buttons"""
        st.subheader(title)
        
        periods = ['1D', '5D', '1M', 'YTD', '1Y', '3Y']
        
        html = '<table style="width:100%; border-collapse:collapse; font-family:Arial; margin-bottom:20px;">'
        
        html += '<tr style="background:#1a1a1a; border-bottom: 2px solid #404040;">'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">Name</th>'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">ETF</th>'
        for period in periods:
            html += f'<th style="padding:12px; text-align:right; color:#FFFFFF; border-right:1px solid #404040;">{period}</th>'
        html += '</tr>'
        
        for idx, row in df.iterrows():
            etf_ticker = row['ETF']
            html += f'<tr style="border-bottom:1px solid #404040;">'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#FFFFFF; border-right:1px solid #404040;">{row["Name"]}</td>'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#DDDDDD; border-right:1px solid #404040;"><strong>{etf_ticker}</strong></td>'
            
            for period in periods:
                if period in row and pd.notna(row[period]):
                    return_val = row[period]
                    bg_color, text_color = get_color_from_return(return_val)
                    html += f'<td style="padding:10px; background:{bg_color}; color:{text_color}; text-align:right; font-weight:bold; border-right:1px solid #404040;">{return_val:+.1f}%</td>'
                else:
                    html += '<td style="padding:10px; background:#2a2a2a; color:#888888; text-align:right; border-right:1px solid #404040;">-</td>'
            
            html += '</tr>'
        
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        
        st.caption("👇 Click to view chart")
        etf_list = df['ETF'].tolist()
        num_cols = 6
        cols = st.columns(num_cols)
        for idx, etf in enumerate(etf_list):
            col_idx = idx % num_cols
            with cols[col_idx]:
                if st.button(f"📊 {etf}", key=f"nav_{etf}_{title}", use_container_width=True):
                    st.session_state.chart_ticker = etf
                    st.session_state.nav_to_chart = True
                    st.rerun()
        
        st.write("")
    
    def create_custom_period_table(df, title):
        """Create HTML table for custom period with navigation"""
        st.subheader(title)
        
        html = '<table style="width:100%; border-collapse:collapse; font-family:Arial; margin-bottom:20px;">'
        
        html += '<tr style="background:#1a1a1a; border-bottom: 2px solid #404040;">'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">Name</th>'
        html += '<th style="padding:12px; text-align:left; color:#FFFFFF; border-right:1px solid #404040;">ETF</th>'
        html += '<th style="padding:12px; text-align:right; color:#FFFFFF;">Total Return</th>'
        html += '</tr>'
        
        for idx, row in df.iterrows():
            etf_ticker = row['ETF']
            return_val = row['Return']
            bg_color, text_color = get_color_from_return(return_val)
            
            html += f'<tr style="border-bottom:1px solid #404040;">'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#FFFFFF; border-right:1px solid #404040;">{row["Name"]}</td>'
            html += f'<td style="padding:10px; background:#2a2a2a; color:#DDDDDD; border-right:1px solid #404040;"><strong>{etf_ticker}</strong></td>'
            html += f'<td style="padding:10px; background:{bg_color}; color:{text_color}; text-align:right; font-weight:bold;">{return_val:+.1f}%</td>'
            html += '</tr>'
        
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        
        st.caption("👇 Click to view chart")
        etf_list = df['ETF'].tolist()
        num_cols = 6
        cols = st.columns(num_cols)
        for idx, etf in enumerate(etf_list):
            col_idx = idx % num_cols
            with cols[col_idx]:
                if st.button(f"📊 {etf}", key=f"nav_{etf}_{title}_custom", use_container_width=True):
                    st.session_state.chart_ticker = etf
                    st.session_state.nav_to_chart = True
                    st.rerun()
        
        st.write("")
    
    with st.spinner("Loading market data with dividends..."):
        if view_mode == "Standard Periods":
            market_performance = fetch_multi_period_performance_with_dividends(MARKET_ETFS)
        else:
            market_performance = fetch_custom_period_performance_with_dividends(
                MARKET_ETFS, 
                datetime.combine(custom_start, datetime.min.time()),
                datetime.combine(custom_end, datetime.min.time())
            )
    
    if market_performance:
        if view_mode == "Standard Periods":
            for category_name in ["Equities", "Fixed Income", "Real Assets", "S&P Sectors", "Developed Markets", "Emerging Markets", "Factors"]:
                if category_name in market_performance:
                    icons = {
                        "Equities": "📈",
                        "Fixed Income": "📊",
                        "Real Assets": "💰",
                        "S&P Sectors": "🏭",
                        "Developed Markets": "🌍",
                        "Emerging Markets": "🌏",
                        "Factors": "🎯"
                    }
                    create_multi_period_table(market_performance[category_name], f"{icons.get(category_name, '')} {category_name}")
        
        else:
            period_label = f"{custom_start.strftime('%m/%d/%Y')} - {custom_end.strftime('%m/%d/%Y')}"
            st.subheader(f"Custom Period: {period_label}")
            
            for category_name in ["Equities", "Fixed Income", "Real Assets", "S&P Sectors", "Developed Markets", "Emerging Markets", "Factors"]:
                if category_name in market_performance:
                    icons = {
                        "Equities": "📈",
                        "Fixed Income": "📊",
                        "Real Assets": "💰",
                        "S&P Sectors": "🏭",
                        "Developed Markets": "🌍",
                        "Emerging Markets": "🌏",
                        "Factors": "🎯"
                    }
                    create_custom_period_table(market_performance[category_name], f"{icons.get(category_name, '')} {category_name}")
        
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
                    f"market_dashboard_total_return_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    else:
        st.warning("No data available for the selected period.")

# ========================================
# SIGNAL HISTORY
# ========================================

elif selected == "Signal History":
    st.header("📜 Signal History")
    
    if not st.session_state.signal_history:
        st.info("No signal history yet.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", ["Active", "Taken", "Skipped", "Expired"], 
                                          default=["Active", "Taken", "Skipped", "Expired"])
        with col2:
            ticker_filter = st.multiselect("Ticker", sorted(list(set([s['symbol'] for s in st.session_state.signal_history]))), 
                                          default=sorted(list(set([s['symbol'] for s in st.session_state.signal_history]))))
        with col3:
            days_back = st.slider("Days Back", 1, 30, 7)
        
        cutoff_time = datetime.now(ZoneInfo("US/Eastern")) - timedelta(days=days_back)
        filtered_signals = [
            sig for sig in st.session_state.signal_history
            if sig.get('status', 'Unknown') in status_filter
            and sig['symbol'] in ticker_filter
            and sig['timestamp'] >= cutoff_time
        ]
        
        st.write(f"**{len(filtered_signals)} signals from last {days_back} days**")
        
        if filtered_signals:
            history_df = pd.DataFrame([{
                'Time': sig['time'],
                'Symbol': sig['symbol'],
                'Type': sig['type'],
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
