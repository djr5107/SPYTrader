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

st.set_page_config(page_title="SPY Pro v2.30", layout="wide")
st.title("SPY Pro v2.30 - Wall Street Grade")
st.caption("Multi-Ticker | Enhanced Technicals | MTD/QTD/YTD Charts | Volume Analysis | CMT Grade Indicators")

# Persistent Storage Paths
DATA_DIR = Path("trading_data")
DATA_DIR.mkdir(exist_ok=True)
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
ACTIVE_TRADES_FILE = DATA_DIR / "active_trades.json"
SIGNAL_QUEUE_FILE = DATA_DIR / "signal_queue.json"
PERFORMANCE_FILE = DATA_DIR / "performance_metrics.json"

# Multi-Ticker Support
TICKERS = ["SPY", "SVXY", "QQQ", "EFA", "EEM", "AGG", "TLT"]

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
   
    st.divider()
    st.subheader("Trading Parameters")
    STOP_LOSS_PCT = -2.0  # Updated to -2%
    TRAILING_STOP_PCT = 1.0  # Updated to 1% for tighter profit locking
    st.caption(f"Stop Loss: {STOP_LOSS_PCT}%")
    st.caption(f"Trailing Stop: {TRAILING_STOP_PCT}%")

# Market Hours
def is_market_open():
    now = datetime.now(ZoneInfo("US/Eastern"))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and market_open <= now <= market_close

# Live Data + Options Chain
@st.cache_data(ttl=30)
def get_market_data():
    """Fetch real-time market data for all tickers"""
    data = {}
    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="1d", interval="1m")
           
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                data[ticker] = {
                    'price': current_price,
                    'change': hist['Close'].iloc[-1] - hist['Close'].iloc[0],
                    'change_pct': ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                    'volume': hist['Volume'].sum(),
                    'high': hist['High'].max(),
                    'low': hist['Low'].min(),
                    'open': hist['Open'].iloc[0]
                }
        except:
            data[ticker] = {
                'price': 0,
                'change': 0,
                'change_pct': 0,
                'volume': 0,
                'high': 0,
                'low': 0,
                'open': 0
            }
    return data

# Get VIX for options
@st.cache_data(ttl=60)
def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        return vix.info.get('regularMarketPrice', 15)
    except:
        return 15

market_data = get_market_data()
S = market_data['SPY']['price']
vix = get_vix()

# Calculate CMT-Grade Technical Indicators
def calculate_technical_indicators(data, periods=[10, 20, 50, 100, 200]):
    """Calculate comprehensive technical indicators used by Chartered Market Technicians"""
    df = data.copy()
   
    # 1. Multiple Moving Averages
    for period in periods:
        if len(df) >= period:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
   
    # 2. RSI (Relative Strength Index) - 14 period standard
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
   
    # 3. MACD (Moving Average Convergence Divergence)
    if len(df) >= 26:
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
   
    # 4. Bollinger Bands (20 period, 2 std dev)
    if len(df) >= 20:
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
   
    # 5. Stochastic Oscillator (14, 3, 3)
    if len(df) >= 14:
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
   
    # 6. ATR (Average True Range) - 14 period for volatility
    if len(df) >= 14:
        df['TR'] = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
   
    # 7. OBV (On-Balance Volume) - volume momentum
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
   
    # 8. ADX (Average Directional Index) - trend strength (14 period)
    if len(df) >= 14:
        df['High-Low'] = df['High'] - df['Low']
        df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                             np.maximum(df['High'] - df['High'].shift(), 0), 0)
        df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                             np.maximum(df['Low'].shift() - df['Low'], 0), 0)
       
        tr = df['TR'] if 'TR' in df.columns else df['High-Low']
        df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / tr.rolling(window=14).mean())
        df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / tr.rolling(window=14).mean())
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].rolling(window=14).mean()
   
    # 9. Volume Ratio
    if len(df) >= 20:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
   
    # 10. Rate of Change (ROC) - momentum
    for period in [5, 10, 20]:
        if len(df) >= period:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
   
    return df

# Log Trade Function
def log_trade(ts, typ, sym, action, size, entry, exit, pnl, status, sig_id,
              entry_numeric=None, exit_numeric=None, pnl_numeric=None, dte=None,
              strategy=None, thesis=None, max_hold=None, actual_hold=None,
              conviction=None, signal_type=None):
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
   
    # Update performance metrics
    if status.startswith('Closed') and pnl_numeric is not None:
        metrics = st.session_state.performance_metrics
        metrics['total_trades'] += 1
        metrics['total_pnl'] += pnl_numeric
       
        if pnl_numeric > 0:
            metrics['winning_trades'] += 1
            metrics['avg_win'] = (metrics['avg_win'] * (metrics['winning_trades'] - 1) + pnl_numeric) / metrics['winning_trades']
        else:
            metrics['losing_trades'] += 1
            metrics['avg_loss'] = (metrics['avg_loss'] * (metrics['losing_trades'] - 1) + pnl_numeric) / metrics['losing_trades']
       
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
       
        # Daily P&L tracking
        trade_date = datetime.strptime(ts.split()[0], "%m/%d").strftime("%Y-%m-%d")
        if trade_date not in metrics['daily_pnl']:
            metrics['daily_pnl'][trade_date] = 0
        metrics['daily_pnl'][trade_date] += pnl_numeric
       
        save_json(PERFORMANCE_FILE, metrics)

# Generate Enhanced Signals with CMT-Grade Technical Analysis
def generate_signal():
    """Generate signals using professional technical analysis across multiple tickers"""
    now = datetime.now(ZoneInfo("US/Eastern"))
    now_str = now.strftime("%m/%d %H:%M")
   
    if not is_market_open() or any(s['time'] == now_str for s in st.session_state.signal_queue):
        return
   
    # Analyze each ticker for signals
    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)
            hist_data = t.history(period="100d", interval="1d") # Get enough data for 200-day SMA
           
            if hist_data.empty or len(hist_data) < 50:
                continue
           
            # Calculate all technical indicators
            df = calculate_technical_indicators(hist_data)
           
            if df.empty or len(df) < 20:
                continue
           
            # Get current values
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
           
            # Get indicator values (with fallbacks)
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else 0
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 20
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            stoch_k = df['Stoch_%K'].iloc[-1] if 'Stoch_%K' in df.columns and not pd.isna(df['Stoch_%K'].iloc[-1]) else 50
           
            # Get SMA values
            sma_values = {}
            for period in [10, 20, 50, 100, 200]:
                col_name = f'SMA_{period}'
                if col_name in df.columns and len(df) >= period:
                    sma_values[period] = df[col_name].iloc[-1]
           
            signal = None
           
            # SVXY SPECIAL LOGIC: Buy on volatility spike drops (mean reversion)
            if ticker == "SVXY" and len(df) >= 5:
                # SVXY drops when VIX spikes - this is a buying opportunity
                five_day_drop = ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
               
                # SVXY Volatility Spike Recovery - 8/10 conviction
                if (five_day_drop < -8 and # Dropped 8%+ in 5 days (VIX spike)
                    volume_ratio > 1.3 and
                    price_change_pct > -1.0): # Not still crashing (stabilizing)
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SVXY Vol Spike Recovery',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Mean Reversion - Vol Spike',
                        'thesis': f"VOLATILITY SPIKE RECOVERY: SVXY dropped {five_day_drop:.1f}% in 5 days (VIX spike). Mean reversion opportunity at ${current_price:.2f}. Markets tend to calm, SVXY recovers. Volume {volume_ratio:.1f}x.",
                        'conviction': 8,
                        'signal_type': 'SVXY Vol Spike Recovery'
                    }
               
                # SVXY Sharp Drop Bounce - 7/10 conviction
                elif (len(df) >= 2 and
                      df['Close'].pct_change().iloc[-2] < -0.03 and # Yesterday dropped 3%+
                      price_change_pct > 0.5 and # Today bouncing up
                      volume_ratio > 1.5):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SVXY Sharp Drop Bounce',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Mean Reversion - Bounce',
                        'thesis': f"SHARP DROP RECOVERY: SVXY bouncing +{price_change_pct:.1f}% after yesterday's {df['Close'].pct_change().iloc[-2]*100:.1f}% drop. VIX likely peaked. Volume {volume_ratio:.1f}x confirms reversal.",
                        'conviction': 7,
                        'signal_type': 'SVXY Sharp Drop Bounce'
                    }
           
            # SVXY CASCADE LOGIC: SMA10 crossing through progressively higher SMAs
            if not signal and ticker == "SVXY":
                has_sma_10 = 10 in sma_values
                has_sma_20 = 20 in sma_values
                has_sma_50 = 50 in sma_values
                has_sma_100 = 100 in sma_values
                has_sma_200 = 200 in sma_values
               
                if has_sma_10 and has_sma_200 and len(df) >= 200:
                    sma_10 = sma_values[10]
                    sma_200 = sma_values[200]
                    sma_10_prev = df['SMA_10'].iloc[-2] if 'SMA_10' in df.columns else 0
                    sma_200_prev = df['SMA_200'].iloc[-2] if 'SMA_200' in df.columns else 0
                   
                    if sma_10 > sma_200 and sma_10_prev <= sma_200_prev and current_price > sma_10:
                        signal = {
                            'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                            'time': now_str,
                            'type': 'SVXY SMA10→200 Cascade',
                            'symbol': ticker,
                            'action': f"BUY 20 shares @ ${current_price:.2f}",
                            'size': 20,
                            'entry_price': current_price,
                            'max_hold': None,
                            'dte': 0,
                            'strategy': f'{ticker} Trend Cascade - Strongest',
                            'thesis': f"SVXY CASCADE (STRONGEST): SMA10 (${sma_10:.2f}) crossed through SMA200 (${sma_200:.2f}). Full trend confirmation. Price ${current_price:.2f}. This is the ultimate confirmation - ADD MAX POSITION.",
                            'conviction': 9,
                            'signal_type': 'SVXY SMA10→200 Cascade'
                        }
               
                if not signal and has_sma_10 and has_sma_100 and len(df) >= 100:
                    sma_10 = sma_values[10]
                    sma_100 = sma_values[100]
                    sma_10_prev = df['SMA_10'].iloc[-2] if 'SMA_10' in df.columns else 0
                    sma_100_prev = df['SMA_100'].iloc[-2] if 'SMA_100' in df.columns else 0
                   
                    if sma_10 > sma_100 and sma_10_prev <= sma_100_prev and current_price > sma_10:
                        signal = {
                            'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                            'time': now_str,
                            'type': 'SVXY SMA10→100 Cascade',
                            'symbol': ticker,
                            'action': f"BUY 18 shares @ ${current_price:.2f}",
                            'size': 18,
                            'entry_price': current_price,
                            'max_hold': None,
                            'dte': 0,
                            'strategy': f'{ticker} Trend Cascade - Very Strong',
                            'thesis': f"SVXY CASCADE: SMA10 (${sma_10:.2f}) crossed through SMA100 (${sma_100:.2f}). Strong intermediate trend. Price ${current_price:.2f}. Consider ADDING to position.",
                            'conviction': 8,
                            'signal_type': 'SVXY SMA10→100 Cascade'
                        }
               
                if not signal and has_sma_10 and has_sma_50 and len(df) >= 50:
                    sma_10 = sma_values[10]
                    sma_50 = sma_values[50]
                    sma_10_prev = df['SMA_10'].iloc[-2] if 'SMA_10' in df.columns else 0
                    sma_50_prev = df['SMA_50'].iloc[-2] if 'SMA_50' in df.columns else 0
                   
                    if sma_10 > sma_50 and sma_10_prev <= sma_50_prev and current_price > sma_10:
                        signal = {
                            'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                            'time': now_str,
                            'type': 'SVXY SMA10→50 Cascade',
                            'symbol': ticker,
                            'action': f"BUY 18 shares @ ${current_price:.2f}",
                            'size': 18,
                            'entry_price': current_price,
                            'max_hold': None,
                            'dte': 0,
                            'strategy': f'{ticker} Trend Cascade - Strong',
                            'thesis': f"SVXY CASCADE: SMA10 (${sma_10:.2f}) crossed through SMA50 (${sma_50:.2f}). Medium-term trend confirmed. Price ${current_price:.2f}. Consider ADDING to existing position.",
                            'conviction': 8,
                            'signal_type': 'SVXY SMA10→50 Cascade'
                        }
               
                if not signal and has_sma_10 and has_sma_20 and len(df) >= 20:
                    sma_10 = sma_values[10]
                    sma_20 = sma_values[20]
                    sma_10_prev = df['SMA_10'].iloc[-2] if 'SMA_10' in df.columns else 0
                    sma_20_prev = df['SMA_20'].iloc[-2] if 'SMA_20' in df.columns else 0
                   
                    if sma_10 > sma_20 and sma_10_prev <= sma_20_prev and current_price > sma_10 and volume_ratio > 1.2:
                        signal = {
                            'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                            'time': now_str,
                            'type': 'SVXY SMA10→20 Cascade',
                            'symbol': ticker,
                            'action': f"BUY 15 shares @ ${current_price:.2f}",
                            'size': 15,
                            'entry_price': current_price,
                            'max_hold': None,
                            'dte': 0,
                            'strategy': f'{ticker} Trend Cascade - Initial',
                            'thesis': f"SVXY CASCADE (INITIAL): SMA10 (${sma_10:.2f}) crossed SMA20 (${sma_20:.2f}). Early trend signal. Price ${current_price:.2f}, Volume {volume_ratio:.1f}x. INITIAL ENTRY - watch for crosses through SMA50, 100, 200 to ADD.",
                            'conviction': 7,
                            'signal_type': 'SVXY SMA10→20 Cascade'
                        }
           
            if not signal and 10 in sma_values and 20 in sma_values and len(df) >= 20:
                sma_10_current = sma_values[10]
                sma_20_current = sma_values[20]
                sma_10_prev = df[f'SMA_10'].iloc[-2] if f'SMA_10' in df.columns else 0
                sma_20_prev = df[f'SMA_20'].iloc[-2] if f'SMA_20' in df.columns else 0
               
                if (sma_10_current > sma_20_current and
                    sma_10_prev <= sma_20_prev and
                    current_price > sma_10_current and
                    volume_ratio > 1.2):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SMA 10/20 Cross',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - SMA 10/20 Cross',
                        'thesis': f"SMA CROSSOVER: {ticker} SMA10 (${sma_10_current:.2f}) crossed above SMA20 (${sma_20_current:.2f}). Short-term momentum shift. Price ${current_price:.2f}, Volume {volume_ratio:.1f}x.",
                        'conviction': 7,
                        'signal_type': 'SMA 10/20 Cross'
                    }
           
            if not signal and 20 in sma_values and 50 in sma_values and len(df) >= 50:
                sma_20_current = sma_values[20]
                sma_50_current = sma_values[50]
                sma_20_prev = df[f'SMA_20'].iloc[-2] if f'SMA_20' in df.columns else 0
                sma_50_prev = df[f'SMA_50'].iloc[-2] if f'SMA_50' in df.columns else 0
               
                if (sma_20_current > sma_50_current and
                    sma_20_prev <= sma_50_prev and
                    current_price > sma_20_current and
                    volume_ratio > 1.2):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SMA 20/50 Cross',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - SMA 20/50 Cross',
                        'thesis': f"MEDIUM-TERM REVERSAL: {ticker} SMA20 (${sma_20_current:.2f}) crossed above SMA50 (${sma_50_current:.2f}). Trend change confirmed. Price ${current_price:.2f}, Volume {volume_ratio:.1f}x.",
                        'conviction': 8,
                        'signal_type': 'SMA 20/50 Cross'
                    }
           
            if not signal and 50 in sma_values and 100 in sma_values and len(df) >= 100:
                sma_50_current = sma_values[50]
                sma_100_current = sma_values[100]
                sma_50_prev = df[f'SMA_50'].iloc[-2] if f'SMA_50' in df.columns else 0
                sma_100_prev = df[f'SMA_100'].iloc[-2] if f'SMA_100' in df.columns else 0
               
                if (sma_50_current > sma_100_current and
                    sma_50_prev <= sma_100_prev and
                    current_price > sma_50_current):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SMA 50/100 Cross',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - SMA 50/100 Cross',
                        'thesis': f"INTERMEDIATE TREND: {ticker} SMA50 (${sma_50_current:.2f}) crossed above SMA100 (${sma_100_current:.2f}). Strong trend shift. Price ${current_price:.2f}.",
                        'conviction': 8,
                        'signal_type': 'SMA 50/100 Cross'
                    }
           
            if not signal and (200 in sma_values and 50 in sma_values and len(df) >= 200):
                sma_50_current = sma_values[50]
                sma_200_current = sma_values[200]
                sma_50_prev = df[f'SMA_50'].iloc[-2]
                sma_200_prev = df[f'SMA_200'].iloc[-2]
               
                if (sma_50_current > sma_200_current and
                    sma_50_prev <= sma_200_prev and
                    current_price > sma_50_current and
                    volume_ratio > 1.2):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'Golden Cross',
                        'symbol': ticker,
                        'action': f"BUY 20 shares @ ${current_price:.2f}",
                        'size': 20,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Golden Cross',
                        'thesis': f"GOLDEN CROSS: {ticker} SMA50 (${sma_50_current:.2f}) crossed above SMA200 (${sma_200_current:.2f}). Price ${current_price:.2f}, Volume {volume_ratio:.1f}x, ADX {adx:.1f}. Historically strongest signal.",
                        'conviction': 9,
                        'signal_type': 'Golden Cross'
                    }
           
            if not signal and (20 in sma_values and 50 in sma_values):
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
               
                if (current_price > sma_20 * 1.005 and
                    sma_20 > sma_50 and
                    volume_ratio > 1.5 and
                    price_change_pct > 0.3 and
                    rsi > 50 and rsi < 70 and
                    adx > 20):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'SMA Breakout',
                        'symbol': ticker,
                        'action': f"BUY 18 shares @ ${current_price:.2f}",
                        'size': 18,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - SMA Breakout',
                        'thesis': f"SMA BREAKOUT: {ticker} ${current_price:.2f} broke above SMA20 (${sma_20:.2f}), uptrend confirmed with SMA20 > SMA50 (${sma_50:.2f}). Volume {volume_ratio:.1f}x, +{price_change_pct:.1f}% momentum, RSI {rsi:.0f}, ADX {adx:.1f}.",
                        'conviction': 8,
                        'signal_type': 'SMA Breakout'
                    }
           
            if not signal and (20 in sma_values):
                sma_20 = sma_values[20]
               
                if (volume_ratio > 2.0 and
                    price_change_pct > 0.5 and
                    current_price > sma_20 and
                    rsi > 55 and rsi < 75 and
                    macd > macd_signal):
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'Volume Breakout',
                        'symbol': ticker,
                        'action': f"BUY 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Long - Volume Breakout',
                        'thesis': f"VOLUME BREAKOUT: {ticker} ${current_price:.2f} with exceptional volume ({volume_ratio:.1f}x avg), +{price_change_pct:.1f}% move. RSI {rsi:.0f}, MACD bullish, above SMA20 (${sma_20:.2f}).",
                        'conviction': 7,
                        'signal_type': 'Volume Breakout'
                    }
           
            if not signal and (rsi < 30 and
                  stoch_k < 20 and
                  price_change_pct > 0.2 and
                  20 in sma_values):
                sma_20 = sma_values[20]
               
                signal = {
                    'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                    'time': now_str,
                    'type': 'Oversold Bounce',
                    'symbol': ticker,
                    'action': f"BUY 12 shares @ ${current_price:.2f}",
                    'size': 12,
                    'entry_price': current_price,
                    'max_hold': None,
                    'dte': 0,
                    'strategy': f'{ticker} Long - Oversold Bounce',
                    'thesis': f"OVERSOLD BOUNCE: {ticker} ${current_price:.2f} showing reversal from oversold. RSI {rsi:.0f}, Stochastic {stoch_k:.0f}, +{price_change_pct:.1f}% bounce. SMA20 at ${sma_20:.2f}.",
                    'conviction': 6,
                    'signal_type': 'Oversold Bounce'
                }
           
            if not signal and (20 in sma_values and 50 in sma_values):
                sma_20 = sma_values[20]
                sma_50 = sma_values[50]
               
                if (current_price < sma_20 * 0.995 and
                    sma_20 < sma_50 and
                    volume_ratio > 1.5 and
                    price_change_pct < -0.3 and
                    rsi < 50 and
                    macd < macd_signal):
                    action_type = "BUY" if ticker == "SVXY" else "SELL SHORT"
                    signal = {
                        'id': f"SIG-{ticker}-{len(st.session_state.signal_queue)+1}-{datetime.now().strftime('%H%M%S')}",
                        'time': now_str,
                        'type': 'Bearish Breakdown',
                        'symbol': ticker,
                        'action': f"{action_type} 15 shares @ ${current_price:.2f}",
                        'size': 15,
                        'entry_price': current_price,
                        'max_hold': None,
                        'dte': 0,
                        'strategy': f'{ticker} Short - Breakdown',
                        'thesis': f"BEARISH BREAKDOWN: {ticker} ${current_price:.2f} broke below SMA20 (${sma_20:.2f}), downtrend confirmed with SMA20 < SMA50 (${sma_50:.2f}). Volume {volume_ratio:.1f}x, {price_change_pct:.1f}% drop, RSI {rsi:.0f}, MACD bearish.",
                        'conviction': 7,
                        'signal_type': 'Bearish Breakdown'
                    }
           
            if signal and np.random.random() < 0.10:
                st.session_state.signal_queue.append(signal)
                save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
                break
               
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

def simulate_exit():
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
       
        if pnl_pct <= STOP_LOSS_PCT:
            exit_triggered = True
            exit_reason = f"Stop Loss (-2%)"
       
        elif pnl_pct >= 4.0:
            max_pnl_reached = trade.get('max_pnl_reached', pnl_pct)
            if pnl_pct > max_pnl_reached:
                trade['max_pnl_reached'] = pnl_pct
            elif (max_pnl_reached - pnl_pct) >= TRAILING_STOP_PCT:
                exit_triggered = True
                exit_reason = f"Trailing Stop (from +{max_pnl_reached:.1f}% to +{pnl_pct:.1f}%)"
       
        try:
            ticker = yf.Ticker(trade['symbol'])
            recent_data = ticker.history(period="5d", interval="1h")
           
            if len(recent_data) >= 20:
                df = calculate_technical_indicators(recent_data)
                volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1.0
                price_change = df['Close'].pct_change().iloc[-1] * 100
               
                if volume_ratio > 2.5 and price_change < -1.0 and pnl_pct > 1.0:
                    exit_triggered = True
                    exit_reason = f"Momentum Reversal (volume {volume_ratio:.1f}x, -{price_change:.1f}%)"
        except:
            pass
       
        if not exit_triggered and pnl_pct < -1.5:
            try:
                ticker = yf.Ticker(trade['symbol'])
                recent_data = ticker.history(period="20d", interval="1d")
               
                if len(recent_data) >= 20:
                    sma_20 = recent_data['Close'].rolling(window=20).mean().iloc[-1]
                   
                    if 'Long' in trade.get('strategy', '') and current_price < sma_20:
                        exit_triggered = True
                        exit_reason = f"SMA20 Break (${current_price:.2f} < ${sma_20:.2f})"
                    elif 'Short' in trade.get('strategy', '') and current_price > sma_20:
                        exit_triggered = True
                        exit_reason = f"SMA20 Break (${current_price:.2f} > ${sma_20:.2f})"
            except:
                pass
       
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

# ==============================
# MAIN APP PAGES
# ==============================

if selected == "Trading Hub":
    st.header("Trading Hub - Multi-Ticker Analysis")
   
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
   
    if is_market_open():
        generate_signal()
        simulate_exit()
   
    st.subheader(f"Trading Signals ({len(st.session_state.signal_queue)} Active)")
   
    for sig in st.session_state.signal_queue:
        st.markdown(f"""
        <div style="background:#ff6b6b;padding:15px;border-radius:10px;text-align:center;">
            <h3>SIGNAL @ {sig['time']}</h3>
            <p><b>{sig['type']}</b> | {sig['symbol']} | {sig['action']} | Conviction: {sig['conviction']}/10</p>
            <p><small>Strategy: {sig['strategy']}</small></p>
            <p><small>Thesis: {sig['thesis']}</small></p>
        </div>
        """, unsafe_allow_html=True)
       
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Take: {sig['id']}", key=f"take_{sig['id']}", use_container_width=True):
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
                    'signal_type': sig.get('signal_type'),
                    'max_pnl_reached': 0
                }
                st.session_state.active_trades.append(trade)
                save_active_trades()
                log_trade(
                    sig['time'], "Open", sig['symbol'], sig['action'], sig['size'],
                    f"${entry_price:.2f}", "Pending", "Open", "Open", sig['id'],
                    entry_numeric=entry_price, dte=sig.get('dte'),
                    strategy=sig['strategy'], thesis=sig['thesis'], max_hold=sig.get('max_hold'),
                    conviction=sig['conviction'], signal_type=sig.get('signal_type')
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
                    strategy=sig['strategy'], thesis=sig['thesis'],
                    conviction=sig['conviction'], signal_type=sig.get('signal_type')
                )
                st.session_state.signal_queue.remove(sig)
                save_json(SIGNAL_QUEUE_FILE, st.session_state.signal_queue)
                st.info("Signal skipped")
                st.rerun()
   
    if st.session_state.active_trades:
        st.subheader("Active Trades")
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
               
                if st.button(f"Close Now", key=f"close_{trade['signal_id']}"):
                    exit_price = current_price
                   
                    if 'SELL SHORT' in trade['action'] or 'Short' in trade.get('strategy', ''):
                        close_action = 'Buy to Cover'
                    else:
                        close_action = 'Sell'
                   
                    now = datetime.now(ZoneInfo("US/Eastern"))
                    log_trade(
                        ts=now.strftime("%m/%d %H:%M"),
                        typ="Close",
                        sym=trade['symbol'],
                        action=close_action,
                        size=trade['size'],
                        entry=f"${entry_price:.2f}",
                        exit=f"${exit_price:.2f}",
                        pnl=f"${pnl:.0f}",
                        status="Closed (Manual)",
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
                    st.success("Trade closed!")
                    st.rerun()
   
    st.divider()
   
    st.subheader("Advanced Price Charts")
   
    chart_ticker = st.selectbox("Select Ticker", TICKERS, key="chart_ticker_select")
   
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["1D", "5D", "1M", "MTD", "QTD", "YTD", "1Y", "3Y", "5Y", "10Y", "MAX", "Custom"],
            index=5,
            key="chart_period"
        )
   
    with col2:
        if time_period == "Custom":
            date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="custom_range"
            )
   
    with col3:
        chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"], key="chart_type")
   
    st.caption("Technical Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_smas = st.multiselect("SMAs", [10, 20, 50, 100, 200], default=[], key="sma_select")
    with col2:
        show_bb = st.checkbox("Bollinger Bands", key="bb_select")
    with col3:
        show_volume = st.checkbox("Volume", value=True, key="vol_select")
   
    @st.cache_data(ttl=300)
    def get_chart_data(ticker, period, custom_dates=None):
        t = yf.Ticker(ticker)
       
        if period == "Custom" and custom_dates:
            start_date, end_date = custom_dates
            data = t.history(start=start_date, end=end_date)
        else:
            now = datetime.now()
           
            if period == "MTD":
                start_date = datetime(now.year, now.month, 1)
                data = t.history(start=start_date, end=now)
            elif period == "QTD":
                quarter_start_month = ((now.month - 1) // 3) * 3 + 1
                start_date = datetime(now.year, quarter_start_month, 1)
                data = t.history(start=start_date, end=now)
            else:
                period_map = {
                    "1D": ("1d", "5m"),
                    "5D": ("5d", "15m"),
                    "1M": ("1mo", "1d"),
                    "YTD": ("ytd", "1d"),
                    "1Y": ("1y", "1d"),
                    "3Y": ("3y", "1wk"),
                    "5Y": ("5y", "1wk"),
                    "10Y": ("10y", "1mo"),
                    "MAX": ("max", "1mo")
                }
               
                yf_period, yf_interval = period_map.get(period, ("5d", "15m"))
                data = t.history(period=yf_period, interval=yf_interval)
       
        if hasattr(data.index, 'dayofweek'):
            data = data[data.index.dayofweek < 5]
       
        return data
   
    try:
        if time_period == "Custom" and 'date_range' in locals():
            chart_data = get_chart_data(chart_ticker, time_period, date_range)
        else:
            chart_data = get_chart_data(chart_ticker, time_period)
       
        if not chart_data.empty:
            chart_data_with_indicators = calculate_technical_indicators(chart_data, periods=show_smas if show_smas else [20])
           
            rows = 2 if show_volume else 1
            row_heights = [0.7, 0.3] if show_volume else [1.0]
           
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=('Price', 'Volume') if show_volume else ('Price',)
            )
           
            if chart_type == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=chart_data_with_indicators.index,
                        y=chart_data_with_indicators['Close'],
                        mode='lines',
                        name=chart_ticker,
                        line=dict(color='#00FFFF', width=3),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            elif chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=chart_data_with_indicators.index,
                        open=chart_data_with_indicators['Open'],
                        high=chart_data_with_indicators['High'],
                        low=chart_data_with_indicators['Low'],
                        close=chart_data_with_indicators['Close'],
                        name=chart_ticker,
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ),
                    row=1, col=1
                )
           
            sma_colors = {
                10: '#0000FF', 20: '#FF0000', 50: '#00CC00', 100: '#FF6600', 200: '#9900CC'
            }
            for sma in show_smas:
                col_name = f'SMA_{sma}'
                if col_name in chart_data_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data_with_indicators.index,
                            y=chart_data_with_indicators[col_name],
                            mode='lines',
                            name=f'SMA{sma}',
                            line=dict(color=sma_colors.get(sma, '#95a5a6'), width=1.5)
                        ),
                        row=1, col=1
                    )
           
            if show_bb and 'BB_Upper' in chart_data_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data_with_indicators.index,
                        y=chart_data_with_indicators['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(250, 128, 114, 0.3)', width=1),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_data_with_indicators.index,
                        y=chart_data_with_indicators['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(250, 128, 114, 0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(250, 128, 114, 0.1)',
                        showlegend=True
                    ),
                    row=1, col=1
                )
           
            current_price = market_data[chart_ticker]['price']
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Current: ${current_price:.2f}",
                row=1, col=1
            )
           
            if show_volume:
                colors = ['red' if chart_data_with_indicators['Close'].iloc[i] < chart_data_with_indicators['Open'].iloc[i] else 'green'
                          for i in range(len(chart_data_with_indicators))]
               
                fig.add_trace(
                    go.Bar(
                        x=chart_data_with_indicators.index,
                        y=chart_data_with_indicators['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5,
                        showlegend=True
                    ),
                    row=2, col=1
                )
           
            fig.update_layout(
                height=700 if show_volume else 600,
                title=f"{chart_ticker} - {time_period}",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                showlegend=True,
                template='plotly_dark'
            )
           
            fig.update_xaxes(title_text="Date", row=rows, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            if show_volume:
                fig.update_yaxes(title_text="Volume", row=2, col=1)
           
            st.plotly_chart(fig, use_container_width=True)
           
            st.subheader("Technical Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
           
            latest = chart_data_with_indicators.iloc[-1]
           
            with col1:
                if 'RSI' in latest and not pd.isna(latest['RSI']):
                    rsi_val = latest['RSI']
                    rsi_color = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "yellow")
                    st.metric("RSI (14)", f"{rsi_val:.1f}", delta="Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
           
            with col2:
                if 'MACD' in latest and not pd.isna(latest['MACD']):
                    macd_val = latest['MACD']
                    macd_signal = latest['MACD_Signal'] if 'MACD_Signal' in latest else 0
                    st.metric("MACD", f"{macd_val:.2f}", delta="Bullish" if macd_val > macd_signal else "Bearish")
           
            with col3:
                if 'ADX' in latest and not pd.isna(latest['ADX']):
                    adx_val = latest['ADX']
                    st.metric("ADX (Trend Strength)", f"{adx_val:.1f}", delta="Strong" if adx_val > 25 else "Weak")
           
            with col4:
                if 'ATR' in latest and not pd.isna(latest['ATR']):
                    atr_val = latest['ATR']
                    st.metric("ATR (Volatility)", f"${atr_val:.2f}")
       
        else:
            st.warning("No data available for the selected period.")
   
    except Exception as e:
        st.error(f"Error loading chart data: {str(e)}")

elif selected == "Options Chain":
    st.header("Options Chain")
    st.caption("Real-time options data with Greeks")
   
    ticker_select = st.selectbox("Select Ticker", TICKERS, key="options_ticker")
   
    try:
        ticker_obj = yf.Ticker(ticker_select)
        available_expirations = ticker_obj.options
       
        if available_expirations:
            exp_with_dte = []
            for exp_date in available_expirations:
                exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
                dte = (exp_datetime - datetime.now()).days
                exp_with_dte.append((exp_date, dte, f"{exp_date} ({dte} DTE)"))
           
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_mode = st.radio("Filter By", ["DTE Range", "Specific Expiration"], horizontal=True)
           
            if filter_mode == "DTE Range":
                with col2:
                    dte_min = st.number_input("Min DTE", 1, 365, 7)
                with col3:
                    dte_max = st.number_input("Max DTE", 1, 365, 60)
               
                selected_expirations = [exp for exp, dte, _ in exp_with_dte if dte_min <= dte <= dte_max]
            else:
                with col2:
                    exp_options = [label for _, _, label in exp_with_dte]
                    selected_label = st.selectbox("Select Expiration", exp_options)
                    selected_expirations = [exp_with_dte[exp_options.index(selected_label)][0]]
                with col3:
                    st.write("")
           
            col1, col2 = st.columns(2)
            with col1:
                opt_type = st.selectbox("Type", ["All", "Calls", "Puts"])
            with col2:
                st.write("")
           
            with st.spinner("Loading options chain..."):
                all_options = []
                for exp_date in selected_expirations:
                    try:
                        chain = ticker_obj.option_chain(exp_date)
                        exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
                        dte = (exp_datetime - datetime.now()).days
                       
                        for opt_type_data, opt_name in [(chain.calls, 'Call'), (chain.puts, 'Put')]:
                            if not opt_type_data.empty:
                                opts = opt_type_data.copy()
                                opts['type'] = opt_name
                                opts['dte'] = dte
                                opts['expiration'] = exp_date
                                opts['mid'] = (opts['bid'] + opts['ask']) / 2
                                all_options.append(opts)
                    except:
                        continue
               
                if all_options:
                    df = pd.concat(all_options, ignore_index=True)
                    df['symbol'] = df['contractSymbol']
                   
                    if opt_type != "All":
                        df = df[df['type'] == opt_type[:-1]]
                   
                    current_price = market_data[ticker_select]['price']
                    df['Distance'] = ((df['strike'] - current_price) / current_price * 100).round(2)
                    df['POP'] = (df['impliedVolatility'] * 100).round(0)
                   
                    df = df.sort_values(['expiration', 'strike'])
                   
                    st.subheader("Smart Filter Options")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        show_all_options = st.checkbox("Show All Options (unfiltered)", value=False, key="show_all_opts")
                    with col2:
                        st.write("")
                   
                    if not show_all_options:
                        lower_bound = current_price * 0.90
                        upper_bound = current_price * 1.10
                       
                        df_filtered = df[(df['strike'] >= lower_bound) & (df['strike'] <= upper_bound)].copy()
                       
                        st.caption(f"Showing strikes within ±10% of current price (${current_price:.2f}): ${lower_bound:.2f} to ${upper_bound:.2f}")
                        st.caption(f"Displaying {len(df_filtered)} of {len(df)} total options. Check 'Show All Options' to see entire chain.")
                       
                        display_df = df_filtered
                    else:
                        st.caption(f"Showing all {len(df)} options contracts")
                        display_df = df
                   
                    display_cols = ['symbol', 'type', 'strike', 'expiration', 'dte', 'lastPrice', 'bid', 'ask', 'mid',
                                   'volume', 'openInterest', 'impliedVolatility', 'delta', 'gamma',
                                   'theta', 'vega', 'Distance', 'POP']
                   
                    available_cols = [col for col in display_cols if col in display_df.columns]
                   
                    st.dataframe(
                        display_df[available_cols],
                        use_container_width=True,
                        height=500
                    )
                else:
                    st.info("No options data available for selected criteria.")
        else:
            st.warning(f"No options available for {ticker_select}")
    except Exception as e:
        st.error(f"Error loading options chain: {str(e)}")

elif selected == "Backtest":
    st.header("Backtest: Enhanced Multi-Signal Strategy")
    st.caption("Real historical data with conviction-based position sizing and no max hold time")
   
    col1, col2 = st.columns(2)
    with col1:
        backtest_mode = st.radio("Backtest Mode", ["Single Ticker", "Portfolio (All Tickers)"], horizontal=True)
    with col2:
        st.write("")
   
    if backtest_mode == "Single Ticker":
        backtest_ticker = st.selectbox("Select Ticker for Backtest", TICKERS, key="backtest_ticker")
        tickers_to_test = [backtest_ticker]
    else:
        tickers_to_test = TICKERS
        st.info(f"Running portfolio backtest across all {len(TICKERS)} tickers: {', '.join(TICKERS)}")
   
    @st.cache_data(ttl=3600)
    def run_enhanced_backtest(ticker):
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
           
            for i in range(200, len(hist)):
                current_time = hist.index[i]
                current_price = hist['Close'].iloc[i]
               
                if not in_position:
                    if ticker == "SVXY":
                        if i >= 5:
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
                                entry_reason = f"SVXY dropped {five_day_drop:.1f}% in 5 days (VIX spike). Mean reversion opportunity. Price ${current_price:.2f}, stabilizing with volume {hist['Volume_Ratio'].iloc[i]:.1f}x"
                                max_gain = 0
                                continue
                           
                            if (hist['Close'].pct_change().iloc[i-1] < -0.03 and
                                hist['Close'].pct_change().iloc[i] > 0.005 and
                                'Volume_Ratio' in hist.columns and hist['Volume_Ratio'].iloc[i] > 1.5):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 7
                                shares = 15
                                signal_type = "SVXY Sharp Drop Bounce"
                                entry_reason = f"SVXY sharp drop recovery. Yesterday -{hist['Close'].pct_change().iloc[i-1]*100:.1f}%, bouncing +{hist['Close'].pct_change().iloc[i]*100:.1f}% with {hist['Volume_Ratio'].iloc[i]:.1f}x volume"
                                max_gain = 0
                                continue
                   
                    if ticker == "SVXY" and not in_position:
                        if ('SMA_10' in hist.columns and 'SMA_200' in hist.columns and
                            len(hist) >= 200 and i > 0):
                            if (hist['SMA_10'].iloc[i] > hist['SMA_200'].iloc[i] and
                                hist['SMA_10'].iloc[i-1] <= hist['SMA_200'].iloc[i-1] and
                                current_price > hist['SMA_10'].iloc[i]):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 9
                                shares = 20
                                signal_type = "SVXY SMA10→200 Cascade"
                                entry_reason = f"SVXY CASCADE (STRONGEST): SMA10 (${hist['SMA_10'].iloc[i]:.2f}) crossed through SMA200 (${hist['SMA_200'].iloc[i]:.2f}). Ultimate confirmation."
                                max_gain = 0
                                continue
                       
                        if ('SMA_10' in hist.columns and 'SMA_100' in hist.columns and
                            len(hist) >= 100 and i > 0):
                            if (hist['SMA_10'].iloc[i] > hist['SMA_100'].iloc[i] and
                                hist['SMA_10'].iloc[i-1] <= hist['SMA_100'].iloc[i-1] and
                                current_price > hist['SMA_10'].iloc[i]):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 8
                                shares = 18
                                signal_type = "SVXY SMA10→100 Cascade"
                                entry_reason = f"SVXY CASCADE: SMA10 (${hist['SMA_10'].iloc[i]:.2f}) crossed through SMA100 (${hist['SMA_100'].iloc[i]:.2f}). Strong intermediate trend. ADD to position."
                                max_gain = 0
                                continue
                       
                        if ('SMA_10' in hist.columns and 'SMA_50' in hist.columns and
                            len(hist) >= 50 and i > 0):
                            if (hist['SMA_10'].iloc[i] > hist['SMA_50'].iloc[i] and
                                hist['SMA_10'].iloc[i-1] <= hist['SMA_50'].iloc[i-1] and
                                current_price > hist['SMA_10'].iloc[i]):
                                in_position = True
                                entry_price = current_price
                                entry_time = current_time
                                conviction = 8
                                shares = 18
                                signal_type = "SVXY SMA10→50 Cascade"
                                entry_reason = f"SVXY CASCADE: SMA10 (${hist['SMA_10'].iloc[i]:.2f}) crossed through SMA50 (${hist['SMA_50'].iloc[i]:.2f}). Medium-term confirmed. ADD to position."
                                max_gain = 0
                                continue
                       
                        if ('SMA_10' in hist.columns and 'SMA_20' in hist.columns and
                            'Volume_Ratio' in hist.columns and len(hist) >= 20 and i > 0):
                            if (hist['SMA_10'].iloc[i] > hist['SMA_20'].iloc[i] and
                                hist['SMA_10'].iloc[i-1] <= hist['SMA_20'].iloc[i-1] and
                                current_price > hist['SMA_10'].iloc[i] and
                                hist['Volume_Ratio'].iloc[i] > 1.2):
                                in_position = True
                                entry_price = current_price
                                entry_time = current
