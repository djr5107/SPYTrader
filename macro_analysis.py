# macro_analysis.py - Economic Indicators & Market Regime Detection
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MacroAnalyzer:
    """
    Fetches and analyzes macro/economic indicators to determine market regime.
    Used to filter signals and adjust conviction levels based on broader market environment.
    """
    
    def __init__(self):
        self.vix_level = None
        self.treasury_trend = None
        self.yield_curve_status = None
        self.market_breadth = None
        self.dollar_trend = None
        self.regime_type = None
        self.regime_data = {}
        
    def fetch_macro_data(self):
        """Fetch all macro indicators in one go"""
        try:
            # VIX - Market Fear Gauge
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="60d")
            if not vix_hist.empty:
                vix_current = vix_hist['Close'].iloc[-1]
                vix_sma_20 = vix_hist['Close'].rolling(20).mean().iloc[-1]
                
                if vix_current < 15:
                    self.vix_level = "LOW"
                elif vix_current < 20:
                    self.vix_level = "NORMAL"
                elif vix_current < 30:
                    self.vix_level = "ELEVATED"
                else:
                    self.vix_level = "HIGH"
                
                self.regime_data['vix'] = vix_current
                self.regime_data['vix_trend'] = "RISING" if vix_current > vix_sma_20 else "FALLING"
            
            # Treasury Yields
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="60d")
            if not tnx_hist.empty:
                treasury_10y = tnx_hist['Close'].iloc[-1]
                treasury_10y_sma = tnx_hist['Close'].rolling(20).mean().iloc[-1]
                self.treasury_trend = "RISING" if treasury_10y > treasury_10y_sma else "FALLING"
                self.regime_data['treasury_10y'] = treasury_10y
            
            # 2-Year for Yield Curve
            fvx = yf.Ticker("^FVX")
            fvx_hist = fvx.history(period="5d")
            if not fvx_hist.empty:
                treasury_2y = fvx_hist['Close'].iloc[-1]
                yield_spread = treasury_10y - treasury_2y
                self.yield_curve_status = "INVERTED" if yield_spread < 0 else "NORMAL"
                self.regime_data['yield_curve'] = yield_spread
            
            # Market Breadth - S&P 500 vs 200-day SMA
            spx = yf.Ticker("^GSPC")
            spx_hist = spx.history(period="210d")
            if not spx_hist.empty and len(spx_hist) >= 200:
                spx_current = spx_hist['Close'].iloc[-1]
                spx_sma_200 = spx_hist['Close'].rolling(200).mean().iloc[-1]
                spx_above_200 = spx_current > spx_sma_200
                
                # Russell 2000 for small cap breadth
                rut = yf.Ticker("^RUT")
                rut_hist = rut.history(period="210d")
                if not rut_hist.empty and len(rut_hist) >= 200:
                    rut_current = rut_hist['Close'].iloc[-1]
                    rut_sma_200 = rut_hist['Close'].rolling(200).mean().iloc[-1]
                    rut_above_200 = rut_current > rut_sma_200
                    
                    self.market_breadth = spx_above_200 and rut_above_200
                    self.regime_data['breadth'] = "BROAD" if self.market_breadth else "NARROW"
                else:
                    self.market_breadth = spx_above_200
                    self.regime_data['breadth'] = "PARTIAL"
            
            # Dollar Strength
            try:
                dxy = yf.Ticker("DX-Y.NYB")
                dxy_hist = dxy.history(period="60d")
                if not dxy_hist.empty:
                    dxy_current = dxy_hist['Close'].iloc[-1]
                    dxy_sma_20 = dxy_hist['Close'].rolling(20).mean().iloc[-1]
                    self.dollar_trend = "STRONG" if dxy_current > dxy_sma_20 else "WEAK"
                    self.regime_data['dollar'] = dxy_current
            except:
                self.dollar_trend = "UNKNOWN"
            
            return True
            
        except Exception as e:
            print(f"Error fetching macro data: {str(e)}")
            # Set defaults if fetch fails
            self.vix_level = "NORMAL"
            self.treasury_trend = "RISING"
            self.yield_curve_status = "NORMAL"
            self.market_breadth = True
            self.dollar_trend = "NEUTRAL"
            return False
    
    def detect_regime(self):
        """
        Classify current market environment based on macro indicators.
        Returns dict with regime info and trading recommendations.
        """
        
        # Ensure we have data
        if not self.vix_level:
            self.fetch_macro_data()
        
        # RISK-ON (Bull Market Conditions)
        if (self.vix_level in ["LOW", "NORMAL"] and
            self.market_breadth == True and
            self.yield_curve_status == "NORMAL"):
            
            self.regime_type = "RISK_ON"
            return {
                'type': 'RISK_ON',
                'environment': 'Bull Market',
                'equity_bias': 'LONG',
                'preferred_tickers': ['QQQ', 'SPY', 'SVXY'],
                'avoid_tickers': [],
                'conviction_boost': +1,
                'description': 'Low volatility, healthy breadth, normal yield curve'
            }
        
        # RISK-OFF (Bear Market / Crisis)
        elif (self.vix_level in ["HIGH", "ELEVATED"] or
              self.yield_curve_status == "INVERTED" or
              self.market_breadth == False):
            
            self.regime_type = "RISK_OFF"
            return {
                'type': 'RISK_OFF',
                'environment': 'Bear Market',
                'equity_bias': 'DEFENSIVE',
                'preferred_tickers': ['TLT', 'AGG'],
                'avoid_tickers': ['SVXY', 'EEM'],
                'conviction_boost': -2,
                'description': 'High volatility or inverted curve - defensive posture'
            }
        
        # ROTATION (Transitioning / Mixed Signals)
        elif self.vix_level == "ELEVATED":
            self.regime_type = "ROTATION"
            return {
                'type': 'ROTATION',
                'environment': 'Market Rotation',
                'equity_bias': 'SELECTIVE',
                'preferred_tickers': ['QQQ', 'SPY'],  # Quality only
                'avoid_tickers': ['SVXY', 'EEM'],  # Avoid vol and EM
                'conviction_boost': 0,
                'description': 'Elevated volatility - focus on quality'
            }
        
        # NORMAL (Balanced / Mixed)
        else:
            self.regime_type = "NORMAL"
            return {
                'type': 'NORMAL',
                'environment': 'Normal Market',
                'equity_bias': 'BALANCED',
                'preferred_tickers': ['SPY', 'QQQ', 'EFA', 'EEM'],
                'avoid_tickers': [],
                'conviction_boost': 0,
                'description': 'Balanced conditions - all assets viable'
            }
    
    def get_regime_summary(self):
        """Return a formatted summary of current macro environment"""
        regime = self.detect_regime()
        summary = f"""
**Market Regime: {regime['environment']}**
- VIX Level: {self.vix_level} ({self.regime_data.get('vix', 'N/A'):.1f})
- Treasury Trend: {self.treasury_trend}
- Yield Curve: {self.yield_curve_status}
- Market Breadth: {self.regime_data.get('breadth', 'Unknown')}
- Equity Bias: {regime['equity_bias']}
- Preferred Tickers: {', '.join(regime['preferred_tickers'])}
"""
        if regime['avoid_tickers']:
            summary += f"- Avoid: {', '.join(regime['avoid_tickers'])}\n"
        
        return summary


def calculate_dynamic_stop_loss(current_pnl_pct, conviction, days_held):
    """
    Calculate dynamic stop loss that adapts to current P&L and position characteristics.
    
    Key principles:
    1. Tighten stops as gains increase (protect profits)
    2. Looser stops for high conviction (more room to work)
    3. Looser stops for longer holds (don't shake out winners)
    
    Args:
        current_pnl_pct: Current P&L as percentage
        conviction: Signal conviction score (1-10)
        days_held: Number of days position has been held
    
    Returns:
        Stop loss percentage (negative value)
    """
    
    # Base stop loss by P&L stage
    if current_pnl_pct <= 2.0:
        # Stage 1: Small loss or tiny gain
        base_stop = -2.0
    elif current_pnl_pct <= 5.0:
        # Stage 2: Small gain - tighten a bit
        base_stop = -1.5
    elif current_pnl_pct <= 10.0:
        # Stage 3: Moderate gain - tighten more
        base_stop = -1.0
    elif current_pnl_pct <= 15.0:
        # Stage 4: Large gain - very tight
        base_stop = -0.5
    else:
        # Stage 5: Huge gain - protect everything
        base_stop = 0.0  # Breakeven stop
    
    # Conviction adjustment
    # Lower conviction = tighter stops (less room for error)
    # Higher conviction = looser stops (more room to work)
    if conviction <= 6:
        conviction_multiplier = 1.2  # 20% tighter
    elif conviction >= 9:
        conviction_multiplier = 0.85  # 15% looser
    else:
        conviction_multiplier = 1.0  # No adjustment
    
    # Time adjustment - Give longer-held positions more room
    # Prevents being shaken out of long-term winners during normal volatility
    if days_held >= 90:  # 3 months
        time_multiplier = 0.7  # 30% looser
    elif days_held >= 60:  # 2 months
        time_multiplier = 0.8  # 20% looser
    elif days_held >= 30:  # 1 month
        time_multiplier = 0.9  # 10% looser
    else:
        time_multiplier = 1.0  # No adjustment
    
    # Calculate final stop
    dynamic_stop = base_stop * conviction_multiplier * time_multiplier
    
    # Ensure reasonable bounds
    # Never tighter than -0.3% (too tight, gets stopped out by noise)
    # Never looser than -5% (too loose, losses too large)
    dynamic_stop = max(-5.0, min(-0.3, dynamic_stop))
    
    return dynamic_stop


def detect_drawdown_regime(ticker_history):
    """
    Detect if ticker is in a significant drawdown from recent highs.
    Used to adjust signal generation during market stress.
    
    Returns: "MAJOR_DRAWDOWN", "MODERATE_DRAWDOWN", or "NORMAL"
    """
    if len(ticker_history) < 60:
        return "NORMAL"
    
    try:
        current_price = ticker_history['Close'].iloc[-1]
        high_60_day = ticker_history['Close'].rolling(60).max().iloc[-1]
        
        drawdown_pct = ((current_price - high_60_day) / high_60_day) * 100
        
        if drawdown_pct < -10:
            return "MAJOR_DRAWDOWN"
        elif drawdown_pct < -5:
            return "MODERATE_DRAWDOWN"
        else:
            return "NORMAL"
    except:
        return "NORMAL"


def calculate_recovery_speed(ticker_history):
    """
    Determine if recovery from drawdown is swift (V-shape) or slow.
    Swift recoveries are strong buy signals (April 2025 type scenario).
    
    Returns: "SWIFT", "NORMAL", or "SLOW"
    """
    if len(ticker_history) < 20:
        return "NORMAL"
    
    try:
        # Compare recent 5-day return vs 20-day return
        return_5d = ticker_history['Close'].pct_change(5).iloc[-1]
        return_20d = ticker_history['Close'].pct_change(20).iloc[-1]
        
        # Swift: Down significantly over 20 days, up sharply in last 5 days
        if return_5d > 0.03 and return_20d < -0.05:
            return "SWIFT"
        # Normal: Both positive or both negative but moderate
        elif return_5d > 0 and return_20d > 0:
            return "NORMAL"
        # Slow: Still declining or barely recovering
        else:
            return "SLOW"
    except:
        return "NORMAL"


def adjust_signal_for_macro(signal, regime_info, ticker_history):
    """
    Adjust signal conviction and thesis based on macro environment.
    
    Args:
        signal: Original signal dict
        regime_info: Current market regime from detect_regime()
        ticker_history: Price history for drawdown analysis
    
    Returns:
        Modified signal dict
    """
    if not signal:
        return None
    
    ticker = signal['symbol']
    original_conviction = signal['conviction']
    
    # Skip tickers that should be avoided in this regime
    if ticker in regime_info['avoid_tickers']:
        return None  # Don't take this signal
    
    # Apply regime conviction boost/penalty
    signal['conviction'] += regime_info['conviction_boost']
    
    # Extra boost for QQQ (best historical performer)
    if ticker == "QQQ" and regime_info['equity_bias'] in ['LONG', 'BALANCED']:
        signal['conviction'] += 1
        signal['thesis'] += " | QQQ FOCUS: Best historical performer"
    
    # Drawdown analysis
    drawdown_regime = detect_drawdown_regime(ticker_history)
    recovery_speed = calculate_recovery_speed(ticker_history)
    
    if drawdown_regime == "MAJOR_DRAWDOWN":
        if recovery_speed == "SWIFT":
            # April 2025 scenario - Major dip with swift recovery
            # This is a STRONG buy signal
            signal['conviction'] += 2
            signal['thesis'] += " | OPPORTUNITY: Swift recovery from major dip"
        else:
            # Major drawdown without recovery confirmation
            signal['conviction'] -= 1
            signal['thesis'] += " | CAUTION: Major drawdown, slow recovery"
    
    # Clamp conviction to valid range
    signal['conviction'] = max(1, min(10, signal['conviction']))
    
    # Add macro context to thesis
    signal['thesis'] += f" | MACRO: {regime_info['environment']}"
    
    return signal


# Example usage functions
def should_take_signal(signal, min_conviction=6):
    """Filter function to determine if signal meets minimum standards"""
    if not signal:
        return False
    return signal['conviction'] >= min_conviction


def calculate_position_size(signal, base_shares=20):
    """Calculate position size based on conviction"""
    # Simple mapping: 
    # Conviction 10: 20 shares
    # Conviction 9: 18 shares  
    # Conviction 8: 15 shares
    # Conviction 7: 12 shares
    # Conviction 6: 10 shares
    
    conviction = signal['conviction']
    if conviction >= 10:
        return 20
    elif conviction >= 9:
        return 18
    elif conviction >= 8:
        return 15
    elif conviction >= 7:
        return 12
    elif conviction >= 6:
        return 10
    else:
        return 8  # Minimum
