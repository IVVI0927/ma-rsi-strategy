"""Comprehensive technical indicators library with 20+ indicators"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
import talib
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Comprehensive technical indicators for A-Share market analysis"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            prices: Close prices
            period: RSI period (default 14)
            
        Returns:
            RSI values (0-100)
        """
        try:
            return talib.RSI(prices.values, timeperiod=period)
        except:
            # Fallback implementation
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD - Moving Average Convergence Divergence
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        try:
            macd_line, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(macd_line, index=prices.index), pd.Series(signal_line, index=prices.index), pd.Series(histogram, index=prices.index)
        except:
            # Fallback implementation
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return pd.Series(upper, index=prices.index), pd.Series(middle, index=prices.index), pd.Series(lower, index=prices.index)
        except:
            # Fallback implementation
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Returns:
            Tuple of (%K, %D)
        """
        try:
            k, d = talib.STOCH(high.values, low.values, close.values, 
                              fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return pd.Series(k, index=close.index), pd.Series(d, index=close.index)
        except:
            # Fallback implementation
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        try:
            return talib.WILLR(high.values, low.values, close.values, timeperiod=period)
        except:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            return williams_r
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        try:
            return talib.CCI(high.values, low.values, close.values, timeperiod=period)
        except:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        try:
            return talib.ATR(high.values, low.values, close.values, timeperiod=period)
        except:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        try:
            return talib.ADX(high.values, low.values, close.values, timeperiod=period)
        except:
            # Simplified ADX calculation
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            atr_val = TechnicalIndicators.atr(high, low, close, period)
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        try:
            return talib.OBV(close.values, volume.values)
        except:
            price_change = close.diff()
            volume_direction = np.where(price_change > 0, volume, 
                                      np.where(price_change < 0, -volume, 0))
            obv = pd.Series(volume_direction, index=close.index).cumsum()
            return obv
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        try:
            return talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period)
        except:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price.diff() > 0, 0)
            negative_flow = money_flow.where(typical_price.diff() < 0, 0)
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfr = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + mfr))
            return mfi
    
    @staticmethod
    def roc(prices: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change"""
        try:
            return talib.ROC(prices.values, timeperiod=period)
        except:
            roc = (prices / prices.shift(period) - 1) * 100
            return roc
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Momentum"""
        try:
            return talib.MOM(prices.values, timeperiod=period)
        except:
            momentum = prices - prices.shift(period)
            return momentum
    
    @staticmethod
    def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                           period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator"""
        try:
            return talib.ULTOSC(high.values, low.values, close.values, 
                               timeperiod1=period1, timeperiod2=period2, timeperiod3=period3)
        except:
            # Simplified calculation
            bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
            tr = pd.concat([high - low, 
                           np.abs(high - close.shift()),
                           np.abs(low - close.shift())], axis=1).max(axis=1)
            
            avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
            
            uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            return uo
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        ema = close.ewm(span=period).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)
        
        return upper, ema, lower
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Aroon Indicator"""
        try:
            aroon_down, aroon_up = talib.AROON(high.values, low.values, timeperiod=period)
            return pd.Series(aroon_up, index=high.index), pd.Series(aroon_down, index=high.index)
        except:
            aroon_up = high.rolling(window=period).apply(lambda x: (period - np.argmax(x)) / period * 100)
            aroon_down = low.rolling(window=period).apply(lambda x: (period - np.argmin(x)) / period * 100)
            return aroon_up, aroon_down
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        try:
            return talib.SAR(high.values, low.values, acceleration=af_start, maximum=af_max)
        except:
            # Simplified SAR calculation
            sar = pd.Series(index=high.index, dtype=float)
            trend = pd.Series(index=high.index, dtype=int)
            af = pd.Series(index=high.index, dtype=float)
            ep = pd.Series(index=high.index, dtype=float)
            
            # Initialize
            sar.iloc[0] = low.iloc[0]
            trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
            af.iloc[0] = af_start
            ep.iloc[0] = high.iloc[0]
            
            for i in range(1, len(high)):
                if trend.iloc[i-1] == 1:  # Uptrend
                    sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                    if low.iloc[i] <= sar.iloc[i]:
                        trend.iloc[i] = -1
                        sar.iloc[i] = ep.iloc[i-1]
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = af_start
                    else:
                        trend.iloc[i] = 1
                        if high.iloc[i] > ep.iloc[i-1]:
                            ep.iloc[i] = high.iloc[i]
                            af.iloc[i] = min(af.iloc[i-1] + af_start, af_max)
                        else:
                            ep.iloc[i] = ep.iloc[i-1]
                            af.iloc[i] = af.iloc[i-1]
                else:  # Downtrend
                    sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                    if high.iloc[i] >= sar.iloc[i]:
                        trend.iloc[i] = 1
                        sar.iloc[i] = ep.iloc[i-1]
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = af_start
                    else:
                        trend.iloc[i] = -1
                        if low.iloc[i] < ep.iloc[i-1]:
                            ep.iloc[i] = low.iloc[i]
                            af.iloc[i] = min(af.iloc[i-1] + af_start, af_max)
                        else:
                            ep.iloc[i] = ep.iloc[i-1]
                            af.iloc[i] = af.iloc[i-1]
            
            return sar
    
    @staticmethod
    def linear_regression_slope(prices: pd.Series, period: int = 14) -> pd.Series:
        """Linear Regression Slope"""
        try:
            return talib.LINEARREG_SLOPE(prices.values, timeperiod=period)
        except:
            def calc_slope(y):
                x = np.arange(len(y))
                if len(y) >= 2:
                    slope, _, _, _, _ = linregress(x, y)
                    return slope
                return 0
            
            slope = prices.rolling(window=period).apply(calc_slope)
            return slope
    
    @staticmethod
    def mass_index(high: pd.Series, low: pd.Series, period: int = 9, sum_period: int = 25) -> pd.Series:
        """Mass Index"""
        hl_ratio = (high - low)
        ema9 = hl_ratio.ewm(span=period).mean()
        ema9_ema9 = ema9.ewm(span=period).mean()
        mass_index = (ema9 / ema9_ema9).rolling(window=sum_period).sum()
        return mass_index
    
    @staticmethod
    def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        vm_plus = np.abs(high - low.shift())
        vm_minus = np.abs(low - high.shift())
        tr = pd.concat([high - low, 
                       np.abs(high - close.shift()),
                       np.abs(low - close.shift())], axis=1).max(axis=1)
        
        vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        
        return vi_plus, vi_minus
    
    @staticmethod
    def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                          fast: int = 3, slow: int = 10) -> pd.Series:
        """Chaikin Oscillator"""
        ad_line = TechnicalIndicators.accumulation_distribution_line(high, low, close, volume)
        chaikin_osc = ad_line.ewm(span=fast).mean() - ad_line.ewm(span=slow).mean()
        return chaikin_osc
    
    @staticmethod
    def accumulation_distribution_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        try:
            return talib.AD(high.values, low.values, close.values, volume.values)
        except:
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)
            ad_line = (clv * volume).cumsum()
            return ad_line
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a given DataFrame"""
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        result_df = df.copy()
        
        try:
            # Trend indicators
            result_df['SMA_5'] = df['Close'].rolling(5).mean()
            result_df['SMA_10'] = df['Close'].rolling(10).mean()
            result_df['SMA_20'] = df['Close'].rolling(20).mean()
            result_df['SMA_50'] = df['Close'].rolling(50).mean()
            result_df['EMA_12'] = df['Close'].ewm(span=12).mean()
            result_df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Oscillators
            result_df['RSI'] = TechnicalIndicators.rsi(df['Close'])
            macd, signal, histogram = TechnicalIndicators.macd(df['Close'])
            result_df['MACD'] = macd
            result_df['MACD_Signal'] = signal
            result_df['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            result_df['BB_Upper'] = bb_upper
            result_df['BB_Middle'] = bb_middle
            result_df['BB_Lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.stochastic(df['High'], df['Low'], df['Close'])
            result_df['Stoch_K'] = stoch_k
            result_df['Stoch_D'] = stoch_d
            
            # Volume indicators
            result_df['OBV'] = TechnicalIndicators.obv(df['Close'], df['Volume'])
            if 'High' in df.columns and 'Low' in df.columns:
                result_df['MFI'] = TechnicalIndicators.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volatility indicators
            result_df['ATR'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
            
            # Other indicators
            result_df['Williams_R'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
            result_df['CCI'] = TechnicalIndicators.cci(df['High'], df['Low'], df['Close'])
            result_df['ROC'] = TechnicalIndicators.roc(df['Close'])
            result_df['Momentum'] = TechnicalIndicators.momentum(df['Close'])
            
        except Exception as e:
            print(f"Warning: Some indicators could not be calculated: {e}")
        
        return result_df