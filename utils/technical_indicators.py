#!/usr/bin/env python3
"""
Módulo de Indicadores Técnicos
Contiene implementaciones de indicadores técnicos comunes para análisis de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

class TechnicalIndicators:
    """Clase para calcular indicadores técnicos"""
    
    def __init__(self):
        self.logger = logging.getLogger('TechnicalIndicators')
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        Args:
            data: Serie de precios (generalmente close)
            period: Período para el cálculo (default: 14)
            
        Returns:
            Serie con valores RSI (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            data: Serie de precios (generalmente close)
            fast_period: Período para EMA rápida (default: 12)
            slow_period: Período para EMA lenta (default: 26)
            signal_period: Período para línea de señal (default: 9)
            
        Returns:
            Diccionario con MACD line, Signal line y Histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            data: Serie de precios (generalmente close)
            period: Período para la media móvil (default: 20)
            std_dev: Número de desviaciones estándar (default: 2.0)
            
        Returns:
            Diccionario con Upper Band, Middle Band (SMA) y Lower Band
        """
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            k_period: Período para %K (default: 14)
            d_period: Período para %D (default: 3)
            
        Returns:
            Diccionario con %K y %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el cálculo (default: 14)
            
        Returns:
            Serie con valores ATR
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift())
        low_close_prev = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = pd.Series(true_range).rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el cálculo (default: 14)
            
        Returns:
            Serie con valores Williams %R (-100 a 0)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el cálculo (default: 20)
            
        Returns:
            Serie con valores CCI
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum
        
        Args:
            data: Serie de precios
            period: Período para el cálculo (default: 10)
            
        Returns:
            Serie con valores de momentum
        """
        return data - data.shift(period)
    
    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Args:
            data: Serie de precios
            period: Período para el cálculo (default: 10)
            
        Returns:
            Serie con valores ROC en porcentaje
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    def analyze_trend(self, data: pd.DataFrame, ma_periods: list = [10, 20, 50]) -> Dict[str, Any]:
        """
        Análisis de tendencia usando múltiples medias móviles
        
        Args:
            data: DataFrame con datos OHLCV
            ma_periods: Lista de períodos para medias móviles
            
        Returns:
            Diccionario con análisis de tendencia
        """
        close = data['close']
        current_price = close.iloc[-1]
        
        trend_analysis = {
            'current_price': current_price,
            'trend_direction': 'neutral',
            'trend_strength': 0,
            'support_levels': [],
            'resistance_levels': []
        }
        
        # Calcular medias móviles
        mas = {}
        for period in ma_periods:
            mas[f'ma_{period}'] = self.sma(close, period).iloc[-1]
        
        # Determinar tendencia
        ma_values = list(mas.values())
        if all(current_price > ma for ma in ma_values):
            if ma_values == sorted(ma_values, reverse=True):
                trend_analysis['trend_direction'] = 'bullish'
                trend_analysis['trend_strength'] = 3
            else:
                trend_analysis['trend_direction'] = 'bullish'
                trend_analysis['trend_strength'] = 2
        elif all(current_price < ma for ma in ma_values):
            if ma_values == sorted(ma_values):
                trend_analysis['trend_direction'] = 'bearish'
                trend_analysis['trend_strength'] = 3
            else:
                trend_analysis['trend_direction'] = 'bearish'
                trend_analysis['trend_strength'] = 2
        else:
            trend_analysis['trend_strength'] = 1
        
        trend_analysis['moving_averages'] = mas
        
        return trend_analysis
    
    def analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análisis de momentum usando RSI, MACD y Stochastic
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis de momentum
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calcular indicadores
        rsi = self.rsi(close).iloc[-1]
        macd_data = self.macd(close)
        macd_current = macd_data['macd'].iloc[-1]
        signal_current = macd_data['signal'].iloc[-1]
        histogram_current = macd_data['histogram'].iloc[-1]
        
        stoch_data = self.stochastic(high, low, close)
        stoch_k = stoch_data['k'].iloc[-1]
        stoch_d = stoch_data['d'].iloc[-1]
        
        momentum_analysis = {
            'rsi': {
                'value': rsi,
                'signal': 'neutral'
            },
            'macd': {
                'macd': macd_current,
                'signal': signal_current,
                'histogram': histogram_current,
                'signal': 'neutral'
            },
            'stochastic': {
                'k': stoch_k,
                'd': stoch_d,
                'signal': 'neutral'
            },
            'overall_momentum': 'neutral'
        }
        
        # Interpretar RSI
        if rsi > 70:
            momentum_analysis['rsi']['signal'] = 'overbought'
        elif rsi < 30:
            momentum_analysis['rsi']['signal'] = 'oversold'
        
        # Interpretar MACD
        if macd_current > signal_current and histogram_current > 0:
            momentum_analysis['macd']['signal'] = 'bullish'
        elif macd_current < signal_current and histogram_current < 0:
            momentum_analysis['macd']['signal'] = 'bearish'
        
        # Interpretar Stochastic
        if stoch_k > 80 and stoch_d > 80:
            momentum_analysis['stochastic']['signal'] = 'overbought'
        elif stoch_k < 20 and stoch_d < 20:
            momentum_analysis['stochastic']['signal'] = 'oversold'
        
        # Momentum general
        signals = [
            momentum_analysis['rsi']['signal'],
            momentum_analysis['macd']['signal'],
            momentum_analysis['stochastic']['signal']
        ]
        
        bullish_signals = signals.count('bullish') + signals.count('oversold')
        bearish_signals = signals.count('bearish') + signals.count('overbought')
        
        if bullish_signals > bearish_signals:
            momentum_analysis['overall_momentum'] = 'bullish'
        elif bearish_signals > bullish_signals:
            momentum_analysis['overall_momentum'] = 'bearish'
        
        return momentum_analysis
    
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análisis de volatilidad usando Bollinger Bands y ATR
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis de volatilidad
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(close)
        bb_upper = bb_data['upper'].iloc[-1]
        bb_middle = bb_data['middle'].iloc[-1]
        bb_lower = bb_data['lower'].iloc[-1]
        current_price = close.iloc[-1]
        
        # ATR
        atr = self.atr(high, low, close).iloc[-1]
        
        # Posición en Bollinger Bands
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        volatility_analysis = {
            'bollinger_bands': {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower,
                'position': bb_position,
                'signal': 'neutral'
            },
            'atr': {
                'value': atr,
                'volatility_level': 'normal'
            },
            'overall_volatility': 'normal'
        }
        
        # Interpretar Bollinger Bands
        if bb_position > 0.8:
            volatility_analysis['bollinger_bands']['signal'] = 'overbought'
        elif bb_position < 0.2:
            volatility_analysis['bollinger_bands']['signal'] = 'oversold'
        
        # Interpretar ATR (comparar con promedio de 20 períodos)
        atr_avg = self.atr(high, low, close, 20).rolling(20).mean().iloc[-1]
        if atr > atr_avg * 1.5:
            volatility_analysis['atr']['volatility_level'] = 'high'
            volatility_analysis['overall_volatility'] = 'high'
        elif atr < atr_avg * 0.5:
            volatility_analysis['atr']['volatility_level'] = 'low'
            volatility_analysis['overall_volatility'] = 'low'
        
        return volatility_analysis
    
    def comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análisis técnico completo combinando tendencia, momentum y volatilidad
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis técnico completo
        """
        try:
            trend_analysis = self.analyze_trend(data)
            momentum_analysis = self.analyze_momentum(data)
            volatility_analysis = self.analyze_volatility(data)
            
            # Puntuación general
            score = 0
            
            # Puntuación de tendencia
            if trend_analysis['trend_direction'] == 'bullish':
                score += trend_analysis['trend_strength']
            elif trend_analysis['trend_direction'] == 'bearish':
                score -= trend_analysis['trend_strength']
            
            # Puntuación de momentum
            if momentum_analysis['overall_momentum'] == 'bullish':
                score += 2
            elif momentum_analysis['overall_momentum'] == 'bearish':
                score -= 2
            
            # Determinar señal general
            if score >= 3:
                overall_signal = 'strong_buy'
            elif score >= 1:
                overall_signal = 'buy'
            elif score <= -3:
                overall_signal = 'strong_sell'
            elif score <= -1:
                overall_signal = 'sell'
            else:
                overall_signal = 'hold'
            
            return {
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volatility': volatility_analysis,
                'overall_score': score,
                'overall_signal': overall_signal,
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis técnico completo: {e}")
            return {
                'error': str(e),
                'overall_signal': 'hold'
            }