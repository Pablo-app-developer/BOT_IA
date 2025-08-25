#!/usr/bin/env python3
"""
Motor de IA Avanzado para Trading Profesional
Combina múltiples técnicas de análisis para toma de decisiones inteligente
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import talib
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import joblib
import os

warnings.filterwarnings('ignore')

class SignalStrength(Enum):
    """Enum para la fuerza de las señales"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class MarketRegime(Enum):
    """Enum para regímenes de mercado"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class TradingSignal:
    """Clase para representar señales de trading"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 - 1.0
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: List[str]
    market_regime: MarketRegime
    timestamp: datetime

class AdvancedAIEngine:
    """
    Motor de IA Avanzado que combina múltiples técnicas de análisis:
    - Análisis técnico multi-timeframe
    - Machine Learning ensemble
    - Análisis de sentimiento
    - Detección de regímenes de mercado
    - Gestión de riesgo adaptativa
    - Correlaciones entre pares
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Modelos de ML
        self.ensemble_model = None
        self.regime_detector = None
        self.volatility_predictor = None
        
        # Escaladores
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Configuración
        self.lookback_periods = [5, 10, 20, 50, 100, 200]
        self.timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        self.correlation_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
        # Cache para optimización
        self.feature_cache = {}
        self.correlation_cache = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializar los modelos de ML"""
        # Ensemble de clasificadores para señales
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_classifier = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('gb', gb_classifier),
                ('mlp', mlp_classifier)
            ],
            voting='soft'
        )
        
        # Detector de régimen de mercado
        self.regime_detector = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Predictor de volatilidad
        self.volatility_predictor = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    
    def create_advanced_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Crear features avanzados para el análisis"""
        try:
            if len(df) < 200:
                self.logger.warning(f"Datos insuficientes para {symbol}: {len(df)} filas")
                return None
            
            features_df = df.copy()
            
            # Convertir a arrays numpy
            high = df['high'].astype(float).values
            low = df['low'].astype(float).values
            close = df['close'].astype(float).values
            volume = df['tick_volume'].astype(float).values if 'tick_volume' in df.columns else np.ones(len(df))
            
            # === FEATURES TÉCNICOS AVANZADOS ===
            
            # 1. Múltiples medias móviles y cruces
            for period in self.lookback_periods:
                if len(df) > period:
                    features_df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                    features_df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                    features_df[f'wma_{period}'] = talib.WMA(close, timeperiod=period)
                    
                    # Distancia relativa a las medias
                    features_df[f'price_to_sma_{period}'] = (close / features_df[f'sma_{period}'] - 1) * 100
                    features_df[f'price_to_ema_{period}'] = (close / features_df[f'ema_{period}'] - 1) * 100
            
            # 2. Indicadores de momentum avanzados
            features_df['rsi_14'] = talib.RSI(close, timeperiod=14)
            features_df['rsi_21'] = talib.RSI(close, timeperiod=21)
            features_df['stoch_k'], features_df['stoch_d'] = talib.STOCH(high, low, close)
            features_df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            features_df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            features_df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # 3. Indicadores de volatilidad
            features_df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features_df['atr_21'] = talib.ATR(high, low, close, timeperiod=21)
            features_df['natr'] = talib.NATR(high, low, close, timeperiod=14)
            
            # Bollinger Bands múltiples
            for period, std in [(20, 2), (20, 2.5), (50, 2)]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std)
                features_df[f'bb_upper_{period}_{std}'] = bb_upper
                features_df[f'bb_lower_{period}_{std}'] = bb_lower
                features_df[f'bb_position_{period}_{std}'] = (close - bb_lower) / (bb_upper - bb_lower)
                features_df[f'bb_width_{period}_{std}'] = (bb_upper - bb_lower) / bb_middle
            
            # 4. MACD avanzado
            macd, macd_signal, macd_hist = talib.MACD(close)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist
            features_df['macd_divergence'] = macd - macd_signal
            
            # 5. Patrones de velas
            features_df['doji'] = talib.CDLDOJI(high, low, close, close)
            features_df['hammer'] = talib.CDLHAMMER(high, low, close, close)
            features_df['engulfing'] = talib.CDLENGULFING(high, low, close, close)
            features_df['harami'] = talib.CDLHARAMI(high, low, close, close)
            features_df['shooting_star'] = talib.CDLSHOOTINGSTAR(high, low, close, close)
            
            # 6. Análisis de volumen
            features_df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features_df['volume_ratio'] = volume / features_df['volume_sma_20']
            features_df['obv'] = talib.OBV(close, volume)
            features_df['ad'] = talib.AD(high, low, close, volume)
            
            # 7. Features de precio y retornos
            for period in [1, 3, 5, 10, 20]:
                features_df[f'return_{period}'] = df['close'].pct_change(period)
                features_df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
                features_df[f'volatility_{period}'] = features_df[f'return_{period}'].rolling(period).std()
            
            # 8. Niveles de soporte y resistencia
            features_df['pivot'] = (high + low + close) / 3
            features_df['resistance_1'] = 2 * features_df['pivot'] - low
            features_df['support_1'] = 2 * features_df['pivot'] - high
            
            # 9. Análisis de tendencia
            features_df['trend_strength'] = self._calculate_trend_strength(close)
            features_df['trend_direction'] = self._calculate_trend_direction(close)
            
            # 10. Features de tiempo
            features_df['hour'] = pd.to_datetime(features_df.index).hour
            features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
            features_df['is_london_session'] = ((features_df['hour'] >= 8) & (features_df['hour'] <= 17)).astype(int)
            features_df['is_ny_session'] = ((features_df['hour'] >= 13) & (features_df['hour'] <= 22)).astype(int)
            features_df['is_overlap'] = ((features_df['hour'] >= 13) & (features_df['hour'] <= 17)).astype(int)
            
            # Limpiar NaN
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creando features para {symbol}: {e}")
            return None
    
    def _calculate_trend_strength(self, close: np.ndarray) -> np.ndarray:
        """Calcular la fuerza de la tendencia"""
        try:
            # Usar ADX como base
            high = close  # Simplificación
            low = close
            adx = talib.ADX(high, low, close, timeperiod=14)
            return adx
        except:
            return np.zeros(len(close))
    
    def _calculate_trend_direction(self, close: np.ndarray) -> np.ndarray:
        """Calcular la dirección de la tendencia"""
        try:
            # Usar pendiente de regresión lineal
            trend_direction = np.zeros(len(close))
            window = 20
            
            for i in range(window, len(close)):
                y = close[i-window:i]
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                trend_direction[i] = slope
            
            return trend_direction
        except:
            return np.zeros(len(close))
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detectar el régimen actual del mercado"""
        try:
            if len(df) < 50:
                return MarketRegime.SIDEWAYS
            
            # Calcular métricas de régimen
            close = df['close'].values
            returns = np.diff(np.log(close))
            
            # Volatilidad
            volatility = np.std(returns[-20:]) * np.sqrt(252)
            
            # Tendencia
            trend_slope = np.polyfit(range(20), close[-20:], 1)[0]
            
            # Clasificar régimen
            if volatility > 0.25:  # Alta volatilidad
                return MarketRegime.VOLATILE
            elif volatility < 0.10:  # Baja volatilidad
                return MarketRegime.LOW_VOLATILITY
            elif trend_slope > 0.001:  # Tendencia alcista
                return MarketRegime.TRENDING_UP
            elif trend_slope < -0.001:  # Tendencia bajista
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Error detectando régimen: {e}")
            return MarketRegime.SIDEWAYS
    
    def analyze_correlations(self, symbol: str) -> Dict[str, float]:
        """Analizar correlaciones con otros pares"""
        correlations = {}
        
        try:
            # Obtener datos del símbolo principal
            main_data = self._get_correlation_data(symbol)
            if main_data is None:
                return correlations
            
            # Calcular correlaciones con otros pares
            for pair in self.correlation_pairs:
                if pair != symbol:
                    pair_data = self._get_correlation_data(pair)
                    if pair_data is not None:
                        correlation = np.corrcoef(main_data, pair_data)[0, 1]
                        correlations[pair] = correlation
            
        except Exception as e:
            self.logger.error(f"Error calculando correlaciones: {e}")
        
        return correlations
    
    def _get_correlation_data(self, symbol: str, periods: int = 100) -> Optional[np.ndarray]:
        """Obtener datos para análisis de correlación"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, periods)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                return df['close'].pct_change().dropna().values
        except:
            pass
        return None
    
    def calculate_risk_metrics(self, df: pd.DataFrame, signal: TradingSignal) -> Dict[str, float]:
        """Calcular métricas de riesgo avanzadas"""
        try:
            close = df['close'].values
            returns = np.diff(np.log(close))
            
            # VaR (Value at Risk)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Volatilidad
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio estimado
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Máximo drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'risk_score': self._calculate_risk_score(var_95, volatility, max_drawdown)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de riesgo: {e}")
            return {}
    
    def _calculate_risk_score(self, var: float, volatility: float, max_dd: float) -> float:
        """Calcular puntuación de riesgo compuesta (0-100)"""
        try:
            # Normalizar métricas (valores más altos = mayor riesgo)
            var_score = min(abs(var) * 1000, 100)  # Normalizar VaR
            vol_score = min(volatility * 100, 100)  # Normalizar volatilidad
            dd_score = min(abs(max_dd) * 100, 100)  # Normalizar drawdown
            
            # Promedio ponderado
            risk_score = (var_score * 0.4 + vol_score * 0.4 + dd_score * 0.2)
            return min(risk_score, 100)
            
        except:
            return 50  # Riesgo medio por defecto
    
    def generate_advanced_signal(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Generar señal de trading avanzada usando IA"""
        try:
            # Crear features avanzados
            features_df = self.create_advanced_features(df, symbol)
            if features_df is None:
                return None
            
            # Detectar régimen de mercado
            market_regime = self.detect_market_regime(features_df)
            
            # Analizar correlaciones
            correlations = self.analyze_correlations(symbol)
            
            # Obtener señal base
            base_signal = self._get_base_signal(features_df)
            if base_signal is None:
                return None
            
            # Ajustar señal según régimen y correlaciones
            adjusted_signal = self._adjust_signal_for_regime(base_signal, market_regime, correlations)
            
            # Calcular métricas de riesgo
            risk_metrics = self.calculate_risk_metrics(features_df, adjusted_signal)
            
            # Validar señal final
            final_signal = self._validate_and_enhance_signal(adjusted_signal, risk_metrics, features_df)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generando señal avanzada: {e}")
            return None
    
    def _get_base_signal(self, features_df: pd.DataFrame) -> Optional[TradingSignal]:
        """Obtener señal base usando análisis técnico"""
        try:
            current = features_df.iloc[-1]
            
            # Análisis de múltiples indicadores
            signals = []
            reasoning = []
            
            # 1. Análisis de medias móviles
            if 'sma_20' in current and 'sma_50' in current:
                if current['sma_20'] > current['sma_50'] and current['close'] > current['sma_20']:
                    signals.append(1)  # Bullish
                    reasoning.append("Tendencia alcista confirmada por SMA 20>50 y precio>SMA20")
                elif current['sma_20'] < current['sma_50'] and current['close'] < current['sma_20']:
                    signals.append(-1)  # Bearish
                    reasoning.append("Tendencia bajista confirmada por SMA 20<50 y precio<SMA20")
                else:
                    signals.append(0)  # Neutral
            
            # 2. Análisis RSI
            if 'rsi_14' in current:
                if current['rsi_14'] < 30:
                    signals.append(1)  # Oversold
                    reasoning.append(f"RSI sobreventa: {current['rsi_14']:.1f}")
                elif current['rsi_14'] > 70:
                    signals.append(-1)  # Overbought
                    reasoning.append(f"RSI sobrecompra: {current['rsi_14']:.1f}")
                else:
                    signals.append(0)
            
            # 3. Análisis MACD
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] > current['macd_signal'] and current['macd_histogram'] > 0:
                    signals.append(1)
                    reasoning.append("MACD bullish: línea MACD > señal")
                elif current['macd'] < current['macd_signal'] and current['macd_histogram'] < 0:
                    signals.append(-1)
                    reasoning.append("MACD bearish: línea MACD < señal")
                else:
                    signals.append(0)
            
            # 4. Análisis Bollinger Bands
            if 'bb_position_20_2' in current:
                bb_pos = current['bb_position_20_2']
                if bb_pos < 0.2:  # Cerca del límite inferior
                    signals.append(1)
                    reasoning.append(f"Precio cerca de BB inferior: {bb_pos:.2f}")
                elif bb_pos > 0.8:  # Cerca del límite superior
                    signals.append(-1)
                    reasoning.append(f"Precio cerca de BB superior: {bb_pos:.2f}")
                else:
                    signals.append(0)
            
            # Calcular señal consenso
            if len(signals) == 0:
                return None
            
            signal_sum = sum(signals)
            signal_strength = abs(signal_sum) / len(signals)
            
            # Determinar acción
            if signal_sum > 0:
                action = "BUY"
            elif signal_sum < 0:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Calcular niveles de precio
            current_price = current['close']
            atr = current.get('atr_14', current_price * 0.01)  # 1% si no hay ATR
            
            if action == "BUY":
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
            elif action == "SELL":
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Calcular ratio riesgo/beneficio
            if action != "HOLD":
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
            else:
                rr_ratio = 0
            
            # Determinar fuerza de señal
            if signal_strength >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif signal_strength >= 0.6:
                strength = SignalStrength.STRONG
            elif signal_strength >= 0.4:
                strength = SignalStrength.MODERATE
            elif signal_strength >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return TradingSignal(
                action=action,
                confidence=signal_strength,
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                reasoning=reasoning,
                market_regime=MarketRegime.SIDEWAYS,  # Se actualizará después
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error obteniendo señal base: {e}")
            return None
    
    def _adjust_signal_for_regime(self, signal: TradingSignal, regime: MarketRegime, 
                                 correlations: Dict[str, float]) -> TradingSignal:
        """Ajustar señal según régimen de mercado y correlaciones"""
        try:
            adjusted_signal = signal
            adjusted_signal.market_regime = regime
            
            # Ajustar según régimen
            if regime == MarketRegime.VOLATILE:
                # En mercados volátiles, reducir confianza y ajustar stops
                adjusted_signal.confidence *= 0.8
                adjusted_signal.reasoning.append("Confianza reducida por alta volatilidad")
                
                # Stops más amplios
                if signal.action == "BUY":
                    adjusted_signal.stop_loss *= 0.95  # Stop más amplio
                elif signal.action == "SELL":
                    adjusted_signal.stop_loss *= 1.05
                    
            elif regime == MarketRegime.LOW_VOLATILITY:
                # En baja volatilidad, targets más conservadores
                adjusted_signal.reasoning.append("Targets ajustados por baja volatilidad")
                
                if signal.action == "BUY":
                    adjusted_signal.take_profit *= 0.8  # Target más conservador
                elif signal.action == "SELL":
                    adjusted_signal.take_profit *= 1.2
            
            elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # En tendencias fuertes, aumentar confianza
                adjusted_signal.confidence = min(adjusted_signal.confidence * 1.2, 1.0)
                adjusted_signal.reasoning.append(f"Confianza aumentada por tendencia {regime.value}")
            
            # Ajustar según correlaciones
            strong_correlations = [pair for pair, corr in correlations.items() if abs(corr) > 0.7]
            if strong_correlations:
                adjusted_signal.reasoning.append(f"Correlaciones fuertes detectadas: {strong_correlations}")
                # Reducir ligeramente la confianza por riesgo de correlación
                adjusted_signal.confidence *= 0.95
            
            # Recalcular ratio riesgo/beneficio
            if adjusted_signal.action != "HOLD":
                risk = abs(adjusted_signal.entry_price - adjusted_signal.stop_loss)
                reward = abs(adjusted_signal.take_profit - adjusted_signal.entry_price)
                adjusted_signal.risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Error ajustando señal: {e}")
            return signal
    
    def _validate_and_enhance_signal(self, signal: TradingSignal, risk_metrics: Dict[str, float], 
                                   features_df: pd.DataFrame) -> TradingSignal:
        """Validar y mejorar la señal final"""
        try:
            enhanced_signal = signal
            
            # Validaciones de riesgo
            risk_score = risk_metrics.get('risk_score', 50)
            
            if risk_score > 80:  # Riesgo muy alto
                enhanced_signal.action = "HOLD"
                enhanced_signal.reasoning.append(f"Señal cancelada por riesgo alto: {risk_score:.1f}")
                return enhanced_signal
            
            elif risk_score > 60:  # Riesgo alto
                enhanced_signal.confidence *= 0.7
                enhanced_signal.reasoning.append(f"Confianza reducida por riesgo: {risk_score:.1f}")
            
            # Validar ratio riesgo/beneficio mínimo
            if enhanced_signal.risk_reward_ratio < 1.5 and enhanced_signal.action != "HOLD":
                enhanced_signal.action = "HOLD"
                enhanced_signal.reasoning.append(f"RR ratio insuficiente: {enhanced_signal.risk_reward_ratio:.2f}")
                return enhanced_signal
            
            # Validar confianza mínima
            if enhanced_signal.confidence < 0.3 and enhanced_signal.action != "HOLD":
                enhanced_signal.action = "HOLD"
                enhanced_signal.reasoning.append(f"Confianza insuficiente: {enhanced_signal.confidence:.2f}")
                return enhanced_signal
            
            # Añadir información adicional
            enhanced_signal.reasoning.append(f"Análisis completado - Confianza final: {enhanced_signal.confidence:.2f}")
            enhanced_signal.reasoning.append(f"Ratio R/R: {enhanced_signal.risk_reward_ratio:.2f}")
            enhanced_signal.reasoning.append(f"Régimen de mercado: {enhanced_signal.market_regime.value}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error validando señal: {e}")
            return signal
    
    def get_signal_summary(self, signal: TradingSignal) -> Dict[str, Any]:
        """Obtener resumen de la señal para logging/display"""
        return {
            'action': signal.action,
            'confidence': round(signal.confidence, 3),
            'strength': signal.strength.name,
            'entry_price': round(signal.entry_price, 5),
            'stop_loss': round(signal.stop_loss, 5),
            'take_profit': round(signal.take_profit, 5),
            'risk_reward_ratio': round(signal.risk_reward_ratio, 2),
            'market_regime': signal.market_regime.value,
            'reasoning_count': len(signal.reasoning),
            'timestamp': signal.timestamp.isoformat()
        }