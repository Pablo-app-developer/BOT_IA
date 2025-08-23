#!/usr/bin/env python3
"""
Módulo de Machine Learning para análisis predictivo y optimización de estrategias
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Optional, Tuple, Any
import talib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Predictor de Machine Learning para análisis de precios
    """
    
    def __init__(self, model_type: str = 'random_forest', 
                 lookback_period: int = 20, 
                 prediction_horizon: int = 5):
        """
        Inicializar el predictor ML
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'linear_regression')
            lookback_period: Número de períodos históricos para features
            prediction_horizon: Horizonte de predicción en períodos
        """
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.logger = logging.getLogger(f'{__name__}.{model_type}')
        
        # Inicializar modelo
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Inicializar el modelo de ML según el tipo especificado
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear características técnicas para el modelo
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con características técnicas
        """
        features_df = df.copy()
        
        # Características básicas de precio
        features_df['returns'] = df['close'].pct_change()
        features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['close_open_ratio'] = df['close'] / df['open']
        
        # Medias móviles
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            features_df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            features_df[f'price_sma_{period}_ratio'] = df['close'] / features_df[f'sma_{period}']
        
        # RSI
        features_df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features_df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_histogram'] = macd_hist
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        features_df['bb_upper'] = bb_upper
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features_df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        features_df['stoch_k'] = stoch_k
        features_df['stoch_d'] = stoch_d
        
        # Williams %R
        features_df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # CCI
        features_df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
        
        # Momentum
        features_df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
        
        # ROC
        features_df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        
        # Características de volatilidad
        features_df['volatility_5'] = features_df['returns'].rolling(5).std()
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        
        # Características de volumen (si está disponible)
        if 'tick_volume' in df.columns:
            features_df['volume_sma_10'] = talib.SMA(df['tick_volume'], timeperiod=10)
            features_df['volume_ratio'] = df['tick_volume'] / features_df['volume_sma_10']
            
            # OBV
            features_df['obv'] = talib.OBV(df['close'], df['tick_volume'])
            
            # Volume Price Trend
            features_df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['tick_volume']).cumsum()
        
        # Características de tiempo
        if 'time' in df.columns:
            features_df['hour'] = pd.to_datetime(df['time']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
            features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
            features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        # Lags de características importantes
        for lag in range(1, min(6, self.lookback_period + 1)):
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            features_df[f'rsi_lag_{lag}'] = features_df['rsi_14'].shift(lag)
            features_df[f'macd_lag_{lag}'] = features_df['macd'].shift(lag)
        
        # Limpiar datos
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        return features_df
    
    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preparar datos para entrenamiento
        
        Args:
            features_df: DataFrame con características
            
        Returns:
            Tupla con características (X) y target (y)
        """
        # Definir target: retorno futuro promedio
        target = features_df['returns'].shift(-self.prediction_horizon).rolling(
            window=self.prediction_horizon
        ).mean()
        
        # Seleccionar características numéricas
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
                       and features_df[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_cols
        
        # Crear matrices X e y
        X = features_df[feature_cols].values
        y = target.values
        
        # Eliminar filas con NaN
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def train(self, symbol: str, timeframe: int, days: int = 180) -> Optional[Dict]:
        """
        Entrenar el modelo con datos históricos
        
        Args:
            symbol: Símbolo a analizar
            timeframe: Marco temporal
            days: Días de datos históricos
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        try:
            # Obtener datos históricos
            if not mt5.initialize():
                self.logger.error("Error al inicializar MT5")
                return None
            
            # Calcular fechas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Obtener datos
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) < self.lookback_period * 2:
                self.logger.error(f"Datos insuficientes para {symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Crear características
            features_df = self._create_features(df)
            
            # Preparar datos de entrenamiento
            X, y = self._prepare_training_data(features_df)
            
            if len(X) < 100:  # Mínimo de datos para entrenamiento
                self.logger.error("Datos insuficientes para entrenamiento")
                return None
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # Calcular métricas
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            self.logger.info(f"Modelo entrenado - R² Test: {test_r2:.4f}, MAE Test: {test_mae:.6f}")
            
            return {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'n_features': len(self.feature_columns),
                'n_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {e}")
            return None
    
    def predict(self, symbol: str, timeframe: int) -> Optional[Dict]:
        """
        Realizar predicción de precio
        
        Args:
            symbol: Símbolo a analizar
            timeframe: Marco temporal
            
        Returns:
            Diccionario con predicción
        """
        try:
            if self.model is None:
                self.logger.error("Modelo no entrenado")
                return None
            
            # Obtener datos recientes
            if not mt5.initialize():
                self.logger.error("Error al inicializar MT5")
                return None
            
            # Obtener datos suficientes para características
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Más datos para características
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) < self.lookback_period:
                self.logger.error(f"Datos insuficientes para predicción de {symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Crear características
            features_df = self._create_features(df)
            
            # Obtener características más recientes
            if self.feature_columns is None:
                self.logger.error("Columnas de características no definidas")
                return None
            
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            
            # Verificar NaN
            if np.isnan(latest_features).any():
                self.logger.warning("Características contienen NaN, rellenando con valores anteriores")
                latest_features = np.nan_to_num(latest_features, nan=0.0)
            
            # Escalar características
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Realizar predicción
            prediction = self.model.predict(latest_features_scaled)[0]
            
            # Obtener precio actual
            current_price = df['close'].iloc[-1]
            
            # Calcular precio predicho
            predicted_price = current_price * (1 + prediction)
            
            # Determinar dirección
            direction = 'BUY' if prediction > 0 else 'SELL'
            
            # Calcular fuerza de la señal
            strength = min(abs(prediction) * 1000, 100)  # Normalizar a 0-100
            
            # Calcular confianza basada en la magnitud de la predicción
            confidence = min(abs(prediction) * 2000, 100)  # Normalizar a 0-100
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': prediction,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return None
    
    def save_model(self, filepath: str) -> bool:
        """
        Guardar modelo entrenado
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            True si se guardó correctamente
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
                'lookback_period': self.lookback_period,
                'prediction_horizon': self.prediction_horizon
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            
            self.logger.info(f"Modelo guardado en {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar modelo: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Cargar modelo entrenado
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            True si se cargó correctamente
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Archivo no encontrado: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.lookback_period = model_data['lookback_period']
            self.prediction_horizon = model_data['prediction_horizon']
            
            self.logger.info(f"Modelo cargado desde {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {e}")
            return False

class StrategyOptimizer:
    """
    Optimizador de parámetros de estrategias
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.StrategyOptimizer')
    
    def optimize_parameters(self, strategy_class, symbol: str, timeframe: int, 
                          days: int, param_ranges: Dict, 
                          metric: str = 'sharpe_ratio') -> Optional[Dict]:
        """
        Optimizar parámetros de estrategia usando Grid Search
        
        Args:
            strategy_class: Clase de la estrategia
            symbol: Símbolo a optimizar
            timeframe: Marco temporal
            days: Días de datos históricos
            param_ranges: Rangos de parámetros a probar
            metric: Métrica a optimizar
            
        Returns:
            Diccionario con mejores parámetros y métricas
        """
        try:
            from itertools import product
            
            # Generar todas las combinaciones de parámetros
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            param_combinations = list(product(*param_values))
            
            best_score = float('-inf')
            best_params = None
            best_metrics = None
            
            self.logger.info(f"Optimizando {len(param_combinations)} combinaciones de parámetros")
            
            for i, param_combo in enumerate(param_combinations):
                try:
                    # Crear diccionario de parámetros
                    params = dict(zip(param_names, param_combo))
                    
                    # Crear instancia de estrategia con parámetros
                    strategy = strategy_class(**params)
                    
                    # Simular estrategia (esto requeriría un backtester)
                    # Por ahora, simulamos métricas básicas
                    metrics = self._simulate_strategy(strategy, symbol, timeframe, days)
                    
                    if metrics and metric in metrics:
                        score = metrics[metric]
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            best_metrics = metrics.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progreso: {i + 1}/{len(param_combinations)}")
                        
                except Exception as e:
                    self.logger.warning(f"Error en combinación {param_combo}: {e}")
                    continue
            
            if best_params is not None:
                self.logger.info(f"Mejores parámetros encontrados: {best_params}")
                self.logger.info(f"Mejor {metric}: {best_score:.4f}")
                
                return {
                    'best_params': best_params,
                    'best_score': best_score,
                    'best_metrics': best_metrics,
                    'total_combinations': len(param_combinations)
                }
            else:
                self.logger.warning("No se encontraron parámetros válidos")
                return None
                
        except Exception as e:
            self.logger.error(f"Error en optimización: {e}")
            return None
    
    def _simulate_strategy(self, strategy, symbol: str, timeframe: int, days: int) -> Optional[Dict]:
        """
        Simular estrategia y calcular métricas
        
        Args:
            strategy: Instancia de estrategia
            symbol: Símbolo
            timeframe: Marco temporal
            days: Días de simulación
            
        Returns:
            Diccionario con métricas
        """
        try:
            # Obtener datos históricos
            if not mt5.initialize():
                return None
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) < 100:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Simular trades (simplificado)
            returns = []
            for i in range(50, len(df) - 1):
                # Obtener señal de estrategia (esto dependería de la implementación)
                # Por ahora simulamos retornos aleatorios
                daily_return = np.random.normal(0.0001, 0.01)  # Retorno diario simulado
                returns.append(daily_return)
            
            if not returns:
                return None
            
            returns = np.array(returns)
            
            # Calcular métricas
            total_return = np.prod(1 + returns) - 1
            volatility = np.std(returns) * np.sqrt(252)  # Anualizada
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error en simulación: {e}")
            return None
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calcular máximo drawdown
        
        Args:
            returns: Array de retornos
            
        Returns:
            Máximo drawdown
        """
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
        except:
            return 0.0