#!/usr/bin/env python3
"""
Módulo de Análisis de Patrones de Velas Japonesas con IA
Utiliza TA-Lib para reconocimiento de patrones y algoritmos de IA para scoring
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime

class PatternType(Enum):
    """Tipos de patrones de velas"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"

class PatternStrength(Enum):
    """Fuerza del patrón detectado"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class CandlestickPattern:
    """Clase para representar un patrón de velas detectado"""
    name: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    signal_value: int  # Valor original de TA-Lib (-100, 0, 100)
    context_score: float  # Score basado en contexto de mercado
    final_score: float  # Score final combinado
    timestamp: datetime
    description: str
    reliability: float  # Confiabilidad histórica del patrón

class CandlestickPatternsAI:
    """
    Analizador de patrones de velas japonesas con IA
    
    Utiliza TA-Lib para detectar patrones y aplica algoritmos de IA
    para evaluar la confiabilidad y fuerza de cada patrón basándose
    en el contexto del mercado.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger('CandlestickPatternsAI')
        self.config = config or {}
        
        # Configuración de parámetros
        self.min_confidence = self.config.get('min_pattern_confidence', 0.6)
        self.context_weight = self.config.get('context_weight', 0.4)
        self.pattern_weight = self.config.get('pattern_weight', 0.6)
        
        # Definir patrones soportados con sus características
        self.pattern_functions = {
            # Patrones de reversión alcista
            'CDLDOJI': {'type': PatternType.NEUTRAL, 'reliability': 0.65, 'description': 'Doji - Indecisión del mercado'},
            'CDLHAMMER': {'type': PatternType.BULLISH, 'reliability': 0.72, 'description': 'Hammer - Reversión alcista'},
            'CDLINVERTEDHAMMER': {'type': PatternType.BULLISH, 'reliability': 0.68, 'description': 'Inverted Hammer - Posible reversión alcista'},
            'CDLMORNINGSTAR': {'type': PatternType.BULLISH, 'reliability': 0.78, 'description': 'Morning Star - Fuerte reversión alcista'},
            'CDLMORNINGDOJISTAR': {'type': PatternType.BULLISH, 'reliability': 0.75, 'description': 'Morning Doji Star - Reversión alcista'},
            'CDLPIERCING': {'type': PatternType.BULLISH, 'reliability': 0.70, 'description': 'Piercing Pattern - Reversión alcista'},
            'CDLBELTHOLD': {'type': PatternType.BULLISH, 'reliability': 0.65, 'description': 'Belt Hold - Continuación o reversión'},
            'CDLENGULFING': {'type': PatternType.BULLISH, 'reliability': 0.73, 'description': 'Bullish Engulfing - Fuerte reversión alcista'},
            
            # Patrones de reversión bajista
            'CDLHANGINGMAN': {'type': PatternType.BEARISH, 'reliability': 0.69, 'description': 'Hanging Man - Reversión bajista'},
            'CDLSHOOTINGSTAR': {'type': PatternType.BEARISH, 'reliability': 0.71, 'description': 'Shooting Star - Reversión bajista'},
            'CDLEVENINGSTAR': {'type': PatternType.BEARISH, 'reliability': 0.76, 'description': 'Evening Star - Fuerte reversión bajista'},
            'CDLEVENINGDOJISTAR': {'type': PatternType.BEARISH, 'reliability': 0.74, 'description': 'Evening Doji Star - Reversión bajista'},
            'CDLDARKCLOUDCOVER': {'type': PatternType.BEARISH, 'reliability': 0.72, 'description': 'Dark Cloud Cover - Reversión bajista'},
            
            # Patrones de continuación
            'CDLTHRUSTING': {'type': PatternType.CONTINUATION, 'reliability': 0.63, 'description': 'Thrusting Pattern - Continuación bajista'},
            'CDLSEPARATINGLINES': {'type': PatternType.CONTINUATION, 'reliability': 0.60, 'description': 'Separating Lines - Continuación de tendencia'},
            
            # Patrones de tres velas
            'CDL3BLACKCROWS': {'type': PatternType.BEARISH, 'reliability': 0.80, 'description': 'Three Black Crows - Fuerte reversión bajista'},
            'CDL3WHITESOLDIERS': {'type': PatternType.BULLISH, 'reliability': 0.79, 'description': 'Three White Soldiers - Fuerte reversión alcista'},
            'CDL3INSIDE': {'type': PatternType.REVERSAL, 'reliability': 0.67, 'description': 'Three Inside Up/Down - Reversión'},
            'CDL3OUTSIDE': {'type': PatternType.REVERSAL, 'reliability': 0.70, 'description': 'Three Outside Up/Down - Reversión'},
            
            # Patrones avanzados
            'CDLHARAMI': {'type': PatternType.REVERSAL, 'reliability': 0.66, 'description': 'Harami - Posible reversión'},
            'CDLHARAMICROSS': {'type': PatternType.REVERSAL, 'reliability': 0.68, 'description': 'Harami Cross - Reversión con doji'},
            'CDLMARUBOZU': {'type': PatternType.CONTINUATION, 'reliability': 0.64, 'description': 'Marubozu - Fuerte continuación'},
            'CDLSPINNINGTOP': {'type': PatternType.NEUTRAL, 'reliability': 0.58, 'description': 'Spinning Top - Indecisión'},
        }
        
        self.logger.info(f"Analizador de patrones de velas inicializado con {len(self.pattern_functions)} patrones")
    
    def analyze_patterns(self, data: pd.DataFrame, symbol: str = None) -> List[CandlestickPattern]:
        """
        Analiza patrones de velas en los datos proporcionados
        
        Args:
            data: DataFrame con columnas OHLCV
            symbol: Símbolo del instrumento (opcional)
            
        Returns:
            Lista de patrones detectados
        """
        if len(data) < 10:
            self.logger.warning("Datos insuficientes para análisis de patrones")
            return []
        
        try:
            # Preparar datos
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            detected_patterns = []
            
            # Detectar cada patrón
            for pattern_name, pattern_info in self.pattern_functions.items():
                try:
                    # Obtener función de TA-Lib
                    pattern_func = getattr(talib, pattern_name)
                    
                    # Calcular patrón
                    pattern_values = pattern_func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Buscar señales en las últimas velas
                    for i in range(max(0, len(pattern_values) - 5), len(pattern_values)):
                        if pattern_values[i] != 0:
                            # Crear objeto de patrón
                            pattern = self._create_pattern_object(
                                pattern_name, pattern_info, pattern_values[i], 
                                data.iloc[i], i, len(data)
                            )
                            
                            # Calcular score contextual
                            context_score = self._calculate_context_score(data, i, pattern)
                            pattern.context_score = context_score
                            
                            # Calcular score final
                            pattern.final_score = self._calculate_final_score(pattern)
                            
                            # Filtrar por confianza mínima
                            if pattern.confidence >= self.min_confidence:
                                detected_patterns.append(pattern)
                                
                except Exception as e:
                    self.logger.warning(f"Error procesando patrón {pattern_name}: {e}")
                    continue
            
            # Ordenar por score final
            detected_patterns.sort(key=lambda x: x.final_score, reverse=True)
            
            self.logger.info(f"Detectados {len(detected_patterns)} patrones válidos")
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error en análisis de patrones: {e}")
            return []
    
    def _create_pattern_object(self, pattern_name: str, pattern_info: Dict, 
                              signal_value: int, candle_data: pd.Series, 
                              index: int, total_candles: int) -> CandlestickPattern:
        """
        Crea objeto de patrón con información básica
        """
        # Calcular fuerza basada en valor de señal
        strength = self._calculate_pattern_strength(abs(signal_value))
        
        # Calcular confianza inicial
        base_confidence = pattern_info['reliability']
        signal_confidence = abs(signal_value) / 100.0
        confidence = (base_confidence + signal_confidence) / 2.0
        
        return CandlestickPattern(
            name=pattern_name.replace('CDL', ''),
            pattern_type=pattern_info['type'],
            strength=strength,
            confidence=confidence,
            signal_value=signal_value,
            context_score=0.0,  # Se calculará después
            final_score=0.0,    # Se calculará después
            timestamp=datetime.now(),
            description=pattern_info['description'],
            reliability=pattern_info['reliability']
        )
    
    def _calculate_pattern_strength(self, signal_value: int) -> PatternStrength:
        """
        Calcula la fuerza del patrón basada en el valor de señal
        """
        if signal_value >= 100:
            return PatternStrength.VERY_STRONG
        elif signal_value >= 80:
            return PatternStrength.STRONG
        elif signal_value >= 60:
            return PatternStrength.MODERATE
        elif signal_value >= 40:
            return PatternStrength.WEAK
        else:
            return PatternStrength.VERY_WEAK
    
    def _calculate_context_score(self, data: pd.DataFrame, index: int, 
                                pattern: CandlestickPattern) -> float:
        """
        Calcula score contextual basado en condiciones de mercado
        """
        try:
            context_score = 0.5  # Score base
            
            # Analizar volumen (si está disponible)
            if 'volume' in data.columns:
                recent_volume = data['volume'].iloc[max(0, index-5):index+1].mean()
                avg_volume = data['volume'].mean()
                if recent_volume > avg_volume * 1.2:
                    context_score += 0.1  # Volumen alto es positivo
            
            # Analizar tendencia
            if index >= 20:
                sma_20 = data['close'].iloc[index-19:index+1].mean()
                current_price = data['close'].iloc[index]
                
                if pattern.pattern_type == PatternType.BULLISH and current_price > sma_20:
                    context_score += 0.15  # Patrón alcista en tendencia alcista
                elif pattern.pattern_type == PatternType.BEARISH and current_price < sma_20:
                    context_score += 0.15  # Patrón bajista en tendencia bajista
            
            # Analizar volatilidad
            if index >= 10:
                recent_volatility = data['close'].iloc[index-9:index+1].std()
                avg_volatility = data['close'].std()
                
                if recent_volatility > avg_volatility * 1.5:
                    context_score -= 0.05  # Alta volatilidad reduce confianza
                elif recent_volatility < avg_volatility * 0.7:
                    context_score += 0.05  # Baja volatilidad aumenta confianza
            
            # Analizar posición en el rango
            if index >= 50:
                high_50 = data['high'].iloc[index-49:index+1].max()
                low_50 = data['low'].iloc[index-49:index+1].min()
                current_price = data['close'].iloc[index]
                
                position_in_range = (current_price - low_50) / (high_50 - low_50)
                
                if pattern.pattern_type == PatternType.BULLISH and position_in_range < 0.3:
                    context_score += 0.1  # Patrón alcista cerca del mínimo
                elif pattern.pattern_type == PatternType.BEARISH and position_in_range > 0.7:
                    context_score += 0.1  # Patrón bajista cerca del máximo
            
            return max(0.0, min(1.0, context_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculando score contextual: {e}")
            return 0.5
    
    def _calculate_final_score(self, pattern: CandlestickPattern) -> float:
        """
        Calcula el score final combinando confianza del patrón y contexto
        """
        pattern_score = pattern.confidence * self.pattern_weight
        context_score = pattern.context_score * self.context_weight
        
        final_score = pattern_score + context_score
        
        # Ajustar por fuerza del patrón
        strength_multiplier = {
            PatternStrength.VERY_WEAK: 0.8,
            PatternStrength.WEAK: 0.9,
            PatternStrength.MODERATE: 1.0,
            PatternStrength.STRONG: 1.1,
            PatternStrength.VERY_STRONG: 1.2
        }
        
        final_score *= strength_multiplier[pattern.strength]
        
        return max(0.0, min(1.0, final_score))
    
    def get_trading_signal(self, patterns: List[CandlestickPattern]) -> Dict[str, Any]:
        """
        Genera señal de trading basada en los patrones detectados
        
        Args:
            patterns: Lista de patrones detectados
            
        Returns:
            Diccionario con señal de trading
        """
        if not patterns:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'patterns_count': 0,
                'reasoning': ['No se detectaron patrones válidos']
            }
        
        # Filtrar patrones por score mínimo
        valid_patterns = [p for p in patterns if p.final_score >= 0.6]
        
        if not valid_patterns:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'patterns_count': len(patterns),
                'reasoning': ['Patrones detectados no alcanzan score mínimo']
            }
        
        # Analizar consenso de patrones
        bullish_score = sum(p.final_score for p in valid_patterns 
                           if p.pattern_type == PatternType.BULLISH)
        bearish_score = sum(p.final_score for p in valid_patterns 
                           if p.pattern_type == PatternType.BEARISH)
        
        bullish_count = len([p for p in valid_patterns if p.pattern_type == PatternType.BULLISH])
        bearish_count = len([p for p in valid_patterns if p.pattern_type == PatternType.BEARISH])
        
        # Determinar señal
        if bullish_score > bearish_score * 1.2 and bullish_count > 0:
            signal = 'BUY'
            confidence = min(0.95, bullish_score / len(valid_patterns))
        elif bearish_score > bullish_score * 1.2 and bearish_count > 0:
            signal = 'SELL'
            confidence = min(0.95, bearish_score / len(valid_patterns))
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # Generar reasoning
        reasoning = []
        for pattern in valid_patterns[:3]:  # Top 3 patrones
            reasoning.append(f"{pattern.description} (Score: {pattern.final_score:.2f})")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'patterns_count': len(valid_patterns),
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'top_patterns': [p.name for p in valid_patterns[:3]],
            'reasoning': reasoning
        }
    
    def get_pattern_summary(self, patterns: List[CandlestickPattern]) -> Dict[str, Any]:
        """
        Genera resumen de patrones detectados
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'neutral_patterns': 0,
                'by_type': {},
                'by_strength': {}
            }
        
        by_type = {}
        by_strength = {}
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for pattern in patterns:
            # Contar por tipo
            type_key = pattern.pattern_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Contar patrones específicos
            if pattern.pattern_type == PatternType.BULLISH:
                bullish_count += 1
            elif pattern.pattern_type == PatternType.BEARISH:
                bearish_count += 1
            else:
                neutral_count += 1
            
            # Contar por fuerza
            strength_key = pattern.strength.name
            by_strength[strength_key] = by_strength.get(strength_key, 0) + 1
        
        return {
            'total_patterns': len(patterns),
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'neutral_patterns': neutral_count,
            'by_type': by_type,
            'by_strength': by_strength,
            'avg_confidence': np.mean([p.confidence for p in patterns]),
            'avg_final_score': np.mean([p.final_score for p in patterns]),
            'top_pattern': patterns[0].name if patterns else None
        }