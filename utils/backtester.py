#!/usr/bin/env python3
"""
Módulo de Backtesting para el Bot de Trading MT5
Permite probar estrategias con datos históricos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import MetaTrader5 as mt5
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestResult:
    """Clase para almacenar resultados del backtesting"""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 10000
        self.final_balance = 10000
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.start_date = None
        self.end_date = None
        
    def calculate_metrics(self):
        """Calcula métricas de rendimiento"""
        if not self.trades:
            return
            
        profits = [trade['profit'] for trade in self.trades]
        
        self.total_trades = len(self.trades)
        self.winning_trades = len([p for p in profits if p > 0])
        self.losing_trades = len([p for p in profits if p < 0])
        self.total_profit = sum([p for p in profits if p > 0])
        self.total_loss = abs(sum([p for p in profits if p < 0]))
        
        # Win rate
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        # Profit factor
        self.profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        # Calcular drawdown máximo
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for value in self.equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_dd:
                    max_dd = drawdown
            self.max_drawdown = max_dd
        
        # Sharpe ratio (simplificado)
        if profits:
            returns = np.array(profits) / self.initial_balance
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte los resultados a diccionario"""
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return': ((self.final_balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'total_profit': round(self.total_profit, 2),
            'total_loss': round(self.total_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else None,
            'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else None
        }

class Backtester:
    """Motor de backtesting para estrategias de trading"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = [initial_balance]
        self.logger = logging.getLogger('Backtester')
        
    def get_historical_data(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Obtiene datos históricos de MT5"""
        try:
            if not mt5.initialize():
                self.logger.error("Error inicializando MT5 para backtesting")
                return pd.DataFrame()
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No se encontraron datos para {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos históricos: {e}")
            return pd.DataFrame()
        finally:
            mt5.shutdown()
    
    def execute_trade(self, signal: Dict[str, Any], current_price: float, timestamp: datetime) -> bool:
        """Ejecuta una operación en el backtesting"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            volume = signal.get('volume', 0.1)
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            if action == 'BUY':
                # Abrir posición de compra
                position = {
                    'symbol': symbol,
                    'type': 'BUY',
                    'volume': volume,
                    'open_price': current_price,
                    'open_time': timestamp,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                position_id = f"{symbol}_{timestamp.timestamp()}"
                self.positions[position_id] = position
                
                self.logger.info(f"Posición BUY abierta: {symbol} @ {current_price}")
                return True
                
            elif action == 'SELL':
                # Abrir posición de venta
                position = {
                    'symbol': symbol,
                    'type': 'SELL',
                    'volume': volume,
                    'open_price': current_price,
                    'open_time': timestamp,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                position_id = f"{symbol}_{timestamp.timestamp()}"
                self.positions[position_id] = position
                
                self.logger.info(f"Posición SELL abierta: {symbol} @ {current_price}")
                return True
                
            elif action == 'CLOSE':
                # Cerrar posiciones del símbolo
                closed_positions = []
                for pos_id, position in self.positions.items():
                    if position['symbol'] == symbol:
                        self._close_position(pos_id, current_price, timestamp)
                        closed_positions.append(pos_id)
                
                for pos_id in closed_positions:
                    del self.positions[pos_id]
                
                return len(closed_positions) > 0
                
        except Exception as e:
            self.logger.error(f"Error ejecutando trade en backtesting: {e}")
            return False
    
    def _close_position(self, position_id: str, close_price: float, close_time: datetime):
        """Cierra una posición y calcula el profit/loss"""
        position = self.positions[position_id]
        
        if position['type'] == 'BUY':
            profit = (close_price - position['open_price']) * position['volume'] * 100000  # Asumiendo forex
        else:  # SELL
            profit = (position['open_price'] - close_price) * position['volume'] * 100000
        
        # Registrar el trade
        trade = {
            'symbol': position['symbol'],
            'type': position['type'],
            'volume': position['volume'],
            'open_price': position['open_price'],
            'close_price': close_price,
            'open_time': position['open_time'],
            'close_time': close_time,
            'profit': profit,
            'duration': (close_time - position['open_time']).total_seconds() / 3600  # horas
        }
        
        self.trades.append(trade)
        self.current_balance += profit
        self.equity_curve.append(self.current_balance)
        
        self.logger.info(f"Posición cerrada: {position['symbol']} {position['type']} - Profit: {profit:.2f}")
    
    def check_stop_loss_take_profit(self, current_data: Dict[str, float], timestamp: datetime):
        """Verifica stop loss y take profit"""
        closed_positions = []
        
        for pos_id, position in self.positions.items():
            symbol = position['symbol']
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]
            
            should_close = False
            
            if position['type'] == 'BUY':
                # Check stop loss
                if position['stop_loss'] and current_price <= position['stop_loss']:
                    should_close = True
                    current_price = position['stop_loss']
                # Check take profit
                elif position['take_profit'] and current_price >= position['take_profit']:
                    should_close = True
                    current_price = position['take_profit']
            
            else:  # SELL
                # Check stop loss
                if position['stop_loss'] and current_price >= position['stop_loss']:
                    should_close = True
                    current_price = position['stop_loss']
                # Check take profit
                elif position['take_profit'] and current_price <= position['take_profit']:
                    should_close = True
                    current_price = position['take_profit']
            
            if should_close:
                self._close_position(pos_id, current_price, timestamp)
                closed_positions.append(pos_id)
        
        # Eliminar posiciones cerradas
        for pos_id in closed_positions:
            del self.positions[pos_id]
    
    def run_backtest(self, strategy, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Ejecuta el backtesting de una estrategia"""
        self.logger.info(f"Iniciando backtesting: {symbol} desde {start_date} hasta {end_date}")
        
        # Obtener datos históricos
        data = self.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            self.logger.error("No se pudieron obtener datos históricos")
            return BacktestResult()
        
        # Resetear estado
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        # Procesar cada barra de datos
        for i in range(len(data)):
            current_bar = data.iloc[i]
            timestamp = data.index[i]
            
            # Verificar stop loss y take profit
            current_prices = {symbol: current_bar['close']}
            self.check_stop_loss_take_profit(current_prices, timestamp)
            
            # Obtener datos suficientes para el análisis
            if i < 50:  # Necesitamos al menos 50 barras para indicadores
                continue
            
            analysis_data = data.iloc[:i+1]
            
            # Analizar con la estrategia
            try:
                signal = strategy.get_signal(symbol, mt5.TIMEFRAME_H1, analysis_data)
                
                if signal and signal['action'] != 'HOLD':
                    self.execute_trade(signal, current_bar['close'], timestamp)
                    
            except Exception as e:
                self.logger.error(f"Error en análisis de estrategia: {e}")
                continue
        
        # Cerrar todas las posiciones abiertas al final
        final_price = data.iloc[-1]['close']
        final_time = data.index[-1]
        
        for pos_id in list(self.positions.keys()):
            self._close_position(pos_id, final_price, final_time)
        
        self.positions = {}
        
        # Crear resultado
        result = BacktestResult()
        result.trades = self.trades.copy()
        result.equity_curve = self.equity_curve.copy()
        result.initial_balance = self.initial_balance
        result.final_balance = self.current_balance
        result.start_date = start_date
        result.end_date = end_date
        result.calculate_metrics()
        
        self.logger.info(f"Backtesting completado: {len(self.trades)} trades, Balance final: {self.current_balance:.2f}")
        
        return result
    
    def save_results(self, result: BacktestResult, filename: str):
        """Guarda los resultados del backtesting"""
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        # Guardar métricas
        metrics_file = results_dir / f"{filename}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Guardar trades
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_file = results_dir / f"{filename}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
        
        # Guardar curva de equity
        if result.equity_curve:
            equity_df = pd.DataFrame({'balance': result.equity_curve})
            equity_file = results_dir / f"{filename}_equity.csv"
            equity_df.to_csv(equity_file, index=False)
        
        self.logger.info(f"Resultados guardados en {results_dir}")
    
    def plot_results(self, result: BacktestResult, filename: str = None):
        """Genera gráficos de los resultados"""
        if not result.equity_curve:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Curva de equity
        ax1.plot(result.equity_curve, linewidth=2)
        ax1.set_title('Curva de Equity')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True, alpha=0.3)
        
        # Distribución de profits
        if result.trades:
            profits = [trade['profit'] for trade in result.trades]
            ax2.hist(profits, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Distribución de Profits/Losses')
            ax2.set_xlabel('Profit/Loss ($)')
            ax2.set_ylabel('Frecuencia')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            results_dir = Path('backtest_results')
            results_dir.mkdir(exist_ok=True)
            plt.savefig(results_dir / f"{filename}_chart.png", dpi=300, bbox_inches='tight')
        
        plt.show()