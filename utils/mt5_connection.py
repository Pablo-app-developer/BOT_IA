import MetaTrader5 as mt5
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

class MT5Connection:
    """
    Clase para manejar la conexión con MetaTrader 5
    """
    
    def __init__(self, config_path: str):
        """
        Inicializa la conexión MT5
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        self.config = self._load_config(config_path)
        self.is_connected = False
        self.account_info = None
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde archivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error cargando configuración: {e}")
    
    def _setup_logging(self):
        """Configura el sistema de logging"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'mt5.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establece conexión con MetaTrader 5
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        try:
            # Inicializar MT5
            if not mt5.initialize():
                self.logger.error(f"Error inicializando MT5: {mt5.last_error()}")
                return False
            
            # Obtener configuración de cuenta
            account_config = self.config.get('account', {})
            
            # Intentar login si se proporcionan credenciales
            if 'login' in account_config and 'password' in account_config:
                login = account_config['login']
                password = account_config['password']
                server = account_config.get('server', '')
                
                if not mt5.login(login, password, server):
                    self.logger.error(f"Error en login MT5: {mt5.last_error()}")
                    return False
            
            # Verificar conexión
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.logger.error("No se pudo obtener información de la cuenta")
                return False
            
            self.is_connected = True
            self.logger.info(f"Conectado a MT5 - Cuenta: {self.account_info.login}")
            self.logger.info(f"Balance: {self.account_info.balance} {self.account_info.currency}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error conectando a MT5: {e}")
            return False
    
    def disconnect(self):
        """Desconecta de MetaTrader 5"""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            self.logger.info("Desconectado de MT5")
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Obtiene información de la cuenta"""
        if not self.is_connected:
            return None
        
        info = mt5.account_info()
        if info is None:
            return None
        
        return info._asdict()
    
    def get_symbols(self) -> List[str]:
        """Obtiene lista de símbolos disponibles"""
        if not self.is_connected:
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        return [symbol.name for symbol in symbols]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un símbolo específico"""
        if not self.is_connected:
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return info._asdict()
    
    def get_rates(self, symbol: str, timeframe: int, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de precios históricos
        
        Args:
            symbol: Símbolo del instrumento
            timeframe: Marco temporal (mt5.TIMEFRAME_*)
            count: Número de barras a obtener
        
        Returns:
            DataFrame con datos OHLCV
        """
        if not self.is_connected:
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                self.logger.error(f"Error obteniendo datos para {symbol}: {mt5.last_error()}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error procesando datos de {symbol}: {e}")
            return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Obtiene posiciones abiertas"""
        if not self.is_connected:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        return [pos._asdict() for pos in positions]
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Obtiene órdenes pendientes"""
        if not self.is_connected:
            return []
        
        orders = mt5.orders_get()
        if orders is None:
            return []
        
        return [order._asdict() for order in orders]
    
    def send_order(self, symbol: str, order_type: int, volume: float, 
                   price: float = 0.0, sl: float = 0.0, tp: float = 0.0, 
                   comment: str = "") -> Dict[str, Any]:
        """
        Envía una orden al mercado
        
        Args:
            symbol: Símbolo del instrumento
            order_type: Tipo de orden (mt5.ORDER_TYPE_*)
            volume: Volumen de la orden
            price: Precio (para órdenes pendientes)
            sl: Stop Loss
            tp: Take Profit
            comment: Comentario
        
        Returns:
            Diccionario con resultado de la orden
        """
        if not self.is_connected:
            return {'success': False, 'error': 'No conectado a MT5'}
        
        try:
            # Preparar request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Agregar precio si es necesario
            if price > 0:
                request["price"] = price
            
            # Agregar SL y TP si se especifican
            if sl > 0:
                request["sl"] = sl
            if tp > 0:
                request["tp"] = tp
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Error enviando orden: {result.comment}")
                return {'success': False, 'error': result.comment, 'retcode': result.retcode}
            
            self.logger.info(f"Orden enviada exitosamente: {result.order}")
            return {
                'success': True, 
                'order': result.order,
                'deal': result.deal,
                'volume': result.volume,
                'price': result.price
            }
            
        except Exception as e:
            self.logger.error(f"Error enviando orden: {e}")
            return {'success': False, 'error': str(e)}
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """
        Cierra una posición específica
        
        Args:
            ticket: Ticket de la posición
        
        Returns:
            Diccionario con resultado del cierre
        """
        if not self.is_connected:
            return {'success': False, 'error': 'No conectado a MT5'}
        
        try:
            # Obtener información de la posición
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Posición no encontrada'}
            
            position = position[0]
            
            # Determinar tipo de orden de cierre
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY
            
            # Preparar request de cierre
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "comment": "Cierre automático",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden de cierre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Error cerrando posición: {result.comment}")
                return {'success': False, 'error': result.comment, 'retcode': result.retcode}
            
            self.logger.info(f"Posición cerrada exitosamente: {ticket}")
            return {
                'success': True,
                'deal': result.deal,
                'volume': result.volume,
                'price': result.price
            }
            
        except Exception as e:
            self.logger.error(f"Error cerrando posición: {e}")
            return {'success': False, 'error': str(e)}
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()