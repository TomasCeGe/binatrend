#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/download.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 3:37:41 pm
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 1:47:50 pm
# Modified By: Tomás (tomas@cege.es>)
# -----
# Copyright (c) 2025 Nousmedis, CeGe
# 
# This product is licensed by CeGe.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import ccxt
import pandas as pd
from dotenv import load_dotenv

from .config import EXCHANGE_ID, MARKET_TYPE, SINCE_UTC
from .utils import ensure_dirs

import pandas as pd
from pathlib import Path

from .save import load_feather, save_feather

class EXCHANGE():
    logger: logging.Logger
    ex: ccxt.Exchange
    def __init__(self,logger:logging.Logger) -> None:
        self.logger=logger
        self.build_exchange()
        pass
    
    def build_exchange(self):
        load_dotenv()  # lee .env
        api_key = os.getenv("BINANCE_API_KEY") or ""
        api_secret = os.getenv("BINANCE_API_SECRET") or ""
        params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": MARKET_TYPE,  # 'spot'
            },
        }
        self.ex = getattr(ccxt, EXCHANGE_ID)(params)
        return self.ex

    def _sleep_on_rate_limit(self,e: Exception, base_sleep: float = 1.0):
        # backoff muy simple
        time.sleep(base_sleep)

    def fetch_ohlcv_all(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: Optional[int] = None,
        limit: int = 1000,
        step_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Descarga paginada completa desde 'since_ms' hasta 'until_ms' (o ahora).
        Devuelve DataFrame con columnas: date, open, high, low, close, volume
        """
        try:
            all_rows: List[List] = []
            t = since_ms
            
            until = until_ms or int(pd.Timestamp.utcnow().timestamp() * 1000)
            self.ex.logger.setLevel(logging.WARNING)  # reduce logs internos ccxt
            # si no nos dan step_ms, inferimos por timeframe
            if step_ms is None:
                tf_map = {
                    "1m": 60_000,
                    "3m": 180_000,
                    "5m": 300_000,
                    "15m": 900_000,
                    "30m": 1_800_000,
                    "1h": 3_600_000,
                    "2h": 7_200_000,
                    "4h": 14_400_000,
                    "6h": 21_600_000,
                    "12h": 43_200_000,
                    "1D": 86_400_000,
                    "1W": 604_800_000,
                    "1M": 2_592_000_000,  # aprox
                }
                step_ms = tf_map.get(timeframe, 300_000) * limit

            while t < until:
                try:
                    rows = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
                    #self.logger.debug(f"  -> fetched {len(rows)} rows from {datetime.fromtimestamp(t/1000, tz=timezone.utc)}")
                except ccxt.RateLimitExceeded as e:
                    self._sleep_on_rate_limit(e, 1.0)
                    self.logger.debug("Rate limit exceeded, sleeping...")
                    continue
                except ccxt.NetworkError as e:
                    self._sleep_on_rate_limit(e, 2.0)
                    self.logger.debug("Network error, sleeping...")
                    continue

                if not rows:
                    # si no hay velas nuevas, avanza por step
                    t += step_ms
                    continue

                all_rows.extend(rows)
                last_ts = rows[-1][0]
                # avanzamos 1 ms por encima para evitar duplicado
                t = last_ts + 1

                # cortes amables para no golpear la API
                time.sleep(0.05)

            if not all_rows:
                return pd.DataFrame(columns=["date","open","high","low","close","volume"])
            
            self.logger.debug(f"  -> descargadas {len(all_rows)} velas.")
            df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
            
            # a datetime (naive UTC)
            df["date"] =  pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            df = df[["date","open","high","low","close","volume"]].astype({
                "open": float, "high": float, "low": float, "close": float, "volume": float
            })
            
            # dedup y orden
            df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
            
            return df
        except Exception as e:
            self.logger.debug("Error en fetch_ohlcv_all:", str(e))
            return pd.DataFrame(columns=["date","open","high","low","close","volume"])
        
    def sync_df(self,fpath: str, symbol: str, meta: dict, timeframe: str, logger) -> tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:

        self.logger.debug(f"🔍 Comprobando histórico en {fpath}")
        df_old : Optional[pd.DataFrame] = load_feather(fpath) if os.path.isfile(fpath) else None
        if df_old is None:
            self.logger.debug(f"⚠️ No se encontró archivo en {fpath}, se creará uno nuevo.\n")
        else:
            self.logger.debug(f"📂 Cargado {len(df_old)} filas desde {fpath}" if df_old is not None else f"📂 No existe {fpath}, se creará uno nuevo.\n")
        
        # self.logger.debug("Columnas:", df_old.columns.tolist() if df_old is not None else "N/A ")
        # self.logger.debug("Ultimas 10 filas:\n", df_old.tail(10) if df_old is not None else "N/A ")
        
        self.logger.debug(f"Determinando punto de sincronización para {symbol} {timeframe}...\n")
        if df_old is None or len(df_old) == 0:
            self.logger.debug(f"⚠ No hay datos previos en {fpath} para {symbol} {timeframe}.")
            since_dt = pd.Timestamp("2024-06-01 00:00:00")
        elif df_old is not None and "date" in df_old.columns:
            last_saved = df_old["date"].iloc[-1]
            since_dt = pd.to_datetime(last_saved,utc=True)
            self.logger.debug(f"📎 Última vela guardada para {symbol} {timeframe}: {last_saved}")
        elif "last_date" in meta:
            since_dt = pd.to_datetime(meta["last_date"],utc=True)
            self.logger.debug(f"📎 Última fecha en meta para {symbol} {timeframe}: {since_dt}")
        else:
            self.logger.debug(f"⚠ No se pudo determinar 'since_dt', se usará fecha por defecto.")
            since_dt = pd.Timestamp("2024-00-01 00:00:00").tz_localize(None)
        return (df_old, since_dt)