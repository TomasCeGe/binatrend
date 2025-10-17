#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/utils.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 3:37:00 pm
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 9:49:20 am
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
import os
import json
from typing import Optional
import pandas as pd

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def ensure_file(path: str) -> None:
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("") # crea archivo vacío

def timeframe_to_pandas_rule(tf: str) -> str:
    """Mapea ccxt timeframe -> regla pandas resample."""
    mapping = {
        "1m": "1T",
        "3m": "3T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1D": "1D",
        "3D": "3D",
        "1W": "1W",
        "1M": "1ME",
    }
    return mapping.get(tf, tf)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV con cierre a la derecha, label al final de vela.
    df: columnas ['date','open','high','low','close','volume'] con 'date' index
    """
    if "date" in df.columns:
        df = df.set_index("date")
    ohlc = df[["open","high","low","close"]].resample(rule, label="right", closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    )
    vol = df["volume"].resample(rule, label="right", closed="right").sum()
    out = ohlc.join(vol)
    out = out.dropna(how="any").reset_index()
    # aseguramos 'date' naive UTC para evitar tz issues
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out

def load_meta(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def feather_path(base_dir: str, pair: str, tf: str) -> str:
    safe_pair = pair.replace("/", "_")
    pair_dir = os.path.join(base_dir, "USDC")  # carpeta por quote si quieres segmentar
    os.makedirs(pair_dir, exist_ok=True)
    return os.path.join(pair_dir, f"{safe_pair}_{tf}.feather")

def meta_path(meta_dir: str, pair: str, tf: str) -> str:
    safe_pair = pair.replace("/", "_")
    os.makedirs(meta_dir, exist_ok=True)
    return os.path.join(meta_dir, f"{safe_pair}_{tf}.json")
