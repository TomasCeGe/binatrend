#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/save.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 3:37:27 pm
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 11:35:49 am
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
from typing import Optional, List
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

def save_feather(df: pd.DataFrame, path: str, columns: Optional[List[str]] = None) -> None:
    """
    Guarda DataFrame en feather (compacto y rápido).
    Convierte 'date' a naive UTC para evitar problemas tz-aware vs naive.
    """
    #print(f"Guardando {len(df)} filas en {path}...")
    
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)

    if columns:
        df = df[columns]

    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path)

def load_feather(path: str) -> pd.DataFrame:
    """
    Carga DataFrame desde feather.
    Convierte 'date' a tz-aware UTC.
    """
    #print(f"Cargando datos desde {path}...")
    table = feather.read_table(path)
    df = table.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df
