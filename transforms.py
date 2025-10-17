#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/transforms.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 3:37:17 pm
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 7:26:25 pm
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
import pandas as pd
import numpy as np

SR_CONFIG = {
    "15m": {"window": 20, "tolerance": 0.002, "touches": 3},
    "1h":  {"window": 30, "tolerance": 0.0025, "touches": 3},
    "4h":  {"window": 40, "tolerance": 0.003, "touches": 3},
    "1d":  {"window": 80, "tolerance": 0.005, "touches": 2},
}

# ========== AQUÍ VAMOS A IR AÑADIENDO CÁLCULOS ==========


def add_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ejemplo de 'función por cálculo' (placeholder)."""
    if "close" in df.columns:
        df = df.copy()  # 🔒 evitar vistas
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR clásico (ejemplo simple)."""
    if not {"high", "low", "close"}.issubset(df.columns):
        return df
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr_all = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr_all.rolling(period).mean()
    return df


def detect_support_resistance(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str = 'BTC/USDC',
    logger: logging.Logger | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Detecta soportes y resistencias usando configuración adaptativa por timeframe.
    ✅ SIN SUFIJOS → columnas siempre consistentes: sr_level, rs_level, mid_level, etc.
    ✅ Añade `candles_back` indicando cuántas velas se escanearon.
    ✅ Propaga últimos niveles detectados para mantener "zona activa".
    """
    df = df.copy()

    # 🎯 Config dinámico
    config = SR_CONFIG.get(
        timeframe, {"window": 20, "tolerance": 0.002, "touches": 3})
    window = config["window"]
    tolerance = config["tolerance"]
    touches_required = config["touches"]

    # ✅ Inicializar SIEMPRE las mismas columnas (sin sufijo)
    base_cols = {
        "is_support": 0,
        "sr_strength": 0,
        "sr_level": float('nan'),
        "sr_range_low": float('nan'),
        "sr_range_high": float('nan'),
        "is_resistance": 0,
        "rs_strength": 0,
        "rs_level": float('nan'),
        "rs_range_low": float('nan'),
        "rs_range_high": float('nan'),
        "mid_level": float('nan'),
        "candles_back": window  # 👈 indica cuántas velas usa este TF para SR
    }
    for col, val in base_cols.items():
        if col not in df.columns:
            df[col] = val

    # 🧠 Detección SR solo hacia atrás (like before, but writing unified cols)
    for i in range(window, len(df)):
        window_slice = df.iloc[i-window:i+1]

        # === SOPORTE ===
        low_current = df["low"].iloc[i]
        if low_current == window_slice["low"].min():
            near_touches = (
                (window_slice["low"] - low_current).abs() / low_current < tolerance).sum()
            if near_touches >= touches_required:
                df.at[i, "is_support"] = 1
            df.at[i, "sr_strength"] = near_touches
            df.at[i, "sr_level"] = low_current
            df.at[i, "sr_range_low"] = low_current * (1 - tolerance)
            df.at[i, "sr_range_high"] = low_current * (1 + tolerance)

        # === RESISTENCIA ===
        high_current = df["high"].iloc[i]
        if high_current == window_slice["high"].max():
            near_touches = (
                (window_slice["high"] - high_current).abs() / high_current < tolerance).sum()
            if near_touches >= touches_required:
                df.at[i, "is_resistance"] = 1
            df.at[i, "rs_strength"] = near_touches
            df.at[i, "rs_level"] = high_current
            df.at[i, "rs_range_low"] = high_current * (1 - tolerance)
            df.at[i, "rs_range_high"] = high_current * (1 + tolerance)

    # 🔁 Forward-fill SR y RS
    for col in ["sr_level", "sr_range_low", "sr_range_high", "sr_strength",
                "rs_level", "rs_range_low", "rs_range_high", "rs_strength"]:
        df[col] = df[col].ffill()


    # 🧭 SR Macro extendido: si NO hay SR detectado en ventana,
    # buscar el mínimo más relevante fuera de ella (macro siguiente soporte)
    if df["sr_level"].iloc[-1] is np.nan or np.isnan(df["sr_level"].iloc[-1]):
        # Busca en TODO el histórico el mínimo más agrupado antes del rango
        macro_slice = df.iloc[:-window]  # velas fuera del rango reciente
        if len(macro_slice) > 0:
            macro_low = macro_slice["low"].min()
            df["sr_macro_level"] = macro_low
        else:
            df["sr_macro_level"] = np.nan
    else:
        df["sr_macro_level"] = np.nan  # solo mostrar macro si no hay fresh SR


    # 🎯 Nivel medio entre SR y RS
    df["mid_level"] = (df["sr_level"] + df["rs_level"]) / 2
    
    

    if logger:
        logger.debug(
            f"[SR] {symbol or ''} {timeframe} | window={window}, tolerance={tolerance}, touches={touches_required}")

    return df


# ========= INDICADORES =========

def add_ema(df: pd.DataFrame, periods=(9, 21, 50)) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if "close" not in df.columns:
        return df
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_vwap_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP por 'sesión' de calendario (día). Para cripto 24/7,
    resetea por fecha del índice si el índice es DatetimeIndex.
    Si no hay índice temporal, hace VWAP acumulado global sin groupby.
    """
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return df

    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0

    if isinstance(df.index, pd.DatetimeIndex):
        session = df.index.date  # agrupa por día natural
        cum_pv = (tp * df["volume"]).groupby(session).cumsum()
        cum_vol = df["volume"].groupby(session).cumsum()
        df["vwap"] = cum_pv / cum_vol
    else:
        # Sin index temporal -> VWAP acumulado global (ÚTIL PARA BATCH INCREMENTAL)
        df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def add_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend clásico (utiliza ATR simple).
    Crea columnas: 'supertrend', 'st_direction' (1 alcista, -1 bajista)
    """
    req = {"high", "low", "close"}
    if not req.issubset(df.columns):
        return df
    df = df.copy()

    # ATR simple
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(),
                   (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # bandas básicas
    hl2 = (h + l) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    # cálculos con arrastre
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    st.iloc[0] = np.nan
    direction.iloc[0] = 0

    for i in range(1, len(df)):
        curr_close = c.iloc[i]
        prev_close = c.iloc[i-1]
        prev_st = st.iloc[i-1]
        prev_dir = direction.iloc[i-1]

        ub = upperband.iloc[i]
        lb = lowerband.iloc[i]

        # ajuste de bandas
        if not np.isnan(prev_st):
            ub = min(ub, prev_st) if prev_dir == 1 else ub
            lb = max(lb, prev_st) if prev_dir == -1 else lb

        # dirección
        if np.isnan(prev_st):
            # inicializar con dirección según precio vs banda
            dir_i = 1 if curr_close > lb else -1
            st_i = lb if dir_i == 1 else ub
        else:
            if prev_close <= prev_st and curr_close > prev_st:
                dir_i = 1
            elif prev_close >= prev_st and curr_close < prev_st:
                dir_i = -1
            else:
                dir_i = prev_dir

            st_i = lb if dir_i == 1 else ub

        st.iloc[i] = st_i
        direction.iloc[i] = dir_i

    df["supertrend"] = st
    df["st_direction"] = direction.replace(0, np.nan)
    return df

# ========= SEÑALES (INTRADÍA LONG/SHORT) =========


def generate_intraday_signals(
    df: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    ema_trend: int = 50,
    rsi_min_long: int = 50,
    rsi_max_short: int = 50,
    breakout_lookback: int = 20,
    atr_period: int = 14,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    use_vwap_filter: bool = True,
    use_supertrend_filter: bool = True,
    prefer_sr_bounce: bool = True
) -> pd.DataFrame:
    """
    Crea columnas:
      - signal: 'long_entry', 'short_entry', 'long_exit', 'short_exit', '' (sin señal)
      - reason: texto breve explicando el porqué
      - sl, tp: stop y take profit planificados (ATR)
    Requiere columnas OHLCV + EMAs/RSI/Supertrend si activas filtros.
    """
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return df

    df = df.copy()

    # Asegurar indicadores mínimos
    if f"ema_{ema_fast}" not in df.columns or f"ema_{ema_slow}" not in df.columns or f"ema_{ema_trend}" not in df.columns:
        df = add_ema(df, periods=(ema_fast, ema_slow, ema_trend))
    if "rsi" not in df.columns:
        df = add_rsi(df, period=14)
    if use_vwap_filter and "vwap" not in df.columns:
        df = add_vwap_session(df)
    if use_supertrend_filter and "supertrend" not in df.columns:
        df = add_supertrend(df, atr_period=10, multiplier=3.0)

    # ATR para SL/TP
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(),
                   (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    df["atr"] = df.get("atr", atr)  # respeta si ya existe, si no, añade

    # cruces EMA
    ema_f = df[f"ema_{ema_fast}"]
    ema_s = df[f"ema_{ema_slow}"]
    ema_t = df[f"ema_{ema_trend}"]

    cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    cross_dn = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

    # breakouts
    recent_high = df["high"].rolling(breakout_lookback).max().shift(1)
    recent_low = df["low"].rolling(breakout_lookback).min().shift(1)
    bo_up = df["close"] > recent_high
    bo_dn = df["close"] < recent_low

    # soporte / resistencia (opcional si ya lo calculaste)
    has_sr = {"is_support", "is_resistance", "sr_range_low",
              "sr_range_high"}.issubset(df.columns)

    # columnas resultado
    df["signal"] = ""
    df["reason"] = ""
    df["sl"] = np.nan
    df["tp"] = np.nan

    # estado: 0 sin posición, 1 long, -1 short
    position = 0

    for i in range(1, len(df)):
        reasons = []
        price = df["close"].iloc[i]
        atr_i = atr.iloc[i]

        trend_ok_long = price > ema_t.iloc[i]
        trend_ok_short = price < ema_t.iloc[i]

        rsi_i = df["rsi"].iloc[i]
        rsi_ok_long = (rsi_i >= rsi_min_long) if not np.isnan(rsi_i) else False
        rsi_ok_short = (rsi_i <= rsi_max_short) if not np.isnan(
            rsi_i) else False

        vwap_ok_long = True
        vwap_ok_short = True
        if use_vwap_filter and "vwap" in df.columns:
            vwap_i = df["vwap"].iloc[i]
            vwap_ok_long = price >= vwap_i
            vwap_ok_short = price <= vwap_i

        st_ok_long = True
        st_ok_short = True
        if use_supertrend_filter and "supertrend" in df.columns and "st_direction" in df.columns:
            st_dir = df["st_direction"].iloc[i]
            st_ok_long = (st_dir == 1)
            st_ok_short = (st_dir == -1)

        # preferencia por rebotes en SR
        sr_long = sr_short = True
        if prefer_sr_bounce and has_sr:
            in_support = (df["is_support"].iloc[i] == 1) or (
                df["sr_range_low"].iloc[i] <= price <= df["sr_range_high"].iloc[i] and df["sr_level"].iloc[i] == df["low"].iloc[i]
            )
            in_resist = (df["is_resistance"].iloc[i] == 1) or (
                df["sr_range_low"].iloc[i] <= price <= df["sr_range_high"].iloc[i] and df["sr_level"].iloc[i] == df["high"].iloc[i]
            )
            # rebote: long cerca de soporte, short cerca de resistencia
            sr_long = in_support or bo_up.iloc[i]
            sr_short = in_resist or bo_dn.iloc[i]

        # === ENTRADAS ===
        if position == 0:
            # LONG
            if ((cross_up.iloc[i] or bo_up.iloc[i]) and trend_ok_long and rsi_ok_long and vwap_ok_long and st_ok_long and sr_long):
                position = 1
                sl = price - sl_atr_mult * \
                    atr_i if not np.isnan(atr_i) else np.nan
                tp = price + tp_atr_mult * \
                    atr_i if not np.isnan(atr_i) else np.nan
                df.at[df.index[i], "signal"] = "long_entry"
                reasons.append(
                    "Cruce EMA rápida>lenta" if cross_up.iloc[i] else "")
                reasons.append("Breakout de máximos" if bo_up.iloc[i] else "")
                if trend_ok_long:
                    reasons.append(f"Tendencia>EMA{ema_trend}")
                if rsi_ok_long:
                    reasons.append(f"RSI≥{rsi_min_long}")
                if use_vwap_filter and vwap_ok_long:
                    reasons.append("Precio≥VWAP")
                if use_supertrend_filter and st_ok_long:
                    reasons.append("Supertrend alcista")
                if prefer_sr_bounce and has_sr and sr_long:
                    reasons.append("Cerca de soporte / confirmación SR")
                df.at[df.index[i], "reason"] = ", ".join(
                    [r for r in reasons if r])
                df.at[df.index[i], "sl"] = sl
                df.at[df.index[i], "tp"] = tp

            # SHORT
            elif ((cross_dn.iloc[i] or bo_dn.iloc[i]) and trend_ok_short and rsi_ok_short and vwap_ok_short and st_ok_short and sr_short):
                position = -1
                sl = price + sl_atr_mult * \
                    atr_i if not np.isnan(atr_i) else np.nan
                tp = price - tp_atr_mult * \
                    atr_i if not np.isnan(atr_i) else np.nan
                df.at[df.index[i], "signal"] = "short_entry"
                reasons.append(
                    "Cruce EMA rápida<lenta" if cross_dn.iloc[i] else "")
                reasons.append("Breakdown de mínimos" if bo_dn.iloc[i] else "")
                if trend_ok_short:
                    reasons.append(f"Tendencia<EMA{ema_trend}")
                if rsi_ok_short:
                    reasons.append(f"RSI≤{rsi_max_short}")
                if use_vwap_filter and vwap_ok_short:
                    reasons.append("Precio≤VWAP")
                if use_supertrend_filter and st_ok_short:
                    reasons.append("Supertrend bajista")
                if prefer_sr_bounce and has_sr and sr_short:
                    reasons.append("Cerca de resistencia / confirmación SR")
                df.at[df.index[i], "reason"] = ", ".join(
                    [r for r in reasons if r])
                df.at[df.index[i], "sl"] = sl
                df.at[df.index[i], "tp"] = tp

        # === SALIDAS (señales de cierre) ===
        else:
            exit_reason = []
            if position == 1:
                # salida por cruce contrario, giro de supertrend, o pérdida de VWAP/tendencia
                if cross_dn.iloc[i]:
                    exit_reason.append("Cruce contrario EMA")
                if use_supertrend_filter and df["st_direction"].iloc[i] == -1:
                    exit_reason.append("Supertrend gira bajista")
                if use_vwap_filter and price < df["vwap"].iloc[i]:
                    exit_reason.append("Pierde VWAP")
                if price < ema_t.iloc[i]:
                    exit_reason.append(f"Pierde EMA{ema_trend}")

                if exit_reason:
                    position = 0
                    df.at[df.index[i], "signal"] = "long_exit"
                    df.at[df.index[i], "reason"] = ", ".join(exit_reason)

            elif position == -1:
                if cross_up.iloc[i]:
                    exit_reason.append("Cruce contrario EMA")
                if use_supertrend_filter and df["st_direction"].iloc[i] == 1:
                    exit_reason.append("Supertrend gira alcista")
                if use_vwap_filter and price > df["vwap"].iloc[i]:
                    exit_reason.append("Recupera VWAP")
                if price > ema_t.iloc[i]:
                    exit_reason.append(f"Recupera EMA{ema_trend}")

                if exit_reason:
                    position = 0
                    df.at[df.index[i], "signal"] = "short_exit"
                    df.at[df.index[i], "reason"] = ", ".join(exit_reason)

    return df

# en transforms.py


def output_last_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform “de salida” que imprime el diagnóstico de la última vela
    y devuelve el df intacto. Evita lambdas y problemas de scope/pickling.
    """
    try:
        # import local para evitar import circular al cargar transforms
        from .show_data import diagnose_last_candle
    except Exception as e:
        print(f"⚠ No pude importar diagnose_last_candle: {e}")
        return df

    # intenta leer el par del propio DataFrame si lo has guardado antes:
    pair = getattr(df, "attrs", {}).get("pair", "BTC/USDC")
    try:
        diagnose_last_candle(df, pair=pair)  # solo imprime
    except Exception as e:
        print(f"⚠ Error en diagnose_last_candle: {e}")
    return df


# ========= REGISTRO DE TRANSFORMS =========
# (añade estos a los que ya tienes)
def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ema(df, periods=(9, 21, 50))
    df = add_rsi(df, period=14)
    df = add_vwap_session(df)
    df = add_supertrend(df, atr_period=10, multiplier=3.0)
    return df


def make_transforms(df: pd.DataFrame, timeframe: str, symbol: str, logger: logging.Logger) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame vacío o None en make_transforms")

    logger.debug(f"⚙ Aplicando transforms a {len(df)} filas nuevas en {symbol} {timeframe}...")

    # 1️⃣ Indicadores base
    df = add_intraday_indicators(df)

    # 2️⃣ SR/RS limpio (SIN window manual)
    df = detect_support_resistance(df, timeframe=timeframe, symbol=symbol, logger=logger)

    # 3️⃣ ATR
    df = add_atr(df)

    return df

