#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/show_data.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 6:29:54 pm
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 12:19:20 am
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
import pandas as pd

def diagnose_last_candle(df: pd.DataFrame, pair: str = "BTC/USDC", ema_trend=50):
    """Diagnóstico operativo aunque no haya señal."""
    if df.empty:
        print("⚠ DataFrame vacío.")
        return

    last = df.iloc[-1]
    idx = df.index[-1]
    
    ts = idx if not isinstance(idx, pd.Timestamp) else idx.strftime("%Y-%m-%d %H:%M")
    price = round(last.get("close", float("nan")), 2)

    # Detectar señal si existe
    signal = str(last.get("signal", "")).strip()
    reason = last.get("reason", "")

    # Estados técnicos
    ema_trend_val = last.get(f"ema_{ema_trend}", None)
    ema_bias = None
    if ema_trend_val:
        ema_bias = "LONG" if price > ema_trend_val else "SHORT"

    rsi = last.get("rsi", None)
    vwap = last.get("vwap", None)
    st_dir = last.get("st_direction", None)

    # Diagnóstico rápido
    print(f"\n📍 Última vela {pair} @ {ts}")
    print(f"💰 Precio: {price}")

    if signal:
        icons = {
            "long_entry": "🟢 LONG ENTRY",
            "short_entry": "🔴 SHORT ENTRY",
            "long_exit": "⚪ EXIT LONG",
            "short_exit": "⚪ EXIT SHORT"
        }
        print(f"✅ Señal detectada: {icons.get(signal, signal)}")
        print(f"🔎 Razón: {reason}")
        sl = last.get("sl", None)
        tp = last.get("tp", None)
        print(f"🎯 SL: {round(sl,2) if sl==sl else 'N/A'} | TP: {round(tp,2) if tp==tp else 'N/A'}")
    else:
        print("⚙ No hay señal formal de entrada ahora mismo.\n")
        print("📊 Estado técnico:")
        if ema_trend_val:
            print(f"   - Tendencia general (EMA{ema_trend}): {'✅ Precio por encima → Sesgo LONG' if price > ema_trend_val else '❌ Precio por debajo → Sesgo SHORT'}")
        if rsi == rsi:
            print(f"   - RSI: {round(rsi,2)} {'(fuerza alcista)' if rsi >= 50 else '(presión bajista)'}")
        if vwap == vwap:
            print(f"   - VWAP: {'✅ Precio por encima → flujo comprador' if price >= vwap else '❌ Precio por debajo → presión vendedora'}")
        if st_dir in [1, -1]:
            print(f"   - Supertrend: {'✅ Alcista' if st_dir == 1 else '❌ Bajista'}")
        sr_support = last.get("is_support", 0)
        sr_resistance = last.get("is_resistance", 0)
        if sr_support == 1:
            print("   - 📌 Zona de soporte detectada (potencial rebote)")
        if sr_resistance == 1:
            print("   - ⚠ Zona de resistencia detectada (riesgo de rechazo)")

        print("\n🎯 Interpretación táctica:")
        if ema_bias == "LONG" and rsi >= 50 and price >= vwap:
            print("   → Mercado con sesgo LONG. Esperar confirmación (cruce EMA9>21 o breakout).")
        elif ema_bias == "SHORT" and rsi < 50 and price <= vwap:
            print("   → Sesgo SHORT. Esperar reacción en resistencia o breakdown.")
        else:
            print("   → Mercado mixto. Esperar claridad técnica antes de actuar.")


def log_signals_to_terminal(df: pd.DataFrame, pair: str = "BTC/USDC"):
    """Imprime en terminal las señales detectadas en las últimas filas procesadas."""
    # Filtrar solo filas con señal
    signals = df[df["signal"] != ""]
    print(f"\nSeñales detectadas para {pair}:")
    for idx, row in signals.iterrows():
        ts = idx if not isinstance(idx, pd.Timestamp) else idx.strftime("%Y-%m-%d %H:%M")
        direction_icon = {
            "long_entry": "🟢 LONG",
            "short_entry": "🔴 SHORT",
            "long_exit": "⚪ EXIT LONG",
            "short_exit": "⚪ EXIT SHORT"
        }.get(row["signal"], "⚙ SIGNAL")

        price = round(row["close"], 2)
        reason = row.get("reason", "")
        is_support = row.get("is_support", False)
        is_resistance = row.get("is_resistance", False)
        sr_strength = row.get("sr_strength", 0)
        sr_range_low = row.get("sr_range_low", 0)
        sr_range_high = row.get("sr_range_high", 0)

        print(f"{ts} | {direction_icon} {pair} @ {price} | {reason} | SR Strength: {sr_strength} | Range: [{sr_range_low}, {sr_range_high}] | Support: {is_support} | Resistance: {is_resistance}")
        # print(f"{ts} | {direction_icon} {pair} @ {price} | {reason}")
    if signals.empty:
        print("No se detectaron señales en el DataFrame proporcionado.")

def log_last_signal(df: pd.DataFrame, pair: str = "BTC/USDC"):
    """Muestra solo la última vela y la acción sugerida."""
    if df.empty:
        print("⚠ DataFrame vacío.")
        return
    
    last = df.iloc[-1]
    idx = df.index[-1]

    ts = idx if not isinstance(idx, pd.Timestamp) else idx.strftime("%Y-%m-%d %H:%M")
    signal = str(last.get("signal", "")).strip()

    price = round(last.get("close", float("nan")), 2)
    sl = last.get("sl", None)
    tp = last.get("tp", None)
    reason = last.get("reason", "")

    # Formato visual
    direction_icon = {
        "long_entry": "🟢 LONG ENTRY",
        "short_entry": "🔴 SHORT ENTRY",
        "long_exit": "⚪ EXIT LONG",
        "short_exit": "⚪ EXIT SHORT"
    }.get(signal, "🤷 NO ACTION")

    print(f"\n📍 Última vela {pair} @ {ts}")
    print(f"💰 Precio: {price}")
    print(f"➡ Acción sugerida: {direction_icon}")
    if reason:
        print(f"🔎 Razón: {reason}")
    if signal in ["long_entry", "short_entry"]:
        print(f"🎯 SL: {round(sl,2) if sl==sl else 'N/A'} | TP: {round(tp,2) if tp==tp else 'N/A'}")
