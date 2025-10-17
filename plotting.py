#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/plotting.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Thursday, October 16th 2025, 3:41:06 pm
# Author: Tomás
# -----
# Last Modified:  Thursday, 16th October 2025 3:41:11 pm
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
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

def plot_support_resistance(df: pd.DataFrame, df_levels: pd.DataFrame, max_levels: int = 10):
    """
    Dibuja gráfico con soportes y resistencias detectados.
    max_levels: limita líneas de soporte/resistencia más relevantes.
    """

    # Limitar niveles a los más relevantes (más cercanos al precio actual)
    current_price = df["close"].iloc[-1]

    support_levels = sorted([lvl for lvl in df_levels["support_levels"].dropna()], key=lambda x: abs(x - current_price))[:max_levels]
    resistance_levels = sorted([lvl for lvl in df_levels["resistance_levels"].dropna()], key=lambda x: abs(x - current_price))[:max_levels]

    plt.figure(figsize=(14, 7))

    # Dibujar velas simplificadas (OHLC box)
    for idx, row in df.tail(300).iterrows():  # Últimas 300 velas
        color = "green" if row["close"] >= row["open"] else "red"
        plt.plot([idx, idx], [row["low"], row["high"]], color="black", linewidth=0.6)
        rect = Rectangle(
            (idx - 0.3, min(row["open"], row["close"])),
            0.6,
            abs(row["close"] - row["open"]),
            linewidth=0,
            facecolor=color,
            alpha=0.5
        )
        plt.gca().add_patch(rect)

    # Dibujar soportes
    for lvl in support_levels:
        plt.axhline(lvl, linestyle="--", linewidth=1, label=f"Soporte {lvl:.2f}")

    # Dibujar resistencias
    for lvl in resistance_levels:
        plt.axhline(lvl, linestyle="-.", linewidth=1, label=f"Resistencia {lvl:.2f}")

    plt.title(f"Detección de Soportes y Resistencias - {df.index.name or 'Precio'}")
    plt.xlabel("Velas")
    plt.ylabel("Precio")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
