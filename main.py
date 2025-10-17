#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
MAIN MULTIFRAME 24/7
- Primera ejecución: baja todos los TF por cada par y guarda.
- Loop cada minuto: solo nuevas velas, aplica transforms incrementales y muestra análisis MTF.
"""

import os
import time
from datetime import datetime, timezone
import logging
import pandas as pd

from .log import setup_logger_with_colors

from .config import PAIRS, SINCE_UTC, DATA_DIR, META_DIR
from .download import EXCHANGE
from .utils import ensure_dirs, ensure_file, feather_path, meta_path, load_meta, save_meta
from .save import save_feather
from .transforms import make_transforms
from .mtf_view import MTF_TACTICS
from colorlog import ColoredFormatter


# Timeframes a mantener vivos (puedes añadir/ordenar)
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
OFFSET_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}

LOGS_DIR = "binatrend/logs"

logger = logging.getLogger(__name__)

SR_CONFIG = {
    "15m": {"window": 20, "tolerance": 0.002, "touches": 3},
    "1h":  {"window": 30, "tolerance": 0.0025, "touches": 3},
    "4h":  {"window": 40, "tolerance": 0.003, "touches": 3},
    "1d":  {"window": 80, "tolerance": 0.005, "touches": 2},
}


def update_pair_timeframe(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Descarga/actualiza un timeframe específico, aplica transforms SOLO a nuevas filas,
    guarda y devuelve el DataFrame completo.
    """
    ex = EXCHANGE(logger)

    ensure_dirs(DATA_DIR, META_DIR)

    fpath = feather_path(DATA_DIR, symbol, timeframe)
    mpath = meta_path(META_DIR, symbol, timeframe)
    meta = load_meta(mpath)

    df, since_dt = ex.sync_df(
        fpath=fpath, symbol=symbol, timeframe=timeframe, meta=meta, logger=logger
    )

    # logger.info("Columnas actuales: %s", df.columns.tolist() if df is not None else "Ninguna")

    if since_dt is None:
        logger.debug(
            f"⚠ No se pudo sincronizar {symbol} {timeframe}. Se descarga todo.")
        since_dt = SINCE_UTC
        since_dt.replace(tzinfo=None)  # naive UTC

    if df is not None and "date" in df.columns and len(df) > 0:
        last_date = df["date"].iloc[-1]
        logger.debug(
            f"DataFrame actual tiene {len(df)} filas. Última fecha: {last_date}")
    # Empuja una vela para evitar solape (según TF)
    offset_min = OFFSET_MINUTES.get(timeframe, 1)
    since_dt = since_dt + pd.Timedelta(minutes=offset_min)

    df_bnc = ex.fetch_ohlcv_all(
        symbol=symbol,
        timeframe=timeframe,
        since_ms=int(pd.Timestamp(since_dt).timestamp() * 1000),
    )
    if df_bnc is not None and len(df_bnc) > 0:
        df_bnc['date'] = pd.to_datetime(df_bnc['date'], utc=True)
        df_bnc = make_transforms(
            df_bnc, symbol=symbol, timeframe=timeframe, logger=logger)
        

    # Merge con histórico
    if df is not None and len(df) > 0:
        if df_bnc is not None and len(df_bnc) > 0:
            df = pd.concat([df, df_bnc], ignore_index=True).drop_duplicates(
                subset="date").sort_values("date").reset_index(drop=True)
            df = df.copy()
        else:
            df = (
                df.drop_duplicates(subset="date")
                .sort_values("date")
                .reset_index(drop=True)
            )
    else:
        df = df_bnc.copy()
    logger.debug(
        f"🔄 Total periodos en analisis: {len(df)} en {symbol} {timeframe}")
    # Guardar meta + feather
    last_date = df["date"].iloc[-1] if len(df) > 0 else None

    if last_date:
        logger.debug(
            f"🗓 Última fecha tras descarga {symbol} {timeframe} {last_date}")
        meta["last_date"] = df["date"].iloc[-1].isoformat()
        save_meta(mpath, meta)
        save_feather(df, fpath)
        logger.debug(
            f"✅ {symbol} {timeframe} actualizado. Total filas: {len(df)}")
    else:
        logger.debug(
            f"⚠ No hay datos tras descarga para {symbol} {timeframe}\n")
    logger.debug("Total columnas finales: %s", df.columns.tolist()
                 if df is not None else "Ninguna")
    return df


def run_update_and_analysis():
    """
    Descarga/actualiza todos los TF por par, construye tf_map y lanza análisis MTF.
    Ahora soporta MULTIPAR: si hay varios pares → render_multi_pair_panel()
    """
    mtf = MTF_TACTICS(logger=logger)
    logger.info("🔄 Iniciando ciclo de actualización y análisis MTF...")

    multi_pair_evals = {}  # 👈 Aquí acumulamos evals por PAR

    for pair in PAIRS:
        logger.debug("\n" + "=" * 70)
        logger.debug(
            "🔥 %s UTC | Procesando %s",
            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            pair
        )

        tf_map = {}
        for tf in TIMEFRAMES:
            try:
                df_tf = update_pair_timeframe(pair, tf)
                logger.debug(
                    f"✅ {pair} {tf} actualizado con {len(df_tf)} filas.")
                tf_map[tf] = df_tf
            except Exception as e:
                logger.debug(f"❌ Error actualizando {pair} {tf}: {e}")
                raise e

        # Análisis por par → guardamos resultado para MULTIPAR
        try:
            evals = mtf.generate_multiframe_tactics(tf_map)
            multi_pair_evals[pair] = evals  # 👈 ACUMULAR
        except Exception as e:
            logger.debug(f"❌ Error generando táctica MTF para {pair}: {e}")
            raise e

    # 🎯 Al terminar TODOS los pares → decidir si MULTIPAR o PANEL normal
    if len(multi_pair_evals) > 1:
        logger.debug("🌐 MULTIPAR DETECTADO — renderizando panel agrupado...")
        mtf.render_multi_pair_panel(multi_pair_evals)
    else:
        # Solo 1 par → mostrar como siempre
        pair = list(multi_pair_evals.keys())[0]
        mtf.render_tactical_panel(
            pair=pair, tf_evals_map=multi_pair_evals[pair])


def sleep_until_next_minute():
    """
    Duerme hasta el inicio del próximo minuto para que el loop quede alineado.
    """
    now = time.time()
    # segundos hasta el siguiente minuto exacto
    wait = 60 - (now % 60)
    if 0 < wait < 60:
        time.sleep(wait)


def main():
    today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR + "/log_" + today_str + ".txt"
    ensure_file(log_file)
    setup_logger_with_colors(log_file)
    logger.debug("🟢 Iniciando Binatrend MTF 24/7")
    # Primera pasada (descarga completa y análisis)
    run_update_and_analysis()

    #
    # logger.debug("\n🔁 Entrando en loop 24/7 (tick cada minuto)...\n")
    while True:
        try:
            sleep_until_next_minute()
            run_update_and_analysis()
        except KeyboardInterrupt:
            logger.debug("\n🛑 Interrumpido por usuario. Saliendo...")
            break
        except Exception as e:
            logger.debug(f"💥 Excepción en loop principal: {e}")
            # Evita bucles pegados; espera un poco y reintenta
            time.sleep(5)


if __name__ == "__main__":
    main()
