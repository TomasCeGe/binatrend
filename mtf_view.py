# === MTF TACTICS ==========================================================
# Requisitos: cada DF debe traer al menos:
#   close, ema_50, rsi, vwap, st_direction (1 alcista, -1 bajista)
# Opcional: signal, reason, sl, tp (si usas generate_intraday_signals en 15m)
# ==========================================================================

from logging import Logger
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import os
import math

# ANSI colores
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAG    = "\033[95m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

class MTF_TACTICS():
    
    logger: Logger
    SR_CONFIG = {
        "15m": {"window": 20, "tolerance": 0.002, "touches": 3},
        "1h":  {"window": 30, "tolerance": 0.0025, "touches": 3},
        "4h":  {"window": 40, "tolerance": 0.003, "touches": 3},
        "1d":  {"window": 80, "tolerance": 0.005, "touches": 2},
    }
    
    def __init__(self,logger: Logger):
        self.console = Console()
        self.logger = logger

    def _safe_get_last(self, df, col: str, default=np.nan):
        return df[col].iloc[-1] if (col in df.columns and len(df) > 0) else default

    def _tf_label(self,tf_key: str) -> str:
        # normaliza etiquetas para la tabla
        m = {
            "15m": "15m", "15min": "15m", "15": "15m",
            "1h": "1H", "1H": "1H", "60m": "1H",
            "4h": "4H", "4H": "4H", "240m": "4H",
            "1d": "1D", "1D": "1D", "D": "1D",
            "macro": "Macro", "1w": "Macro", "1W": "Macro"
        }
        return m.get(tf_key, tf_key)

    def evaluate_single_timeframe(self, df: pd.DataFrame, ema_trend: int = 50, timeframe: str = '1h') -> dict:
        """
        Evalúa la última vela sin sufijos SR/RS.
        Salida unificada: sr_level, rs_level, mid_level, candles_back, etc.
        """
        out = {
            "bias": "Neutral",
            "score": 0,
            "price": np.nan,
            "ema": np.nan,
            "rsi": np.nan,
            "vwap": np.nan,
            "st_dir": 0,
            "rsi_ok": None,
            "vwap_ok": None,
            "st_txt": "N/A",
            "note": "Sin datos suficientes",
            "sr_level": np.nan,
            "sr_range_low": np.nan,
            "sr_range_high": np.nan,
            "sr_strength": 0,
            "rs_level": np.nan,
            "rs_range_low": np.nan,
            "rs_range_high": np.nan,
            "rs_strength": 0,
            "mid_level": np.nan,
            "candles_back": df.get("candles_back", np.nan).iloc[-1] if "candles_back" in df.columns else np.nan
        }

        if df is None or len(df) == 0:
            self.logger.error(f"No hay datos para evaluar timeframe {timeframe}")
            return out

        last = df.iloc[-10]
        self.logger.debug(f"Última fila para {timeframe}: {last.to_dict()}")
        # === Base fields ===
        price = last.get("close", np.nan)
        ema   = last.get(f"ema_{ema_trend}", np.nan)
        rsi   = last.get("rsi", np.nan)
        vwap  = last.get("vwap", np.nan)
        st_dir = last.get("st_direction", np.nan)

        # === SR/RS direct fields ===
        sr_level = last.get("sr_level", np.nan)
        rs_level = last.get("rs_level", np.nan)
        mid_level = last.get("mid_level", np.nan)

        # === Bias logic ===
        ema_bias = "Long" if price > ema else ("Short" if price < ema else "Neutral")
        rsi_ok = (rsi >= 50) if not np.isnan(rsi) else False
        vwap_ok = (price >= vwap) if not np.isnan(vwap) else False
        st_txt = "Alcista" if st_dir == 1 else ("Bajista" if st_dir == -1 else "Indefinido")

        votes = 0
        if ema_bias == "Long": votes += 1
        if ema_bias == "Short": votes -= 1
        votes += 0.5 if rsi_ok else -0.5 if not np.isnan(rsi) else 0
        votes += 0.5 if vwap_ok else -0.5 if not np.isnan(vwap) else 0
        votes += 0.5 if st_dir == 1 else -0.5 if st_dir == -1 else 0

        bias = "Long" if votes >= 1 else "Short" if votes <= -1 else "Neutral"

        note = (
            "Sesgo LONG; buscar pullback a EMA9/21 o confirmación de breakout." if bias == "Long" else
            "Sesgo SHORT; buscar rebote a resistencias o breakdown con volumen." if bias == "Short" else
            "Mixto/neutral; paciencia hasta señal clara."
        )

        out.update({
            "bias": bias,
            "score": votes,
            "price": price,
            "ema": ema,
            "rsi": rsi,
            "vwap": vwap,
            "st_dir": st_dir,
            "rsi_ok": rsi_ok,
            "vwap_ok": vwap_ok,
            "st_txt": st_txt,
            "note": note,
            "sr_level": sr_level,
            "sr_range_low": last.get("sr_range_low", np.nan),
            "sr_range_high": last.get("sr_range_high", np.nan),
            "sr_strength": last.get("sr_strength", np.nan),
            "rs_level": rs_level,
            "rs_range_low": last.get("rs_range_low", np.nan),
            "rs_range_high": last.get("rs_range_high", np.nan),
            "rs_strength": last.get("rs_strength", np.nan),
            "mid_level": mid_level
        })

        return out

    def _bias_icon(self,bias: str) -> str:
        if bias == "Long": return f"{GREEN}🟢 Long{RESET}"
        if bias == "Short": return f"{RED}🔴 Short{RESET}"
        return f"{YELLOW}⚠ Neutral{RESET}"

    def _fmt_state(self,rsi, vwap_ok, st_txt):
        rsi_txt = "RSI N/A" if rsi != rsi else f"RSI {round(rsi,1)}"
        vwap_txt = "VWAP OK" if vwap_ok else "VWAP -" if vwap_ok is not None else "VWAP N/A"
        st = {"Alcista": "ST+", "Bajista": "ST-", "Indefinido": "ST?"}.get(st_txt, "ST?")
        return f"{rsi_txt} / {vwap_txt} / {st}"

    def print_table_header(self,pair: str):
        self.logger.debug("\n" + "─" * 60)
        self.logger.debug(f"{BLUE}📊 VISIÓN MULTIMARCO {pair}{RESET}")
        self.logger.debug("─" * 60)
        self.logger.debug(f"{'TF':7} | {'Sesgo':10} | {'Estado RSI/VWAP/ST':24} | Nota táctica")
        self.logger.debug("─" * 60)

    def print_table_row(self,tf_key: str, evald: dict):
        tf = f"{self._tf_label(tf_key):7}"
        bias = f"{self._bias_icon(evald['bias']):10}"
        state = f"{self._fmt_state(evald['rsi'], evald['vwap_ok'], evald['st_txt']):24}"
        note = evald["note"]
        self.logger.debug(f"{tf} | {bias} | {state} | {note}")

    def print_conclusion(self,conclusion_lines):
        self.logger.debug("─" * 60)
        self.logger.debug(f"{MAG}🎯 Conclusión operativa:{RESET}")
        for line in conclusion_lines:
            self.logger.debug("   " + line)
        self.logger.debug("─" * 60)

    def generate_multiframe_tactics(self, tf_dfs: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        tf_dfs: dict como {"15m": df_15m, "1h": df_1h, "4h": df_4h, "1d": df_1d}
        Evalúa cada timeframe y devuelve dict limpio sin sufijos.
        """
        # Orden sugerido de ejecución
        tf_order = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}
        ordered_keys = sorted(tf_dfs.keys(), key=lambda x: tf_order.get(x, 999))

        evals = {}
        for tf_key in ordered_keys:
            df = tf_dfs[tf_key]
            self.logger.debug(f"Evaluando marco {tf_key} ({len(df)} filas)...")
            evals[tf_key] = self.evaluate_single_timeframe(df, timeframe=tf_key)

        # 🎯 Conclusión operativa MTF (sin sufijos ni lógica window)
        b15 = evals.get("15m", {}).get("bias", "Neutral")
        b1h = evals.get("1h", {}).get("bias", "Neutral")
        b4h = evals.get("4h", {}).get("bias", "Neutral")
        b1d = evals.get("1d", {}).get("bias", "Neutral")

        conclusion = []
        if b15 == "Long":
            if b1h == "Long" and b4h in ("Long", "Neutral"):
                conclusion.append("🟢 Long permitido: 1H confirma, 4H alineado/neutro.")
            elif b1h == "Neutral":
                conclusion.append("🟡 Long en espera: 1H está neutral.")
            else:
                conclusion.append("🔴 Bloquear Long: 1H contradice.")
        elif b15 == "Short":
            if b1h == "Short" and b4h in ("Short", "Neutral"):
                conclusion.append("🔴 Short permitido: 1H confirma, 4H alineado/neutro.")
            elif b1h == "Neutral":
                conclusion.append("🟡 Short en espera: 1H está neutral.")
            else:
                conclusion.append("🟢 Bloquear Short: 1H contradice.")
        else:
            conclusion.append("⚪ 15m neutral: no forzar entradas.")

        # Contexto macro diario
        if b1d == "Long":
            conclusion.append("📈 1D alcista: viento de cola para compras.")
        elif b1d == "Short":
            conclusion.append("📉 1D bajista: cuidado con largos, puede dar presión de venta.")

        # Log interno (si quieres ver EVALS en crudo para debug)
        # self.logger.debug(f"Evals MTF: {evals}")
        for line in conclusion:
            self.logger.debug(f"• {line}")

        # 🚀 Ya entregamos los evals "limpios" sin sufijos para el panel
        return evals

    # ====== (Opcional) Filtro sobre señales del 15m ===========================
    def filter_15m_signal_by_mtf(self,df_15m: pd.DataFrame, evals_mtf: Dict[str, dict]) -> Tuple[str, str]:
        """
        Si la última vela de 15m trae 'signal' (long_entry/short_entry),
        valida/invalidala según el sesgo de 1H y 4H.
        Devuelve (status, msg) donde status ∈ {'ENTRY_ALLOWED','ENTRY_BLOCKED','NO_SIGNAL'}.
        """
        if df_15m is None or len(df_15m) == 0 or "signal" not in df_15m.columns:
            return ("NO_SIGNAL", "No hay señal en 15m o no existe columna 'signal'.")

        sig = str(df_15m["signal"].iloc[-1] or "").strip()
        if sig not in ("long_entry", "short_entry"):
            return ("NO_SIGNAL", "La última vela 15m no es entrada (puede ser exit o vacío).")

        b1h = evals_mtf.get("1h", {}).get("bias", "Neutral")
        b4h = evals_mtf.get("4h", {}).get("bias", "Neutral")

        if sig == "long_entry":
            if b1h == "Long" and b4h in ("Long", "Neutral"):
                return ("ENTRY_ALLOWED", "Long permitido por concordancia MTF (1H ok, 4H ok/neutral).")
            elif b1h == "Neutral":
                return ("ENTRY_BLOCKED", "Esperar confirmación: 1H neutral.")
            else:
                return ("ENTRY_BLOCKED", f"Bloqueado: 1H={b1h}.")
        else:  # short_entry
            if b1h == "Short" and b4h in ("Short", "Neutral"):
                return ("ENTRY_ALLOWED", "Short permitido por concordancia MTF (1H ok, 4H ok/neutral).")
            elif b1h == "Neutral":
                return ("ENTRY_BLOCKED", "Esperar confirmación: 1H neutral.")
            else:
                return ("ENTRY_BLOCKED", f"Bloqueado: 1H={b1h}.")

    def _assign_role(self, tf_key: str) -> str:
        """
        Devuelve el rol táctico según el timeframe.
        """
        if tf_key == "15m":
            return "🟢 GATILLO"
        elif tf_key == "1h":
            return "✅ CONFIRMA"
        elif tf_key == "4h":
            return "🌊 TENDENCIA"
        elif tf_key == "1d":
            return "🧭 MACRO"
        else:
            return "ℹ CONTEXTO"

    def check_mtf_alert(self, evals: dict, pair: str):
        """
        Detecta si hay una señal alineada en múltiples marcos y lanza alerta con TP/SL institucional.
        """
        b15 = evals.get("15m", {}).get("bias")
        b1h = evals.get("1h", {}).get("bias")
        b4h = evals.get("4h", {}).get("bias")
        b1d = evals.get("1d", {}).get("bias")

        # 🚦 Condición núcleo: hay gatillo real en 15m
        if b15 in ("Long", "Short"):
            direction = b15  # dirección operativa

            # Confirmaciones jerárquicas
            if b1h in (direction, "Neutral") and \
               b4h in (direction, "Neutral") and \
               not (b1d and b1d != direction and b1d != "Neutral"):

                # ✅ ALERTA ACTIVADA
                eval_entry = evals.get("15m", {})  # Usamos 15m para punto de entrada
                sl, tp, rr = None, None, None

                # 🔧 Calcular SL/TP basado en SR/RS
                price = eval_entry.get("price")
                sr = eval_entry.get("sr_level")
                rs = eval_entry.get("rs_level")

                if direction == "Long":
                    sl = sr if not np.isnan(sr) else price * 0.99
                    tp = rs if not np.isnan(rs) else price * 1.02
                elif direction == "Short":
                    sl = rs if not np.isnan(rs) else price * 1.01
                    tp = sr if not np.isnan(sr) else price * 0.98

                rr = abs((tp - price) / (price - sl)) if sl and tp else None

                # 💬 Mostrar alerta avanzada
                self.console.print(f"\n[bold red]🚨 ALERTA MTF ACTIVADA → {pair}[/bold red]")
                self.console.print(f"[cyan]Dirección:[/cyan] {direction.upper()}  |  Precio actual: {price:.2f}")
                self.console.print(f"[yellow]→ SL estructural:[/yellow] {sl:.2f}")
                self.console.print(f"[green]→ TP institucional:[/green] {tp:.2f}")
                if rr:
                    self.console.print(f"[magenta]→ RIESGO/BENEFICIO (R/R):[/magenta] {rr:.2f}R")
                self.console.print(f"[dim]Estructura SR/RS evaluada desde análisis MTF...[/dim]\n")

                return {"direction": direction, "price": price, "sl": sl, "tp": tp, "rr": rr}

        return False

    def generate_structured_tp_sl(eval_entry: dict) -> dict:
        """
        Genera SL y TP institucional basado en SR/RS estructural,
        sin usar ATR como base primaria.
        """
        direction = eval_entry.get("bias")
        price = eval_entry.get("price")
        sr = eval_entry.get("sr_level")
        rs = eval_entry.get("rs_level")

        if direction == "Long":
            sl = sr if not np.isnan(sr) else price * 0.99  # fallback si aún no hay SR
            tp = rs if not np.isnan(rs) else price * 1.02  # fallback provisional
        elif direction == "Short":
            sl = rs if not np.isnan(rs) else price * 1.01
            tp = sr if not np.isnan(sr) else price * 0.98
        else:
            return {"sl": None, "tp": None, "note": "No hay gatillo operativo"}

        return {
            "sl": sl,
            "tp": tp,
            "rr_ratio": abs((tp - price) / (price - sl)) if sl and tp else None
        }



    def render_tactical_panel(self, pair: str, tf_evals_map: dict):
        """
        Panel táctico limpio sin sufijos.
        Usa: sr_level, rs_level, mid_level, candles_back
        """
        # ---------- Helpers ----------
        def _is_nan(x):
            try:
                return x is None or (isinstance(x, float) and math.isnan(x))
            except Exception:
                return False

        def _fmt_num(x, dec=2):
            # Si es string (por ejemplo "[dim]101200[/dim]") → devolver directo
            if isinstance(x, str):
                return x
            # Si es NaN → mostrar gris
            try:
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    return "[dim]—[/dim]"
            except:
                pass
            # Si es número → formatear normal
            try:
                return f"{float(x):.{dec}f}"
            except:
                # fallback si algo raro entra
                return str(x)
            
        def _fmt_pct(x):
            if _is_nan(x):
                return "[dim]—[/dim]"
            return f"{x:+.2f}%"

        def _fmt_bias(bias: str):
            if bias == "Long":
                return "[green]🔺 LONG[/green]"
            if bias == "Short":
                return "[red]🔻 SHORT[/red]"
            return "[yellow]🔶 NEUTRAL[/yellow]"

        def _fmt_ema(price, ema):
            if _is_nan(price) or _is_nan(ema):
                return "[dim]—[/dim]"
            return "[green]>EMA[/green]" if price > ema else "[red]<EMA[/red]"

        def _fmt_rsi(rsi):
            if _is_nan(rsi):
                return "[dim]—[/dim]"
            arrow = "↑" if rsi >= 50 else "↓"
            color = "green" if rsi >= 50 else "red"
            return f"[{color}]{rsi:.1f} {arrow}[/{color}]"

        # Orden visual de TF si existen
        tf_order = {"15m": 0, "1h": 1, "4h": 2, "1d": 3, "1w": 4}
        ordered_items = sorted(
            tf_evals_map.items(),
            key=lambda kv: tf_order.get(kv[0], 999)
        )
        

        # ---------- Limpiar pantalla ----------
        os.system("cls" if os.name == "nt" else "clear")

        # ---------- Construir tabla ----------
        table = Table(title=f"📊 VISIÓN MULTIMARCO - {pair}", header_style="bold cyan")

        table.add_column("PAR", justify="center", no_wrap=True)
        table.add_column("TF", justify="center", no_wrap=True)
        table.add_column("ROL", justify="left", no_wrap=True)
        table.add_column("Bias", justify="left", no_wrap=True)
        table.add_column("Precio", justify="right", no_wrap=True)
        table.add_column("EMA", justify="center", no_wrap=True)
        table.add_column("RSI", justify="center", no_wrap=True)
        table.add_column("SR", justify="right", no_wrap=True)
        table.add_column("RS", justify="right", no_wrap=True)
        table.add_column("Dist SR", justify="center", no_wrap=True)
        table.add_column("Dist RS", justify="center", no_wrap=True)
        table.add_column("MID", justify="right", no_wrap=True)
        table.add_column("Velas", justify="center", no_wrap=True)
        table.add_column("Nota", justify="left")

        for tf_key, ev in ordered_items:
            price = ev.get("price", float("nan"))
            ema   = ev.get("ema", float("nan"))
            rsi   = ev.get("rsi", float("nan"))

            sr = ev.get("sr_level", float("nan"))
            sr_macro = ev.get("sr_macro_level", np.nan)
            sr_display = _fmt_num(sr) if not _is_nan(sr) else f"[dim]{_fmt_num(sr_macro)}[/dim]"

            rs = ev.get("rs_level", float("nan"))
            mid = ev.get("mid_level", float("nan"))
            candles_back = ev.get("candles_back", float("nan"))

            # Distancias % relativas al precio
            dist_sr = ((price - sr) / price * 100) if (not _is_nan(price) and not _is_nan(sr) and price != 0) else float("nan")
            dist_rs = ((rs - price) / price * 100) if (not _is_nan(price) and not _is_nan(rs) and price != 0) else float("nan")

            # Nota: si no hay SR ni RS, mostramos mensaje especial
            no_levels = _is_nan(sr) and _is_nan(rs)
            note = "🕒 Aún sin niveles" if no_levels else ev.get("note", "")
            table.add_row(
                pair,
                tf_key,
                self._assign_role(tf_key),   # 👈 AÑADIDO AQUÍ
                _fmt_bias(ev.get("bias", "Neutral")),
                _fmt_num(price, 2),
                _fmt_ema(price, ema),
                _fmt_rsi(rsi),
                _fmt_num(sr_display, 2),
                _fmt_num(rs, 2),
                _fmt_pct(dist_sr),
                _fmt_pct(dist_rs),
                _fmt_num(mid, 2),
                _fmt_num(candles_back, 0),
                note
            )


        self.console.print(table)
       
       
    def render_multi_pair_panel(self, multi_tf_evals: dict):
        """
        Renderiza una tabla agrupando TODOS los pares.
        Estructura esperada:
        {
        "BTC/USDC": {"15m": eval, "1h": eval, "4h": eval, "1d": eval},
        "ETH/USDC": { ... }
        }
        """
        table = Table(title="📊 VISIÓN MULTIPAR - MTF", header_style="bold cyan")

        # Columnas globales
        table.add_column("PAR", justify="center", no_wrap=True)
        table.add_column("TF", justify="center", no_wrap=True)
        table.add_column("ROL", justify="center", no_wrap=True)
        table.add_column("Bias", justify="center", no_wrap=True)
        table.add_column("Precio", justify="right", no_wrap=True)
        table.add_column("EMA", justify="center", no_wrap=True)
        table.add_column("RSI", justify="center", no_wrap=True)
        table.add_column("SR", justify="right", no_wrap=True)
        table.add_column("RS", justify="right", no_wrap=True)
        table.add_column("Dist SR", justify="center", no_wrap=True)
        table.add_column("Dist RS", justify="center", no_wrap=True)
        table.add_column("MID", justify="right", no_wrap=True)
        table.add_column("Velas", justify="center", no_wrap=True)
        table.add_column("Nota", justify="left")

        ROLE_MAP = {
            "15m": "🟢 GATILLO",
            "1h": "✅ CONFIRMA",
            "4h": "🌊 TENDENCIA",
            "1d": "🧭 MACRO"
        }

        def _is_nan(x):
            try:
                return x is None or (isinstance(x, float) and math.isnan(x))
            except Exception:
                return False

        def _fmt_num(x, dec=2):
            if _is_nan(x):
                return "[dim]—[/dim]"
            try:
                return f"{float(x):.{dec}f}"
            except Exception:
                return "[dim]—[/dim]"

        def _fmt_pct(x):
            if _is_nan(x):
                return "[dim]—[/dim]"
            return f"{x:+.2f}%"

        def _fmt_bias(bias: str):
            if bias == "Long":
                return "[green]🔺 LONG[/green]"
            if bias == "Short":
                return "[red]🔻 SHORT[/red]"
            return "[yellow]🔶 NEUTRAL[/yellow]"

        def _fmt_ema(price, ema):
            if _is_nan(price) or _is_nan(ema):
                return "[dim]—[/dim]"
            return "[green]>EMA[/green]" if price > ema else "[red]<EMA[/red]"

        def _fmt_rsi(rsi):
            if _is_nan(rsi):
                return "[dim]—[/dim]"
            arrow = "↑" if rsi >= 50 else "↓"
            color = "green" if rsi >= 50 else "red"
            return f"[{color}]{rsi:.1f} {arrow}[/{color}]"

        # Recorrer pares y TF ordenados
        for pair, eval_map in multi_tf_evals.items():
            for tf in ["15m", "1h", "4h", "1d"]:
                if tf not in eval_map:
                    continue
                ev = eval_map[tf]
                price = ev.get("price", float("nan"))
                ema   = ev.get("ema", float("nan"))
                rsi   = ev.get("rsi", float("nan"))
                sr = ev.get("sr_level", float("nan"))
                rs = ev.get("rs_level", float("nan"))
                mid = ev.get("mid_level", float("nan"))
                candles_back = ev.get("candles_back", float("nan"))

                dist_sr = ((price - sr) / price * 100) if (not _is_nan(price) and not _is_nan(sr) and price != 0) else float("nan")
                dist_rs = ((rs - price) / price * 100) if (not _is_nan(price) and not _is_nan(rs) and price != 0) else float("nan")

                note = "🕒 Aún sin niveles" if (_is_nan(sr) and _is_nan(rs)) else ev.get("note", "")

                table.add_row(
                    pair,
                    tf,
                    ROLE_MAP.get(tf, "—"),
                    _fmt_bias(ev.get("bias", "Neutral")),
                    _fmt_num(price, 2),
                    _fmt_ema(price, ema),
                    _fmt_rsi(rsi),
                    _fmt_num(sr, 2),
                    _fmt_num(rs, 2),
                    _fmt_pct(dist_sr),
                    _fmt_pct(dist_rs),
                    _fmt_num(mid, 2),
                    _fmt_num(candles_back, 0),
                    note
                )
            table.add_section()

        self.console.print(table)   

