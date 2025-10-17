from datetime import datetime, timezone

# === pares en array para añadir/quitar fácilmente ===
PAIRS = [
    "BTC/USDC",
    "ETH/USDC",

]

# timeframe base de descarga
BASE_TIMEFRAME = "5m"

# resamples adicionales a generar (OHLCV)
RESAMPLES = ["15m", "1h", "4h", "1D", "1W", "1M"]

# fecha inicial (UTC) de descarga
SINCE_UTC = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

# rutas de salida
DATA_DIR = "binatrend/data"                   # raíz para .feather
META_DIR = "binatrend/data/_meta"             # metadatos por par/timeframe

# exchange
EXCHANGE_ID = "binance"
MARKET_TYPE = "spot"                # spot por tu requerimiento
