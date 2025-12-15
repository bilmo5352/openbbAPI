"""
Analysis service that wraps the existing logic for fetching data and computing indicators.

Functions here are imported by the FastAPI interface so the API file stays minimal.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import inspect
import math
import numpy as np
import pandas as pd

# Reuse the existing logic from main.py
from main import (
    obb,
    obb_technical_available,
    ta,
    talib,
    talib_available,
    compute_indicators,
    fetch_historical_zerodha,
    fetch_historical_yfinance,
    fetch_profile_openbb,
    fetch_profile_yfinance,
    kite,
    openbb_available,
)

DEFAULT_LOOKBACK_DAYS = 90

# Indicator catalog: name -> config
# Supports indicators from: manual, OpenBB, TA-Lib, and pandas-ta
# Priority: OpenBB > TA-Lib > pandas-ta > manual
INDICATOR_CATALOG: Dict[str, Dict[str, Any]] = {
    # ===== MANUAL IMPLEMENTATIONS (Always Available) =====
    "sma": {"kind": "manual", "fn": "sma", "params": {"length": 20}, "min_bars": 20},
    "ema": {"kind": "manual", "fn": "ema", "params": {"length": 20}, "min_bars": 20},
    "rsi": {"kind": "manual", "fn": "rsi", "params": {"length": 14}, "min_bars": 14},
    "bbands": {"kind": "manual", "fn": "bbands", "params": {"length": 20, "std": 2.0}, "min_bars": 20},
    "macd": {"kind": "manual", "fn": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "min_bars": 26},
    "atr": {"kind": "manual", "fn": "atr", "params": {"length": 14}, "min_bars": 14},
    "vwap": {"kind": "manual", "fn": "vwap", "params": {}, "min_bars": 1},
    "ichimoku": {"kind": "manual", "fn": "ichimoku", "params": {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26}, "min_bars": 52},
    
    # ===== OPENBB TECHNICAL EXTENSION INDICATORS =====
    "adx": {"kind": "obb", "fn": "adx", "params": {"length": 14}, "min_bars": 14},
    "obv": {"kind": "obb", "fn": "obv", "params": {}, "min_bars": 2},
    "kc": {"kind": "obb", "fn": "kc", "params": {"length": 20}, "min_bars": 20},
    "hma": {"kind": "obb", "fn": "hma", "params": {"length": 20}, "min_bars": 20},
    "wma": {"kind": "obb", "fn": "wma", "params": {"length": 20}, "min_bars": 20},
    "fib": {"kind": "obb", "fn": "fib", "params": {}, "min_bars": 1},
    "demark": {"kind": "obb", "fn": "demark", "params": {}, "min_bars": 1},
    "relative_rotation": {"kind": "obb", "fn": "relative_rotation", "params": {}, "min_bars": 1},
    "cg": {"kind": "obb", "fn": "cg", "params": {"length": 10}, "min_bars": 10},
    "clenow": {"kind": "obb", "fn": "clenow", "params": {"lookback": 20}, "min_bars": 20},
    "aroon": {"kind": "obb", "fn": "aroon", "params": {"length": 14}, "min_bars": 14},
    "fisher": {"kind": "obb", "fn": "fisher", "params": {"length": 9}, "min_bars": 9},
    "cci": {"kind": "obb", "fn": "cci", "params": {"length": 20}, "min_bars": 20},
    "donchian": {"kind": "obb", "fn": "donchian", "params": {"length": 20}, "min_bars": 20},
    "stoch": {"kind": "obb", "fn": "stoch", "params": {"fastk": 14, "fastd": 3}, "min_bars": 14},
    "adosc": {"kind": "obb", "fn": "adosc", "params": {}, "min_bars": 2},
    "ad": {"kind": "obb", "fn": "ad", "params": {}, "min_bars": 2},
    "cones": {"kind": "obb", "fn": "cones", "params": {}, "min_bars": 2},
    "zlma": {"kind": "obb", "fn": "zlma", "params": {"length": 20}, "min_bars": 20},
    
    # ===== TA-LIB INDICATORS (Overlap Studies) =====
    "dema": {"kind": "talib", "fn": "DEMA", "params": {"timeperiod": 30}, "min_bars": 30},
    "tema": {"kind": "talib", "fn": "TEMA", "params": {"timeperiod": 30}, "min_bars": 30},
    "trima": {"kind": "talib", "fn": "TRIMA", "params": {"timeperiod": 30}, "min_bars": 30},
    "kama": {"kind": "talib", "fn": "KAMA", "params": {"timeperiod": 30}, "min_bars": 30},
    "mama": {"kind": "talib", "fn": "MAMA", "params": {"fastlimit": 0.5, "slowlimit": 0.05}, "min_bars": 10},
    "t3": {"kind": "talib", "fn": "T3", "params": {"timeperiod": 5, "vfactor": 0.7}, "min_bars": 5},
    "typprice": {"kind": "talib", "fn": "TYPPRICE", "params": {}, "min_bars": 1},
    "wclprice": {"kind": "talib", "fn": "WCLPRICE", "params": {}, "min_bars": 1},
    "avgprice": {"kind": "talib", "fn": "AVGPRICE", "params": {}, "min_bars": 1},
    "medprice": {"kind": "talib", "fn": "MEDPRICE", "params": {}, "min_bars": 1},
    
    # ===== TA-LIB INDICATORS (Momentum) =====
    "adxr": {"kind": "talib", "fn": "ADXR", "params": {"timeperiod": 14}, "min_bars": 14},
    "apo": {"kind": "talib", "fn": "APO", "params": {"fastperiod": 12, "slowperiod": 26}, "min_bars": 26},
    "aroonosc": {"kind": "talib", "fn": "AROONOSC", "params": {"timeperiod": 14}, "min_bars": 14},
    "bop": {"kind": "talib", "fn": "BOP", "params": {}, "min_bars": 1},
    "cmo": {"kind": "talib", "fn": "CMO", "params": {"timeperiod": 14}, "min_bars": 14},
    "dx": {"kind": "talib", "fn": "DX", "params": {"timeperiod": 14}, "min_bars": 14},
    "mfi": {"kind": "talib", "fn": "MFI", "params": {"timeperiod": 14}, "min_bars": 14},
    "minus_di": {"kind": "talib", "fn": "MINUS_DI", "params": {"timeperiod": 14}, "min_bars": 14},
    "plus_di": {"kind": "talib", "fn": "PLUS_DI", "params": {"timeperiod": 14}, "min_bars": 14},
    "minus_dm": {"kind": "talib", "fn": "MINUS_DM", "params": {"timeperiod": 14}, "min_bars": 14},
    "plus_dm": {"kind": "talib", "fn": "PLUS_DM", "params": {"timeperiod": 14}, "min_bars": 14},
    "mom": {"kind": "talib", "fn": "MOM", "params": {"timeperiod": 10}, "min_bars": 10},
    "ppo": {"kind": "talib", "fn": "PPO", "params": {"fastperiod": 12, "slowperiod": 26}, "min_bars": 26},
    "roc": {"kind": "talib", "fn": "ROC", "params": {"timeperiod": 10}, "min_bars": 10},
    "rocp": {"kind": "talib", "fn": "ROCP", "params": {"timeperiod": 10}, "min_bars": 10},
    "rocr": {"kind": "talib", "fn": "ROCR", "params": {"timeperiod": 10}, "min_bars": 10},
    "rocr100": {"kind": "talib", "fn": "ROCR100", "params": {"timeperiod": 10}, "min_bars": 10},
    "stochf": {"kind": "talib", "fn": "STOCHF", "params": {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0}, "min_bars": 5},
    "stochrsi": {"kind": "talib", "fn": "STOCHRSI", "params": {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3, "fastd_matype": 0}, "min_bars": 14},
    "trix": {"kind": "talib", "fn": "TRIX", "params": {"timeperiod": 30}, "min_bars": 30},
    "ultosc": {"kind": "talib", "fn": "ULTOSC", "params": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28}, "min_bars": 28},
    "willr": {"kind": "talib", "fn": "WILLR", "params": {"timeperiod": 14}, "min_bars": 14},
    
    # ===== TA-LIB INDICATORS (Volume) =====
    # AD, ADOSC, OBV already covered above
    
    # ===== TA-LIB INDICATORS (Volatility) =====
    "natr": {"kind": "talib", "fn": "NATR", "params": {"timeperiod": 14}, "min_bars": 14},
    "trange": {"kind": "talib", "fn": "TRANGE", "params": {}, "min_bars": 1},
    
    # ===== TA-LIB INDICATORS (Pattern Recognition - Sample) =====
    "cdl2crows": {"kind": "talib", "fn": "CDL2CROWS", "params": {}, "min_bars": 3},
    "cdl3blackcrows": {"kind": "talib", "fn": "CDL3BLACKCROWS", "params": {}, "min_bars": 3},
    "cdl3inside": {"kind": "talib", "fn": "CDL3INSIDE", "params": {}, "min_bars": 3},
    "cdl3linestrike": {"kind": "talib", "fn": "CDL3LINESTRIKE", "params": {}, "min_bars": 4},
    "cdl3outside": {"kind": "talib", "fn": "CDL3OUTSIDE", "params": {}, "min_bars": 3},
    "cdl3whitesoldiers": {"kind": "talib", "fn": "CDL3WHITESOLDIERS", "params": {}, "min_bars": 3},
    "cdlabandonedbaby": {"kind": "talib", "fn": "CDLABANDONEDBABY", "params": {}, "min_bars": 3},
    "cdladvanceblock": {"kind": "talib", "fn": "CDLADVANCEBLOCK", "params": {}, "min_bars": 3},
    "cdlbelthold": {"kind": "talib", "fn": "CDLBELTHOLD", "params": {}, "min_bars": 1},
    "cdlbreakaway": {"kind": "talib", "fn": "CDLBREAKAWAY", "params": {}, "min_bars": 5},
    "cdlclosingmarubozu": {"kind": "talib", "fn": "CDLCLOSINGMARUBOZU", "params": {}, "min_bars": 1},
    "cdlconcealbabyswall": {"kind": "talib", "fn": "CDLCONCEALBABYSWALL", "params": {}, "min_bars": 5},
    "cdlcounterattack": {"kind": "talib", "fn": "CDLCOUNTERATTACK", "params": {}, "min_bars": 2},
    "cdldarkcloudcover": {"kind": "talib", "fn": "CDLDARKCLOUDCOVER", "params": {}, "min_bars": 2},
    "cdldoji": {"kind": "talib", "fn": "CDLDOJI", "params": {}, "min_bars": 1},
    "cdldojistar": {"kind": "talib", "fn": "CDLDOJISTAR", "params": {}, "min_bars": 1},
    "cdldragonflydoji": {"kind": "talib", "fn": "CDLDRAGONFLYDOJI", "params": {}, "min_bars": 1},
    "cdlengulfing": {"kind": "talib", "fn": "CDLENGULFING", "params": {}, "min_bars": 2},
    "cdleveningdojistar": {"kind": "talib", "fn": "CDLEVENINGDOJISTAR", "params": {}, "min_bars": 3},
    "cdleveningstar": {"kind": "talib", "fn": "CDLEVENINGSTAR", "params": {}, "min_bars": 3},
    "cdlgapsidesidewhite": {"kind": "talib", "fn": "CDLGAPSIDESIDEWHITE", "params": {}, "min_bars": 3},
    "cdlgravestonedoji": {"kind": "talib", "fn": "CDLGRAVESTONEDOJI", "params": {}, "min_bars": 1},
    "cdlhammer": {"kind": "talib", "fn": "CDLHAMMER", "params": {}, "min_bars": 1},
    "cdlhangingman": {"kind": "talib", "fn": "CDLHANGINGMAN", "params": {}, "min_bars": 1},
    "cdlharami": {"kind": "talib", "fn": "CDLHARAMI", "params": {}, "min_bars": 2},
    "cdlharamicross": {"kind": "talib", "fn": "CDLHARAMICROSS", "params": {}, "min_bars": 2},
    "cdlhighwave": {"kind": "talib", "fn": "CDLHIGHWAVE", "params": {}, "min_bars": 5},
    "cdlhikkake": {"kind": "talib", "fn": "CDLHIKKAKE", "params": {}, "min_bars": 3},
    "cdlhikkakemod": {"kind": "talib", "fn": "CDLHIKKAKEMOD", "params": {}, "min_bars": 5},
    "cdlhomingpigeon": {"kind": "talib", "fn": "CDLHOMINGPIGEON", "params": {}, "min_bars": 2},
    "cdlidentical3crows": {"kind": "talib", "fn": "CDLIDENTICAL3CROWS", "params": {}, "min_bars": 3},
    "cdlinneck": {"kind": "talib", "fn": "CDLINNECK", "params": {}, "min_bars": 3},
    "cdlinvertedhammer": {"kind": "talib", "fn": "CDLINVERTEDHAMMER", "params": {}, "min_bars": 1},
    "cdlkicking": {"kind": "talib", "fn": "CDLKICKING", "params": {}, "min_bars": 2},
    "cdlkickingbylength": {"kind": "talib", "fn": "CDLKICKINGBYLENGTH", "params": {}, "min_bars": 2},
    "cdlladderbottom": {"kind": "talib", "fn": "CDLLADDERBOTTOM", "params": {}, "min_bars": 5},
    "cdllongleggeddoji": {"kind": "talib", "fn": "CDLLONGLEGGEDDOJI", "params": {}, "min_bars": 1},
    "cdllongline": {"kind": "talib", "fn": "CDLLONGLINE", "params": {}, "min_bars": 1},
    "cdlmarubozu": {"kind": "talib", "fn": "CDLMARUBOZU", "params": {}, "min_bars": 1},
    "cdlmatchinglow": {"kind": "talib", "fn": "CDLMATCHINGLOW", "params": {}, "min_bars": 2},
    "cdlmathold": {"kind": "talib", "fn": "CDLMATHOLD", "params": {}, "min_bars": 5},
    "cdlmorningdojistar": {"kind": "talib", "fn": "CDLMORNINGDOJISTAR", "params": {}, "min_bars": 3},
    "cdlmorningstar": {"kind": "talib", "fn": "CDLMORNINGSTAR", "params": {}, "min_bars": 3},
    "cdlonneck": {"kind": "talib", "fn": "CDLONNECK", "params": {}, "min_bars": 2},
    "cdlpiercing": {"kind": "talib", "fn": "CDLPIERCING", "params": {}, "min_bars": 2},
    "cdlrickshawman": {"kind": "talib", "fn": "CDLRICKSHAWMAN", "params": {}, "min_bars": 1},
    "cdlrisefall3methods": {"kind": "talib", "fn": "CDLRISEFALL3METHODS", "params": {}, "min_bars": 5},
    "cdlseparatinglines": {"kind": "talib", "fn": "CDLSEPARATINGLINES", "params": {}, "min_bars": 2},
    "cdlshootingstar": {"kind": "talib", "fn": "CDLSHOOTINGSTAR", "params": {}, "min_bars": 1},
    "cdlshortline": {"kind": "talib", "fn": "CDLSHORTLINE", "params": {}, "min_bars": 1},
    "cdlspinningtop": {"kind": "talib", "fn": "CDLSPINNINGTOP", "params": {}, "min_bars": 1},
    "cdlstalledpattern": {"kind": "talib", "fn": "CDLSTALLEDPATTERN", "params": {}, "min_bars": 4},
    "cdlsticksandwich": {"kind": "talib", "fn": "CDLSTICKSANDWICH", "params": {}, "min_bars": 3},
    "cdltakuri": {"kind": "talib", "fn": "CDLTAKURI", "params": {}, "min_bars": 1},
    "cdltasukigage": {"kind": "talib", "fn": "CDLTASUKIGAGE", "params": {}, "min_bars": 5},
    "cdlthrusting": {"kind": "talib", "fn": "CDLTHRUSTING", "params": {}, "min_bars": 2},
    "cdltristar": {"kind": "talib", "fn": "CDLTRISTAR", "params": {}, "min_bars": 3},
    "cdlunique3river": {"kind": "talib", "fn": "CDLUNIQUE3RIVER", "params": {}, "min_bars": 3},
    "cdlupsidegap2crows": {"kind": "talib", "fn": "CDLUPSIDEGAP2CROWS", "params": {}, "min_bars": 3},
    "cdlxsidegap3methods": {"kind": "talib", "fn": "CDLXSIDEGAP3METHODS", "params": {}, "min_bars": 5},
    
    # ===== PANDAS-TA INDICATORS (Momentum) =====
    "ao": {"kind": "pandas_ta", "fn": "ao", "params": {"fast": 5, "slow": 34}, "min_bars": 34},
    "apo": {"kind": "pandas_ta", "fn": "apo", "params": {"fast": 12, "slow": 26}, "min_bars": 26},
    "bias": {"kind": "pandas_ta", "fn": "bias", "params": {"length": 26}, "min_bars": 26},
    "bop": {"kind": "pandas_ta", "fn": "bop", "params": {}, "min_bars": 1},
    "brar": {"kind": "pandas_ta", "fn": "brar", "params": {"length": 26}, "min_bars": 26},
    "cci": {"kind": "pandas_ta", "fn": "cci", "params": {"length": 20}, "min_bars": 20},
    "cfo": {"kind": "pandas_ta", "fn": "cfo", "params": {"length": 14}, "min_bars": 14},
    "cg": {"kind": "pandas_ta", "fn": "cg", "params": {"length": 10}, "min_bars": 10},
    "cmo": {"kind": "pandas_ta", "fn": "cmo", "params": {"length": 14}, "min_bars": 14},
    "coppock": {"kind": "pandas_ta", "fn": "coppock", "params": {"length": 10, "fast": 11, "slow": 14}, "min_bars": 14},
    "er": {"kind": "pandas_ta", "fn": "er", "params": {"length": 10}, "min_bars": 10},
    "eri": {"kind": "pandas_ta", "fn": "eri", "params": {"length": 13}, "min_bars": 13},
    "fisher": {"kind": "pandas_ta", "fn": "fisher", "params": {"length": 10}, "min_bars": 10},
    "inertia": {"kind": "pandas_ta", "fn": "inertia", "params": {"length": 14, "rvi_length": 14}, "min_bars": 14},
    "kdj": {"kind": "pandas_ta", "fn": "kdj", "params": {"length": 9, "signal": 3}, "min_bars": 9},
    "kst": {"kind": "pandas_ta", "fn": "kst", "params": {"roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30, "sma1": 10, "sma2": 10, "sma3": 10, "sma4": 15}, "min_bars": 30},
    "mom": {"kind": "pandas_ta", "fn": "mom", "params": {"length": 10}, "min_bars": 10},
    "pgo": {"kind": "pandas_ta", "fn": "pgo", "params": {"length": 14}, "min_bars": 14},
    "ppo": {"kind": "pandas_ta", "fn": "ppo", "params": {"fast": 12, "slow": 26}, "min_bars": 26},
    "psl": {"kind": "pandas_ta", "fn": "psl", "params": {"length": 12}, "min_bars": 12},
    "pvo": {"kind": "pandas_ta", "fn": "pvo", "params": {"fast": 12, "slow": 26}, "min_bars": 26},
    "qqe": {"kind": "pandas_ta", "fn": "qqe", "params": {"length": 14, "smooth": 5}, "min_bars": 14},
    "roc": {"kind": "pandas_ta", "fn": "roc", "params": {"length": 10}, "min_bars": 10},
    "rsi": {"kind": "pandas_ta", "fn": "rsi", "params": {"length": 14}, "min_bars": 14},
    "rsx": {"kind": "pandas_ta", "fn": "rsx", "params": {"length": 14}, "min_bars": 14},
    "rvgi": {"kind": "pandas_ta", "fn": "rvgi", "params": {"length": 14}, "min_bars": 14},
    "slope": {"kind": "pandas_ta", "fn": "slope", "params": {"length": 5}, "min_bars": 5},
    "smi": {"kind": "pandas_ta", "fn": "smi", "params": {"fast": 13, "slow": 25}, "min_bars": 25},
    "squeeze": {"kind": "pandas_ta", "fn": "squeeze", "params": {"bb_length": 20, "bb_std": 2, "kc_length": 20, "kc_scalar": 1.5}, "min_bars": 20},
    "squeeze_pro": {"kind": "pandas_ta", "fn": "squeeze_pro", "params": {"bb_length": 20, "bb_std": 2, "kc_length": 20, "kc_scalar": 1.5}, "min_bars": 20},
    "stoch": {"kind": "pandas_ta", "fn": "stoch", "params": {"k": 14, "d": 3, "smooth_k": 1}, "min_bars": 14},
    "stochrsi": {"kind": "pandas_ta", "fn": "stochrsi", "params": {"length": 14, "rsi_length": 14, "k": 3, "d": 3}, "min_bars": 14},
    "td_seq": {"kind": "pandas_ta", "fn": "td_seq", "params": {"asint": True}, "min_bars": 4},
    "tsi": {"kind": "pandas_ta", "fn": "tsi", "params": {"fast": 13, "slow": 25}, "min_bars": 25},
    "uo": {"kind": "pandas_ta", "fn": "uo", "params": {"fast": 7, "medium": 14, "slow": 28}, "min_bars": 28},
    "willr": {"kind": "pandas_ta", "fn": "willr", "params": {"length": 14}, "min_bars": 14},
    
    # ===== PANDAS-TA INDICATORS (Trend) =====
    "alligator": {
        "kind": "pandas_ta",
        "fn": "alligator",
        "params": {
            "length_jaw": 13,
            "length_teeth": 8,
            "length_lips": 5,
        },
        "min_bars": 30,
    },
    "adx": {"kind": "pandas_ta", "fn": "adx", "params": {"length": 14}, "min_bars": 14},
    "amat": {"kind": "pandas_ta", "fn": "amat", "params": {"fast": 8, "slow": 21}, "min_bars": 21},
    "aroon": {"kind": "pandas_ta", "fn": "aroon", "params": {"length": 14}, "min_bars": 14},
    "chop": {"kind": "pandas_ta", "fn": "chop", "params": {"length": 14}, "min_bars": 14},
    "cksp": {"kind": "pandas_ta", "fn": "cksp", "params": {"p": 2, "x": 8}, "min_bars": 8},
    "decay": {"kind": "pandas_ta", "fn": "decay", "params": {"length": 5}, "min_bars": 5},
    "decreasing": {"kind": "pandas_ta", "fn": "decreasing", "params": {"length": 2}, "min_bars": 2},
    "dpo": {"kind": "pandas_ta", "fn": "dpo", "params": {"length": 20}, "min_bars": 20},
    "increasing": {"kind": "pandas_ta", "fn": "increasing", "params": {"length": 2}, "min_bars": 2},
    "long_run": {"kind": "pandas_ta", "fn": "long_run", "params": {"length": 2}, "min_bars": 2},
    "psar": {"kind": "pandas_ta", "fn": "psar", "params": {"af": 0.02, "max_af": 0.2}, "min_bars": 2},
    "qstick": {"kind": "pandas_ta", "fn": "qstick", "params": {"length": 10}, "min_bars": 10},
    "short_run": {"kind": "pandas_ta", "fn": "short_run", "params": {"length": 2}, "min_bars": 2},
    "supertrend": {"kind": "pandas_ta", "fn": "supertrend", "params": {"length": 7, "multiplier": 3.0}, "min_bars": 7},
    "ttm_trend": {"kind": "pandas_ta", "fn": "ttm_trend", "params": {"length": 5}, "min_bars": 5},
    "vortex": {"kind": "pandas_ta", "fn": "vortex", "params": {"length": 14}, "min_bars": 14},
    
    # ===== PANDAS-TA INDICATORS (Volatility) =====
    "aberration": {"kind": "pandas_ta", "fn": "aberration", "params": {"length": 5}, "min_bars": 5},
    "accbands": {"kind": "pandas_ta", "fn": "accbands", "params": {"length": 20}, "min_bars": 20},
    "atr": {"kind": "pandas_ta", "fn": "atr", "params": {"length": 14}, "min_bars": 14},
    "bbands": {"kind": "pandas_ta", "fn": "bbands", "params": {"length": 20, "std": 2}, "min_bars": 20},
    "donchian": {"kind": "pandas_ta", "fn": "donchian", "params": {"lower_length": 20, "upper_length": 20}, "min_bars": 20},
    "kc": {"kind": "pandas_ta", "fn": "kc", "params": {"length": 20, "scalar": 2}, "min_bars": 20},
    "massi": {"kind": "pandas_ta", "fn": "massi", "params": {"fast": 9, "slow": 25}, "min_bars": 25},
    "natr": {"kind": "pandas_ta", "fn": "natr", "params": {"length": 14}, "min_bars": 14},
    "pdist": {"kind": "pandas_ta", "fn": "pdist", "params": {}, "min_bars": 1},
    "rvi": {"kind": "pandas_ta", "fn": "rvi", "params": {"length": 14}, "min_bars": 14},
    "thermo": {"kind": "pandas_ta", "fn": "thermo", "params": {"length": 20, "mamode": "ema"}, "min_bars": 20},
    "true_range": {"kind": "pandas_ta", "fn": "true_range", "params": {}, "min_bars": 1},
    "ui": {"kind": "pandas_ta", "fn": "ui", "params": {"length": 14}, "min_bars": 14},
    
    # ===== PANDAS-TA INDICATORS (Volume) =====
    "ad": {"kind": "pandas_ta", "fn": "ad", "params": {}, "min_bars": 2},
    "adosc": {"kind": "pandas_ta", "fn": "adosc", "params": {"fast": 3, "slow": 10}, "min_bars": 10},
    "aobv": {"kind": "pandas_ta", "fn": "aobv", "params": {"fast": 4, "slow": 12}, "min_bars": 12},
    "cmf": {"kind": "pandas_ta", "fn": "cmf", "params": {"length": 20}, "min_bars": 20},
    "efi": {"kind": "pandas_ta", "fn": "efi", "params": {"length": 13}, "min_bars": 13},
    "eom": {"kind": "pandas_ta", "fn": "eom", "params": {"length": 14, "divisor": 100000000}, "min_bars": 14},
    "fwma": {"kind": "pandas_ta", "fn": "fwma", "params": {"length": 10}, "min_bars": 10},
    "kvo": {"kind": "pandas_ta", "fn": "kvo", "params": {"fast": 2, "slow": 5}, "min_bars": 5},
    "mfi": {"kind": "pandas_ta", "fn": "mfi", "params": {"length": 14}, "min_bars": 14},
    "nvi": {"kind": "pandas_ta", "fn": "nvi", "params": {"length": 255}, "min_bars": 255},
    "obv": {"kind": "pandas_ta", "fn": "obv", "params": {}, "min_bars": 2},
    "pvi": {"kind": "pandas_ta", "fn": "pvi", "params": {"length": 255}, "min_bars": 255},
    "pvol": {"kind": "pandas_ta", "fn": "pvol", "params": {}, "min_bars": 1},
    "pvr": {"kind": "pandas_ta", "fn": "pvr", "params": {}, "min_bars": 1},
    "pvt": {"kind": "pandas_ta", "fn": "pvt", "params": {}, "min_bars": 2},
    "vwap": {"kind": "pandas_ta", "fn": "vwap", "params": {"anchor": None}, "min_bars": 1},
    "vwma": {"kind": "pandas_ta", "fn": "vwma", "params": {"length": 10}, "min_bars": 10},
    
    # ===== PANDAS-TA INDICATORS (Overlap) =====
    "alma": {"kind": "pandas_ta", "fn": "alma", "params": {"length": 9, "sigma": 6, "offset": 0.85}, "min_bars": 9},
    "dema": {"kind": "pandas_ta", "fn": "dema", "params": {"length": 10}, "min_bars": 10},
    "ema": {"kind": "pandas_ta", "fn": "ema", "params": {"length": 10}, "min_bars": 10},
    "fwma": {"kind": "pandas_ta", "fn": "fwma", "params": {"length": 10}, "min_bars": 10},
    "hilo": {"kind": "pandas_ta", "fn": "hilo", "params": {"high_length": 13, "low_length": 21, "mamode": "sma"}, "min_bars": 21},
    "hl2": {"kind": "pandas_ta", "fn": "hl2", "params": {}, "min_bars": 1},
    "hlc3": {"kind": "pandas_ta", "fn": "hlc3", "params": {}, "min_bars": 1},
    "hma": {"kind": "pandas_ta", "fn": "hma", "params": {"length": 10}, "min_bars": 10},
    "hwc": {"kind": "pandas_ta", "fn": "hwc", "params": {"na": None}, "min_bars": 1},
    "hwma": {"kind": "pandas_ta", "fn": "hwma", "params": {"na": None}, "min_bars": 1},
    "jma": {"kind": "pandas_ta", "fn": "jma", "params": {"length": 7, "phase": 0, "offset": 0}, "min_bars": 7},
    "kama": {"kind": "pandas_ta", "fn": "kama", "params": {"length": 10, "fast": 2, "slow": 30}, "min_bars": 10},
    "linreg": {"kind": "pandas_ta", "fn": "linreg", "params": {"length": 14, "offset": 0}, "min_bars": 14},
    "mcgd": {"kind": "pandas_ta", "fn": "mcgd", "params": {"length": 10}, "min_bars": 10},
    "midpoint": {"kind": "pandas_ta", "fn": "midpoint", "params": {"length": 2}, "min_bars": 2},
    "midprice": {"kind": "pandas_ta", "fn": "midprice", "params": {"length": 2}, "min_bars": 2},
    "ohlc4": {"kind": "pandas_ta", "fn": "ohlc4", "params": {}, "min_bars": 1},
    "pwma": {"kind": "pandas_ta", "fn": "pwma", "params": {"length": 10}, "min_bars": 10},
    "rma": {"kind": "pandas_ta", "fn": "rma", "params": {"length": 10}, "min_bars": 10},
    "sinwma": {"kind": "pandas_ta", "fn": "sinwma", "params": {"length": 14}, "min_bars": 14},
    "sma": {"kind": "pandas_ta", "fn": "sma", "params": {"length": 10}, "min_bars": 10},
    "ssf": {"kind": "pandas_ta", "fn": "ssf", "params": {"length": 5}, "min_bars": 5},
    "swma": {"kind": "pandas_ta", "fn": "swma", "params": {"length": 10}, "min_bars": 10},
    "t3": {"kind": "pandas_ta", "fn": "t3", "params": {"length": 10}, "min_bars": 10},
    "tema": {"kind": "pandas_ta", "fn": "tema", "params": {"length": 10}, "min_bars": 10},
    "trima": {"kind": "pandas_ta", "fn": "trima", "params": {"length": 10}, "min_bars": 10},
    "vidya": {"kind": "pandas_ta", "fn": "vidya", "params": {"length": 14, "dr": None}, "min_bars": 14},
    "wcp": {"kind": "pandas_ta", "fn": "wcp", "params": {}, "min_bars": 1},
    "wma": {"kind": "pandas_ta", "fn": "wma", "params": {"length": 10}, "min_bars": 10},
    "zlma": {"kind": "pandas_ta", "fn": "zlma", "params": {"length": 10}, "min_bars": 10},
}


def get_available_indicators() -> Dict[str, Any]:
    """
    Return indicator metadata for discovery.

    We expose:
    - supported: all indicators in the catalog
    - available_now: indicators that can run in the current runtime
      (manual always; obb only if extension installed; talib only if installed; pandas_ta only if installed)
    """
    supported = []
    available_now = []

    for name, cfg in sorted(INDICATOR_CATALOG.items(), key=lambda x: x[0]):
        kind = cfg.get("kind")
        item = {
            "name": name,
            "kind": kind,
            "min_bars": cfg.get("min_bars", 1),
            "params": cfg.get("params", {}) or {},
        }
        supported.append(item)

        is_available = False
        if kind == "manual":
            is_available = True
        elif kind == "obb":
            is_available = bool(obb_technical_available and obb is not None)
        elif kind == "talib":
            is_available = bool(talib_available and talib is not None)
        elif kind == "pandas_ta" or kind == "ta":
            is_available = ta is not None

        if is_available:
            available_now.append(item)

    return _sanitize_for_json(
        {
            "openbb_technical_available": bool(obb_technical_available and obb is not None),
            "talib_available": bool(talib_available and talib is not None),
            "pandas_ta_available": ta is not None,
            "supported": supported,
            "available_now": available_now,
            "total_supported": len(supported),
            "total_available": len(available_now),
        }
    )


def _require_ohlcv(out: pd.DataFrame) -> Tuple[bool, str]:
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        return False, f"Missing OHLCV columns: {', '.join(missing)}"
    return True, "ok"


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return atr


def _apply_manual_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"

    ok, reason = _require_ohlcv(out)
    if not ok:
        return out, False, reason

    fn_name = cfg.get("fn")
    params = cfg.get("params", {}) or {}

    try:
        if fn_name == "sma":
            length = int(params.get("length", 20))
            out[f"SMA_{length}"] = out["Close"].rolling(window=length, min_periods=length).mean()
            return out, True, "ok"

        if fn_name == "ema":
            length = int(params.get("length", 20))
            out[f"EMA_{length}"] = _ema(out["Close"], length)
            return out, True, "ok"

        if fn_name == "rsi":
            length = int(params.get("length", 14))
            out[f"RSI_{length}"] = _rsi(out["Close"], length)
            return out, True, "ok"

        if fn_name == "bbands":
            length = int(params.get("length", 20))
            std_mult = float(params.get("std", 2.0))
            mid = out["Close"].rolling(window=length, min_periods=length).mean()
            std = out["Close"].rolling(window=length, min_periods=length).std()
            out[f"BBM_{length}"] = mid
            out[f"BBU_{length}"] = mid + std_mult * std
            out[f"BBL_{length}"] = mid - std_mult * std
            return out, True, "ok"

        if fn_name == "macd":
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            ema_fast = _ema(out["Close"], fast)
            ema_slow = _ema(out["Close"], slow)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = macd_line - signal_line
            out[f"MACD_{fast}_{slow}_{signal}"] = macd_line
            out[f"MACDs_{fast}_{slow}_{signal}"] = signal_line
            out[f"MACDh_{fast}_{slow}_{signal}"] = hist
            return out, True, "ok"

        if fn_name == "atr":
            length = int(params.get("length", 14))
            out[f"ATR_{length}"] = _atr(out["High"], out["Low"], out["Close"], length)
            return out, True, "ok"

        if fn_name == "vwap":
            tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
            pv = tp * out["Volume"].fillna(0)
            cum_pv = pv.cumsum()
            cum_vol = out["Volume"].fillna(0).cumsum().replace(0, pd.NA)
            out["VWAP"] = cum_pv / cum_vol
            return out, True, "ok"

        if fn_name == "ichimoku":
            tenkan = int(params.get("tenkan", 9))
            kijun = int(params.get("kijun", 26))
            senkou_b = int(params.get("senkou_b", 52))
            shift = int(params.get("shift", 26))

            tenkan_sen = (out["High"].rolling(tenkan, min_periods=tenkan).max() + out["Low"].rolling(tenkan, min_periods=tenkan).min()) / 2.0
            kijun_sen = (out["High"].rolling(kijun, min_periods=kijun).max() + out["Low"].rolling(kijun, min_periods=kijun).min()) / 2.0
            senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(shift)
            senkou_b_line = ((out["High"].rolling(senkou_b, min_periods=senkou_b).max() + out["Low"].rolling(senkou_b, min_periods=senkou_b).min()) / 2.0).shift(shift)
            chikou = out["Close"].shift(-shift)

            out["ICH_TENKAN"] = tenkan_sen
            out["ICH_KIJUN"] = kijun_sen
            out["ICH_SA"] = senkou_a
            out["ICH_SB"] = senkou_b_line
            out["ICH_CHIKOU"] = chikou
            return out, True, "ok"

        return out, False, "manual indicator not implemented"
    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def _extract_values_from_result(result: Any, preferred_name: Optional[str] = None) -> Optional[Tuple[List[str], pd.DataFrame]]:
    """
    Try to convert an OpenBB result (OBBject/DataFrame/Series) to a DataFrame of indicators.
    Returns tuple (column_names, dataframe) or None if extraction fails.
    """
    result_df = None
    if result is None:
        return None

    if hasattr(result, "to_df"):
        try:
            result_df = result.to_df()
        except Exception as e:
            print(f"Error converting to_df: {e}")
            return None
    elif isinstance(result, pd.DataFrame):
        result_df = result
    elif isinstance(result, pd.Series):
        # Convert series to single-column DataFrame
        col_name = preferred_name or "indicator"
        result_df = pd.DataFrame({col_name: result})
    else:
        return None

    if result_df is None or result_df.empty:
        return None

    cols = list(result_df.columns)

    # If preferred name provided, try to pick matching column
    if preferred_name:
        for c in cols:
            if c.upper() == preferred_name.upper():
                return [c], result_df[[c]]
        for c in cols:
            if preferred_name.upper() in c.upper():
                return [c], result_df[[c]]

    # Otherwise, return all columns
    return cols, result_df


def _apply_obb_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Apply a single OpenBB indicator to the dataframe."""
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"

    fn_name = cfg.get("fn")
    params = cfg.get("params", {}) or {}
    if not obb_technical_available or obb is None:
        return out, False, "OpenBB technical extension not available"

    fn = getattr(obb.technical, fn_name, None)
    if fn is None:
        return out, False, f"OpenBB function {fn_name} not found"

    # Build lowercase OHLCV for OpenBB
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in out.columns for col in required_cols):
        return out, False, "Missing OHLCV columns"
    out_lower = out[required_cols].copy()
    out_lower.rename(columns={c: c.lower() for c in required_cols}, inplace=True)

    try:
        result = fn(data=out_lower, **params)
        extracted = _extract_values_from_result(result, preferred_name=name.upper())
        if extracted is None:
            return out, False, "no data returned"
        cols, result_df = extracted

        # Add indicator columns (skip price-like columns)
        price_cols = {"open", "high", "low", "close", "volume"}
        added = 0
        for col in cols:
            if col.lower() in price_cols:
                continue
            series = result_df[col]
            if len(series) == len(out):
                out[col] = series.values
                added += 1
        if added == 0:
            return out, False, "no indicator columns added"
        return out, True, "ok"
    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def _apply_talib_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Apply a TA-Lib indicator to the dataframe."""
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"
    
    if not talib_available or talib is None:
        return out, False, "TA-Lib not available"
    
    fn_name = cfg.get("fn")
    params = cfg.get("params", {}) or {}
    
    # Get TA-Lib function
    talib_fn = getattr(talib, fn_name, None)
    if talib_fn is None:
        return out, False, f"TA-Lib function {fn_name} not found"
    
    # Prepare OHLCV data
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in out.columns for col in required_cols):
        return out, False, "Missing OHLCV columns"
    
    try:
        # Map parameter names from our format to TA-Lib format
        # TA-Lib uses timeperiod, fastperiod, slowperiod, etc.
        talib_params = {}
        for key, value in params.items():
            # Map common parameter names
            if key == "length":
                talib_params["timeperiod"] = int(value)
            elif key == "fast":
                talib_params["fastperiod"] = int(value)
            elif key == "slow":
                talib_params["slowperiod"] = int(value)
            elif key == "signal":
                talib_params["signalperiod"] = int(value)
            elif key == "fastk":
                talib_params["fastk_period"] = int(value)
            elif key == "fastd":
                talib_params["fastd_period"] = int(value)
            else:
                talib_params[key] = value
        
        # Call TA-Lib function based on its signature
        # Most TA-Lib functions take (high, low, close, ...) or (close, ...)
        result = None
        
        # Pattern recognition functions take OHLC
        if fn_name.startswith("CDL"):
            result = talib_fn(
                out["Open"].values,
                out["High"].values,
                out["Low"].values,
                out["Close"].values,
                **talib_params
            )
        # Volume indicators
        elif fn_name in ["AD", "ADOSC", "OBV"]:
            if fn_name == "AD" or fn_name == "ADOSC":
                result = talib_fn(
                    out["High"].values,
                    out["Low"].values,
                    out["Close"].values,
                    out["Volume"].values,
                    **talib_params
                )
            else:  # OBV
                result = talib_fn(out["Close"].values, out["Volume"].values, **talib_params)
        # Price transform functions
        elif fn_name in ["TYPPRICE", "WCLPRICE", "AVGPRICE", "MEDPRICE"]:
            if fn_name == "TYPPRICE":
                result = talib_fn(out["High"].values, out["Low"].values, out["Close"].values)
            elif fn_name == "WCLPRICE":
                result = talib_fn(out["High"].values, out["Low"].values, out["Close"].values)
            elif fn_name == "AVGPRICE":
                result = talib_fn(
                    out["Open"].values,
                    out["High"].values,
                    out["Low"].values,
                    out["Close"].values
                )
            else:  # MEDPRICE
                result = talib_fn(out["High"].values, out["Low"].values)
        # ATR, NATR, TRANGE
        elif fn_name in ["ATR", "NATR", "TRANGE"]:
            result = talib_fn(
                out["High"].values,
                out["Low"].values,
                out["Close"].values,
                **talib_params
            )
        # ADX, ADXR, DX, MINUS_DI, PLUS_DI
        elif fn_name in ["ADX", "ADXR", "DX", "MINUS_DI", "PLUS_DI"]:
            result = talib_fn(
                out["High"].values,
                out["Low"].values,
                out["Close"].values,
                **talib_params
            )
        # MINUS_DM, PLUS_DM
        elif fn_name in ["MINUS_DM", "PLUS_DM"]:
            result = talib_fn(out["High"].values, out["Low"].values, **talib_params)
        # MFI
        elif fn_name == "MFI":
            result = talib_fn(
                out["High"].values,
                out["Low"].values,
                out["Close"].values,
                out["Volume"].values,
                **talib_params
            )
        # Stochastic variants
        elif fn_name in ["STOCH", "STOCHF", "STOCHRSI"]:
            if fn_name == "STOCH":
                slowk, slowd = talib_fn(
                    out["High"].values,
                    out["Low"].values,
                    out["Close"].values,
                    **talib_params
                )
                out[f"{name.upper()}_K"] = slowk
                out[f"{name.upper()}_D"] = slowd
                return out, True, "ok"
            elif fn_name == "STOCHF":
                fastk, fastd = talib_fn(
                    out["High"].values,
                    out["Low"].values,
                    out["Close"].values,
                    **talib_params
                )
                out[f"{name.upper()}_K"] = fastk
                out[f"{name.upper()}_D"] = fastd
                return out, True, "ok"
            else:  # STOCHRSI
                fastk, fastd = talib_fn(out["Close"].values, **talib_params)
                out[f"{name.upper()}_K"] = fastk
                out[f"{name.upper()}_D"] = fastd
                return out, True, "ok"
        # Bollinger Bands
        elif fn_name == "BBANDS":
            upper, middle, lower = talib_fn(out["Close"].values, **talib_params)
            out[f"{name.upper()}_UPPER"] = upper
            out[f"{name.upper()}_MIDDLE"] = middle
            out[f"{name.upper()}_LOWER"] = lower
            return out, True, "ok"
        # MACD variants
        elif fn_name in ["MACD", "MACDEXT", "MACDFIX"]:
            if fn_name == "MACD":
                macd, signal, hist = talib_fn(out["Close"].values, **talib_params)
            elif fn_name == "MACDEXT":
                macd, signal, hist = talib_fn(out["Close"].values, **talib_params)
            else:  # MACDFIX
                macd, signal, hist = talib_fn(out["Close"].values, **talib_params)
            out[f"{name.upper()}_MACD"] = macd
            out[f"{name.upper()}_SIGNAL"] = signal
            out[f"{name.upper()}_HIST"] = hist
            return out, True, "ok"
        # AROON
        elif fn_name == "AROON":
            aroondown, aroonup = talib_fn(out["High"].values, out["Low"].values, **talib_params)
            out[f"{name.upper()}_DOWN"] = aroondown
            out[f"{name.upper()}_UP"] = aroonup
            return out, True, "ok"
        # MAMA
        elif fn_name == "MAMA":
            mama, fama = talib_fn(out["Close"].values, **talib_params)
            out[f"{name.upper()}_MAMA"] = mama
            out[f"{name.upper()}_FAMA"] = fama
            return out, True, "ok"
        # Most other indicators take just close price
        else:
            result = talib_fn(out["Close"].values, **talib_params)
        
        if result is not None:
            # Handle numpy array result
            if isinstance(result, np.ndarray):
                if len(result) == len(out):
                    out[name.upper()] = result
                    return out, True, "ok"
                else:
                    return out, False, f"length mismatch: expected {len(out)}, got {len(result)}"
            else:
                return out, False, "unexpected result type"
        else:
            return out, False, "no data returned"
            
    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def _apply_pandas_ta_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Apply a pandas-ta indicator to the dataframe.

    Uses introspection to only pass arguments that the underlying function expects,
    to avoid 'multiple values for argument' and missing-argument errors.
    """
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"

    if ta is None:
        return out, False, "pandas_ta not available"

    fn_name = cfg.get("fn")
    params = cfg.get("params", {}) or {}

    pandas_ta_fn = getattr(ta, fn_name, None)
    if pandas_ta_fn is None:
        return out, False, f"pandas-ta function {fn_name} not found"

    # We generally need at least Close prices; High/Low/Open/Volume improve coverage.
    required_cols = ["Close"]
    if not all(col in out.columns for col in required_cols):
        return out, False, "Missing Close column"

    try:
        sig = inspect.signature(pandas_ta_fn)
        param_names = set(sig.parameters.keys())

        kwargs: Dict[str, Any] = {}

        # Common argument names used by pandas-ta; only pass if the func accepts them.
        if "close" in param_names:
            kwargs["close"] = out["Close"]
        if "high" in param_names and "High" in out.columns:
            kwargs["high"] = out["High"]
        if "low" in param_names and "Low" in out.columns:
            kwargs["low"] = out["Low"]
        if ("open" in param_names or "open_" in param_names) and "Open" in out.columns:
            if "open" in param_names:
                kwargs["open"] = out["Open"]
            if "open_" in param_names:
                kwargs["open_"] = out["Open"]
        if "volume" in param_names and "Volume" in out.columns:
            kwargs["volume"] = out["Volume"]
        # Some functions support df-wide input
        if "df" in param_names:
            kwargs["df"] = out

        # Only pass configured params that the function actually accepts
        for k, v in params.items():
            if k in param_names and k not in kwargs:
                kwargs[k] = v

        result = pandas_ta_fn(**kwargs)

        if result is None:
            return out, False, "no data returned"

        # pandas-ta usually returns a DataFrame or Series
        if isinstance(result, pd.DataFrame):
            added = 0
            for col in result.columns:
                series = result[col]
                if len(series) == len(out):
                    out[col] = series.values
                    added += 1
            if added > 0:
                return out, True, "ok"
            return out, False, "no columns added"

        if isinstance(result, pd.Series):
            if len(result) == len(out):
                out[name.upper()] = result.values
                return out, True, "ok"
            return out, False, f"length mismatch: expected {len(out)}, got {len(result)}"

        return out, False, "unexpected result type"

    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def _apply_ta_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Legacy function - redirects to pandas_ta"""
    return _apply_pandas_ta_indicator(out, name, cfg)


def compute_selected_indicators(df: pd.DataFrame, indicators: List[str]) -> Tuple[pd.DataFrame, List[str], List[Dict[str, str]]]:
    """
    Compute only the requested indicators.
    Tries multiple libraries in priority order: OpenBB > TA-Lib > pandas-ta > manual
    Returns (df_with_indicators, computed_list, skipped_list)
    skipped_list entries: {"name": "...", "reason": "...", "library": "..."}
    """
    if df is None or df.empty:
        raise ValueError("No data to compute indicators.")
    out = df.copy()
    computed: List[str] = []
    skipped: List[Dict[str, str]] = []

    for ind in indicators:
        ind_l = ind.lower()
        cfg = INDICATOR_CATALOG.get(ind_l)
        if cfg is None:
            skipped.append({"name": ind, "reason": "unknown indicator", "library": "none"})
            continue
        
        kind = cfg.get("kind", "obb")
        min_bars = cfg.get("min_bars", 1)
        if len(out) < min_bars:
            skipped.append({"name": ind, "reason": f"insufficient data (need {min_bars})", "library": "none"})
            continue

        # Try libraries in priority order
        ok = False
        reason = ""
        library_used = ""
        
        # Priority 1: OpenBB (if kind is obb or if OpenBB is available and has the indicator)
        if kind == "obb" or (obb_technical_available and obb is not None):
            if kind == "obb":
                out, ok, reason = _apply_obb_indicator(out, ind_l, cfg)
                if ok:
                    library_used = "openbb"
            # Also try OpenBB for other kinds if available
            elif obb_technical_available and obb is not None:
                # Check if OpenBB has this indicator
                if hasattr(obb.technical, ind_l):
                    out, ok, reason = _apply_obb_indicator(out, ind_l, cfg)
                    if ok:
                        library_used = "openbb"
        
        # Priority 2: TA-Lib (if not succeeded and kind is talib or TA-Lib is available)
        if not ok and (kind == "talib" or talib_available):
            if kind == "talib":
                out, ok, reason = _apply_talib_indicator(out, ind_l, cfg)
                if ok:
                    library_used = "talib"
            # Also try TA-Lib for other kinds if available
            elif talib_available and talib is not None:
                # Check if TA-Lib has this indicator (uppercase function name)
                talib_fn_name = cfg.get("fn", ind_l.upper())
                if hasattr(talib, talib_fn_name):
                    out, ok, reason = _apply_talib_indicator(out, ind_l, cfg)
                    if ok:
                        library_used = "talib"
        
        # Priority 3: pandas-ta (if not succeeded and kind is pandas_ta or pandas-ta is available)
        if not ok and (kind == "pandas_ta" or ta is not None):
            if kind == "pandas_ta":
                out, ok, reason = _apply_pandas_ta_indicator(out, ind_l, cfg)
                if ok:
                    library_used = "pandas_ta"
            # Also try pandas-ta for other kinds if available
            elif ta is not None:
                # Check if pandas-ta has this indicator
                if hasattr(ta, ind_l):
                    out, ok, reason = _apply_pandas_ta_indicator(out, ind_l, cfg)
                    if ok:
                        library_used = "pandas_ta"
        
        # Priority 4: Manual (if not succeeded and kind is manual)
        if not ok and kind == "manual":
            out, ok, reason = _apply_manual_indicator(out, ind_l, cfg)
            if ok:
                library_used = "manual"
        
        # Legacy: try old "ta" kind (pandas-ta)
        if not ok and kind == "ta":
            out, ok, reason = _apply_ta_indicator(out, ind_l, cfg)
            if ok:
                library_used = "pandas_ta"

        if ok:
            computed.append(ind)
        else:
            skipped.append({"name": ind, "reason": reason, "library": library_used or "none"})

    return out, computed, skipped


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _serialize_df(df: pd.DataFrame) -> list[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    # Reset index to keep date in the payload
    if out.index.name is not None:
        out = out.reset_index()
    # Convert datetime to isoformat
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Ensure JSON-safe floats (no NaN/Inf) before to_dict
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notnull(out), None)
    return out.to_dict(orient="records")


def _sanitize_for_json(obj: Any) -> Any:
    """
    Convert NaN/Inf and numpy/pandas scalar types to JSON-safe Python values.
    Starlette/FastAPI disallows NaN/Infinity in JSON by default.
    """
    if obj is None:
        return None

    # Pandas NA
    if obj is pd.NA:
        return None

    # Datetimes
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()

    # Numpy scalars
    if isinstance(obj, np.generic):
        obj = obj.item()

    # Floats (including converted numpy floats)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # Containers
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    return obj


def analyze_ticker(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    exchange: str = "NSE",
    use_yfinance_fallback: bool = True,
    indicators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fetch historical data (Zerodha first, then optional yfinance fallback),
    compute indicators, and return profile + data in serializable form.
    """
    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    start_str = _format_date(start_dt)
    end_str = _format_date(end_dt)

    df = None
    profile = None

    # Zerodha first (NSE/BSE)
    if kite is not None:
        try:
            df = fetch_historical_zerodha(ticker, start_str, end_str, exchange=exchange)
        except Exception as e:
            print(f"Zerodha fetch failed in service: {e}")

    # Optional fallback to yfinance (e.g., for non-Indian tickers)
    if (df is None or df.empty) and use_yfinance_fallback:
        try:
            df = fetch_historical_yfinance(ticker, start_str, end_str)
        except Exception as e:
            print(f"yfinance fetch failed in service: {e}")

    if df is None or df.empty:
        raise ValueError("No historical data available from Zerodha or fallback sources.")

    # Profile: prefer OpenBB if available, else yfinance
    if openbb_available:
        try:
            profile = fetch_profile_openbb(ticker)
        except Exception as e:
            print(f"OpenBB profile fetch failed: {e}")

    if profile is None:
        try:
            profile = fetch_profile_yfinance(ticker)
        except Exception as e:
            print(f"yfinance profile fetch failed: {e}")
            profile = None

    # Compute indicators
    computed_meta: List[str] = []
    skipped_meta: List[Dict[str, str]] = []
    if indicators:
        df_with_ind, computed_meta, skipped_meta = compute_selected_indicators(df, indicators)
    else:
        df_with_ind = compute_indicators(df)
        # No explicit meta when using default compute_indicators

    payload = {
        "ticker": ticker,
        "exchange": exchange,
        "start": start_str,
        "end": end_str,
        "profile": profile,
        "rows": len(df_with_ind) if df_with_ind is not None else 0,
        "data": _serialize_df(df_with_ind),
        "computed_indicators": computed_meta,
        "skipped_indicators": skipped_meta,
    }
    return _sanitize_for_json(payload)


