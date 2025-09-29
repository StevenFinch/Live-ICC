from __future__ import annotations
"""
backend/icc_market_live.py – live Implied Cost of Capital (ICC) estimator
=========================================================================
CLI examples
------------
# S&P 500
python backend/icc_market_live.py sp500
# entire US market (slow)
python backend/icc_market_live.py usall
# custom list
python backend/icc_market_live.py AAPL MSFT NVDA
"""

# stdlib
import datetime as dt
import logging
import pathlib
import random
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional
import io
import time
import typing as t
import requests

# third-party
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import exceptions as yfexc

# -----------------------------------------------------------------------------
ROOT   = pathlib.Path(__file__).resolve().parents[1]
TODAY  = dt.date.today().isoformat()

PAUSE_EVERY, PAUSE_SEC = 20, 0.6       # rate-limit safety
G_MIN, G_MAX           = 0.01, 0.75
BAD_SYM                = re.compile(r"[^A-Z\\.\\-]")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# 1  Ticker universes
# -----------------------------------------------------------------------------
_DATAHUB = {
    "nasdaq": "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv",
    "nyse":   "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv",
    "amex":   "https://datahub.io/core/nyse-other-listings/r/amex-listed.csv",
}

@lru_cache(maxsize=None)
def get_us_tickers() -> List[str]:
    symbols: set[str] = set()
    for name, url in _DATAHUB.items():
        try:
            df = pd.read_csv(url, dtype=str, usecols=[0])
            col = df.columns[0]
            raw = df[col].astype(str).str.replace(r"\.", "-", regex=True).str.upper()
            symbols.update(s for s in raw if not BAD_SYM.search(s))
            logging.info("%s tickers loaded: %d", name.upper(), len(raw))
        except Exception as exc:
            logging.warning("failed to load %s: %s", name, exc)
    logging.info("total US symbols: %d", len(symbols))
    return sorted(symbols)

_WIKI_TICK = re.compile(r"(symbol|ticker)", re.I)
_INDEX_URL = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp100":  "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}

def _get(url: str, max_retries: int = 3, timeout: int = 20) -> str:
    """GET with a real User-Agent and a few retries (helps avoid 403s)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} tries") from last_err

def _find_symbol_column(df: pd.DataFrame) -> str:
    """Locate the column that contains ticker symbols."""
    candidates = [c for c in df.columns]
    lowers = {c.lower(): c for c in candidates}
    for key in ("symbol", "ticker", "code"):
        if key in lowers:
            return lowers[key]
    # fallback: first column
    return candidates[0]

def _fetch_index_from_wikipedia(url: str) -> list[str]:
    """Return Yahoo-friendly tickers from a Wikipedia index page."""
    html = _get(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found on {url}")

    # Heuristic: pick the table with the most rows that also has a symbol-like column
    best: t.Optional[pd.DataFrame] = None
    best_rows = -1
    for tbl in tables:
        try:
            sym_col = _find_symbol_column(tbl)
        except Exception:
            continue
        rows = len(tbl)
        if rows > best_rows and sym_col in tbl.columns:
            best = tbl
            best_rows = rows
    if best is None:
        best = max(tables, key=len)

    sym_col = _find_symbol_column(best)
    df = best.dropna(subset=[sym_col]).copy()
    df[sym_col] = df[sym_col].astype(str).str.strip()

    # raw symbols (e.g., "BRK.B") → Yahoo symbols ("BRK-B")
    ytickers = (
        df[sym_col]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    # filter weird symbols
    ytickers = [s for s in ytickers if s and not BAD_SYM.search(s)]

    # sanity check (avoid partial parses)
    if len(ytickers) < 30:
        raise RuntimeError(f"Only parsed {len(ytickers)} symbols from {url}; page format may have changed.")

    return ytickers

@lru_cache(maxsize=None)
def _index_tickers(code: str) -> List[str]:
    """Fetch index constituents from Wikipedia and return Yahoo-friendly tickers ('.' → '-')."""
    url = _INDEX_URL[code]
    return _fetch_index_from_wikipedia(url)

# -----------------------------------------------------------------------------
# 2  Li–Ng–Swaminathan helpers
# -----------------------------------------------------------------------------
def _eps_path(fe1: float, g2: float,
              T: int = 15, g_long: float = 0.04) -> List[float]:
    out = [fe1, fe1 * (1 + g2)]
    fade = np.exp(np.log(g_long / g2) / T)
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out

def _pv(eps: List[float], b1: float, r: float,
        T: int = 15, g_long: float = 0.04) -> float:
    b_ss = np.clip(g_long / r, 0, 1)
    step = (b1 - b_ss) / T
    pv = 0.0
    for k in range(1, T + 1):
        b_k = np.clip(b1 - step * (k - 1), 0, 1)
        pv += eps[k - 1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return pv + tv

def _solve_icc(price: float, fe1: float, g2: float, div: float) -> Optional[float]:
    if min(price, fe1) <= 0 or not np.isfinite(price):
        return None
    eps = _eps_path(fe1, g2)
    b1  = np.clip(1 - div / fe1, 0, 1)
    lo, hi = 0.01, 0.40
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pv  = _pv(eps, b1, mid)
        lo, hi = (mid, hi) if pv > price else (lo, mid)
    return mid

# -----------------------------------------------------------------------------
# 3  yfinance fetch (sequential)
# -----------------------------------------------------------------------------
def _fetch(sym: str) -> Optional[Dict[str, Any]]:
    if BAD_SYM.search(sym):
        return None
    try:
        tkr = yf.Ticker(sym)
        info = tkr.info or {}
        price = info.get("regularMarketPrice")
        if price is None:
            return None

        # try analyst trend first
        fe1, g2 = None, None
        try:
            trend = tkr.get_eps_trend(as_dict=True) or {}
            curr  = trend.get("current", {})
            fe_now, fe_next = curr.get("0y"), curr.get("+1y")
            if fe_now and fe_next and fe_now > 0:
                fe1 = fe_now
                g2  = np.clip(fe_next / fe_now - 1, G_MIN, G_MAX)
        except Exception:
            pass

        if fe1 is None:
            fe1 = info.get("forwardEps")
        if g2 is None and info.get("earningsGrowth") is not None:
            g2 = np.clip(info["earningsGrowth"], G_MIN, G_MAX)
        if fe1 is None or fe1 <= 0:
            return None
        if g2 is None:
            g2 = 0.04

        return dict(
            ticker   = sym,
            price    = price,
            dividend = info.get("dividendRate") or 0.0,
            mktcap   = info.get("marketCap"),
            shares   = info.get("sharesOutstanding"),
            bvps     = info.get("bookValue"),
            sector   = info.get("sector"),
            name     = info.get("shortName") or info.get("longName"),
            FE1      = fe1,
            g2       = g2,
        )
    except yfexc.YFRateLimitError:
        time.sleep(1.5)
        return _fetch(sym)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 4  Panel builder
# -----------------------------------------------------------------------------
def get_live_panel(symbols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, sym in enumerate(symbols, 1):
        rec = _fetch(sym)
        if rec:
            rows.append(rec)
        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("all fetches failed")
        return df

    df["bm"] = np.where(
        (df.bvps > 0) & (df.shares > 0) & (df.mktcap > 0),
        (df.bvps * df.shares) / df.mktcap,
        np.nan,
    )
    df["ICC"] = df.apply(
        lambda r: _solve_icc(r.price, r.FE1, r.g2, r.dividend), axis=1
    )
    df = df.dropna(subset=["ICC"]).reset_index(drop=True)
    df["date"] = TODAY
    logging.info("panel rows: %d / universe %d", len(df), len(symbols))
    return df

# -----------------------------------------------------------------------------
# 5  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: icc_market_live.py {usall|sp500|sp100|dow30|ndx100|TICKERS…}")
        sys.exit(0)

    arg = sys.argv[1].lower()
    if arg == "usall":
        universe = get_us_tickers()
    elif arg in _INDEX_URL:
        universe = _index_tickers(arg)
    else:
        universe = [s.upper() for s in sys.argv[1:]]

    panel = get_live_panel(universe)
    out = ROOT / "icc_live_sample.csv"
    panel.to_csv(out, index=False)
    print(panel.head())
    logging.info("saved %s (%d rows)", out, len(panel))
