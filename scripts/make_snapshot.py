# scripts/make_snapshot.py
from __future__ import annotations
import json, logging, pathlib, time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "frontend" / "public" / "snapshot.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

INDEX_CODES: Dict[str, str] = {
    "sp500":  "S&P 500",
    "sp100":  "S&P 100",
    "dow30":  "Dow Jones 30",
    "ndx100": "Nasdaq-100",
}

_INDEX_URL = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp100":  "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}
_WIKI_TICK = lambda s: str(s).lower() in ("symbol", "ticker")

def index_tickers(code: str) -> List[str]:
    url = _INDEX_URL[code]
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        for tbl in pd.read_html(r.text):
            for col in tbl.columns:
                if _WIKI_TICK(col):
                    return (tbl[col].astype(str)
                            .str.replace(r"\.", "-", regex=True)
                            .str.upper().tolist())
    except Exception as e:
        logging.warning("index %s fetch failed: %s", code, e)
    return []

# ---------- Li–Ng–Swaminathan ICC helpers ----------
def eps_path(fe1: float, g2: float, T: int = 15, g_long: float = 0.04) -> List[float]:
    out = [fe1, fe1 * (1 + g2)]
    if g2 <= 0: g2 = 0.04
    fade = float(np.exp(np.log(g_long / g2) / T))
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out

def pv(eps: List[float], b1: float, r: float, T: int = 15, g_long: float = 0.04) -> float:
    b_ss = float(np.clip(g_long / r, 0, 1))
    step = (b1 - b_ss) / T
    pv_ = 0.0
    for k in range(1, T + 1):
        b_k = float(np.clip(b1 - step * (k - 1), 0, 1))
        pv_ += eps[k - 1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return pv_ + tv

def solve_icc(price: float, fe1: float, g2: float, div: float) -> float | None:
    if min(price, fe1) <= 0 or not np.isfinite(price): return None
    g2 = float(np.clip(g2, 0.01, 0.75))
    eps = eps_path(fe1, g2)
    b1  = float(np.clip(1 - (div or 0.0) / fe1, 0, 1))
    lo, hi = 0.01, 0.40
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pv_mid = pv(eps, b1, mid)
        lo, hi = (mid, hi) if pv_mid > price else (lo, mid)
    return mid

# ---------- Fetch panel ----------
PAUSE_EVERY, PAUSE_SEC = 25, 0.6

def fetch(sym: str):
    try:
        t = yf.Ticker(sym)
        info = t.info or {}
        price = info.get("regularMarketPrice")
        if price is None: return None

        fe1 = info.get("forwardEps")
        g2  = info.get("earningsGrowth")
        if fe1 is None or fe1 <= 0: return None
        if g2 is None: g2 = 0.04

        icc = solve_icc(price, float(fe1), float(g2), float(info.get("dividendRate") or 0.0))
        if icc is None: return None

        return dict(
            ticker   = sym,
            price    = float(price),
            dividend = float(info.get("dividendRate") or 0.0),
            mktcap   = info.get("marketCap"),
            shares   = info.get("sharesOutstanding"),
            bvps     = info.get("bookValue"),
            sector   = info.get("sector"),
            ICC      = icc,
        )
    except Exception:
        return None

def get_panel(universe: List[str]) -> pd.DataFrame:
    rows = []
    for i, sym in enumerate(universe, 1):
        r = fetch(sym)
        if r: rows.append(r)
        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["bm"] = np.where(
        (df.bvps > 0) & (df.shares > 0) & (df.mktcap > 0),
        (df.bvps * df.shares) / df.mktcap,
        np.nan,
    )
    return df.dropna(subset=["ICC"]).reset_index(drop=True)

# ---------- Portfolios & stats ----------
def build_ports(df: pd.DataFrame, idx_map: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    ports: Dict[str, pd.DataFrame] = {"all": df}

    def size_fallback(n: int) -> pd.DataFrame:
        sub = df.dropna(subset=["mktcap"])
        return sub.nlargest(n, "mktcap") if not sub.empty else df.iloc[0:0]

    # Index buckets (official list → size fallback)
    fallback_n = {"sp500":500, "sp100":100, "dow30":30, "ndx100":100}
    for code, label in INDEX_CODES.items():
        tickers = idx_map.get(code, [])
        subset  = df[df.ticker.isin(tickers)]
        if subset.empty:
            subset = size_fallback(fallback_n[code])
        ports[code] = subset

    # Sectors
    for sec in df.sector.dropna().unique():
        s = str(sec).strip()
        if s:
            ports[f"sec_{s.split()[0].lower()}"] = df[df.sector == sec]

    # Size buckets
    sub = df.dropna(subset=["mktcap"])
    if not sub.empty:
        ports["top50"]    = sub.nlargest(50, "mktcap")
        ports["bottom50"] = sub.nsmallest(50, "mktcap")

    # Value / Growth by book-to-market
    val_df = df.dropna(subset=["bm"])
    if not val_df.empty:
        q70, q30 = val_df.bm.quantile([0.7, 0.3])
        ports["value"]  = val_df[val_df.bm >= q70]
        ports["growth"] = val_df[val_df.bm <= q30]

    return ports

def stats(sub: pd.DataFrame) -> Dict:
    if sub.empty: return dict(n=0)
    vw = lambda d: float(np.average(d.ICC, weights=d.mktcap))
    res = dict(n=len(sub), market_icc=vw(sub))
    if sub.bm.notna().any():
        q70, q30 = sub.bm.quantile([0.7, 0.3])
        hi, lo   = sub[sub.bm >= q70], sub[sub.bm <= q30]
        if not hi.empty and not lo.empty:
            res.update(
                icc_value    = vw(hi),
                icc_growth   = vw(lo),
                ivp          = vw(hi) - vw(lo),
                value_spread = float(np.log(hi.bm.mean()) - np.log(lo.bm.mean())),
            )
    return res

def main():
    # Build index universes (robust)
    idx_map: Dict[str, List[str]] = {c: index_tickers(c) for c in INDEX_CODES}
    universe: List[str] = sorted({s for arr in idx_map.values() for s in arr})
    if not universe:
        # Hard fallback: a small mega-cap set ensures non-empty output
        universe = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","JPM","AVGO"]
    logging.info("universe size: %d", len(universe))

    panel = get_panel(universe)
    if panel.empty and OUT.exists():
        logging.warning("panel empty; keeping previous snapshot")
        return
    elif panel.empty:
        logging.warning("panel empty and no previous snapshot; writing placeholders")

    ports = build_ports(panel, idx_map) if not panel.empty else {code: pd.DataFrame() for code in
              ["all","sp500","sp100","dow30","ndx100","value","growth","top50","bottom50"]}

    today = pd.Timestamp.today().date().isoformat()
    arr = []
    for code, sub in ports.items():
        label = INDEX_CODES.get(code, code)
        rec = {"code": code, "label": label, "date": today, **stats(sub)}
        arr.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arr), encoding="utf-8")
    logging.info("wrote %s (%d portfolios)", OUT, len(arr))

if __name__ == "__main__":
    main()
