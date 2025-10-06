# scripts/make_snapshot.py
from __future__ import annotations
import io, json, logging, os, pathlib, time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "frontend" / "public" / "snapshot.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- Controls (override in workflow env if desired) ----------------
USALL_MAX   = int(os.getenv("USALL_MAX", "5000"))   # 0 = no cap (may take long)
PAUSE_EVERY = int(os.getenv("PAUSE_EVERY", "25"))
PAUSE_SEC   = float(os.getenv("PAUSE_SEC", "0.6"))
G_LONG      = float(os.getenv("G_LONG", "0.04"))
T_HORIZON   = int(os.getenv("T_HORIZON", "15"))

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"}

DATAHUB = {
    "nasdaq": "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv",
    "nyse":   "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv",
    "amex":   "https://datahub.io/core/nyse-other-listings/r/amex-listed.csv",
}

INDEX_CODES: Dict[str, str] = {
    "sp500":  "S&P 500",
    "sp100":  "S&P 100",
    "dow30":  "Dow Jones 30",
    "ndx100": "Nasdaq-100",
}
INDEX_WIKI = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp100":  "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}

# ---------------- Helpers: universes ----------------
def _read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), dtype=str)

def get_us_tickers() -> List[str]:
    syms: set[str] = set()
    for name, url in DATAHUB.items():
        try:
            df = _read_csv(url)
            col = df.columns[0]
            raw = (df[col].astype(str)
                         .str.replace(r"\.", "-", regex=True)
                         .str.upper()
                         .str.strip())
            syms.update([s for s in raw if s and s.isascii()])
            logging.info("%s tickers loaded: %d", name.upper(), len(raw))
        except Exception as e:
            logging.warning("failed to load %s: %s", name, e)
    out = sorted(s for s in syms if s and not any(ch in s for ch in [" ", "/"]))
    logging.info("US universe size: %d", len(out))
    return out

def index_tickers(code: str) -> List[str]:
    url = INDEX_WIKI[code]
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        def _sym_col(tbl: pd.DataFrame):
            for c in tbl.columns:
                if str(c).lower() in ("symbol","ticker","code"):
                    return str(c)
            return None
        best = None; best_rows = -1
        for t in tables:
            col = _sym_col(t)
            if col is not None and len(t) > best_rows:
                best = t; best_rows = len(t)
        if best is None:
            return []
        col = _sym_col(best)
        syms = (best[col].astype(str)
                        .str.replace(r"\s+","", regex=True)
                        .str.replace(".","-", regex=False)
                        .str.upper())
        return [s for s in syms if s and s.isascii()]
    except Exception as e:
        logging.warning("index %s fetch failed: %s", code, e)
        return []

# ---------------- ICC model (Li–Ng–Swaminathan style) ----------------
def eps_path(fe1: float, g2: float, T: int = T_HORIZON, g_long: float = G_LONG) -> List[float]:
    out = [fe1, fe1 * (1 + g2)]
    g2 = max(g2, 1e-6)
    fade = float(np.exp(np.log(g_long / g2) / T))
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out

def pv(eps: List[float], b1: float, r: float, T: int = T_HORIZON, g_long: float = G_LONG) -> float:
    b_ss = float(np.clip(g_long / r, 0.0, 1.0))
    step = (b1 - b_ss) / T
    out = 0.0
    for k in range(1, T + 1):
        b_k = float(np.clip(b1 - step*(k-1), 0.0, 1.0))
        out += eps[k-1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return out + tv

def solve_icc(price: float, fe1: float, g2: float, div: float) -> float | None:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(fe1) or fe1 <= 0:
        return None
    g2 = float(np.clip(g2 if np.isfinite(g2) else 0.04, 0.01, 0.75))
    eps = eps_path(float(fe1), g2)
    b1  = float(np.clip(1.0 - (float(div) if np.isfinite(div) else 0.0)/float(fe1), 0.0, 1.0))
    lo, hi = 0.01, 0.40
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if pv(eps, b1, mid) > price:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

# ---------------- yfinance panel ----------------
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

        icc = solve_icc(float(price), float(fe1), float(g2), float(info.get("dividendRate") or 0.0))
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
        rec = fetch(sym)
        if rec: rows.append(rec)
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

# ---------------- Portfolios built FROM THE FULL PANEL ----------------
def build_ports(df: pd.DataFrame, idx_map: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    ports: Dict[str, pd.DataFrame] = {}

    # all = FULL US UNIVERSE (this is what you want)
    ports["all"] = df

    # Index buckets: intersect with full panel; fallback to size if needed
    def size_fallback(n: int) -> pd.DataFrame:
        sub = df.dropna(subset=["mktcap"])
        return sub.nlargest(n, "mktcap") if not sub.empty else df.iloc[0:0]

    fallback_n = {"sp500":500, "sp100":100, "dow30":30, "ndx100":100}
    for code, label in INDEX_CODES.items():
        tickers = idx_map.get(code, [])
        sub = df[df.ticker.isin(tickers)]
        if sub.empty:
            sub = size_fallback(fallback_n[code])
        ports[code] = sub

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

    # Value / Growth using BM within the FULL panel
    val_df = df.dropna(subset=["bm"])
    if not val_df.empty:
        q70, q30 = val_df.bm.quantile([0.7, 0.3])
        ports["value"]  = val_df[val_df.bm >= q70]
        ports["growth"] = val_df[val_df.bm <= q30]

    return ports

def vw_icc(d: pd.DataFrame) -> float:
    return float(np.average(d.ICC, weights=d.mktcap))

def stats(sub: pd.DataFrame) -> Dict:
    if sub.empty: return dict(n=0)
    res = dict(n=len(sub), market_icc=vw_icc(sub))
    if sub.bm.notna().any():
        q70, q30 = sub.bm.quantile([0.7, 0.3])
        hi, lo   = sub[sub.bm >= q70], sub[sub.bm <= q30]
        if not hi.empty and not lo.empty:
            res.update(
                icc_value    = vw_icc(hi),
                icc_growth   = vw_icc(lo),
                ivp          = vw_icc(hi) - vw_icc(lo),
                value_spread = float(np.log(hi.bm.mean()) - np.log(lo.bm.mean())),
            )
    return res

# ---------------- Main ----------------
def main():
    # 1) Full US universe
    universe = get_us_tickers()
    if not universe:
        logging.warning("universe empty; aborting")
        return
    if USALL_MAX > 0:
        universe = universe[:USALL_MAX]
    logging.info("using %d tickers for ALL (US-all)", len(universe))

    # 2) Build panel
    panel = get_panel(universe)
    if panel.empty:
        if OUT.exists():
            logging.warning("panel empty; keeping previous snapshot")
            return
        # first-run placeholders (never ship an empty file)
        today = pd.Timestamp.today().date().isoformat()
        base_codes = ["all","sp500","sp100","dow30","ndx100","value","growth","top50","bottom50"]
        base = [{"code": c, "label": INDEX_CODES.get(c, c.upper()), "date": today, "n": 0} for c in base_codes]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(base), encoding="utf-8")
        logging.warning("wrote placeholder snapshot (no data)")
        return

    # 3) Index lists (for slicing within full panel)
    idx_map = {c: index_tickers(c) for c in INDEX_CODES}

    # 4) Build all portfolios FROM the full panel
    ports = build_ports(panel, idx_map)

    # 5) Summaries (NOTE: 'all' reflects the FULL US universe)
    today = pd.Timestamp.today().date().isoformat()
    arr = []
    for code, sub in ports.items():
        label = INDEX_CODES.get(code, code)
        arr.append({"code": code, "label": label, "date": today, **stats(sub)})

    # 6) Write snapshot.json
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arr), encoding="utf-8")
    logging.info("wrote %s (%d portfolios; panel rows=%d)", OUT, len(arr), len(panel))

if __name__ == "__main__":
    main()
