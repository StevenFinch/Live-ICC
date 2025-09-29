from __future__ import annotations
"""
backend/server.py — FastAPI backend for resilient live ICC data

- Always exposes index portfolios (sp500/sp100/dow30/ndx100), with size fallbacks.
- Never returns an empty /api/portfolios (placeholders + keep last snapshot).
- Refresh runs fully in the background (startup doesn't await it).
"""

# ───────── stdlib ───────── #
import asyncio
import datetime as dt
import logging
import pathlib
from typing import Dict, List

# ───────── third-party ───────── #
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ───────── app paths ───────── #
ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB  = ROOT / "frontend" / "public"

app = FastAPI(title="US-Market Live ICC API", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB, html=True), name="static")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/static/" if WEB.exists() else "/docs")

# Import ICC engine (sync code)
from . import icc_market_live as live  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s: %(message)s")

# ───────── constants ───────── #
INDEX_CODES: Dict[str, str] = {
    "sp500":  "S&P 500",
    "sp100":  "S&P 100",
    "dow30":  "Dow Jones 30",
    "ndx100": "Nasdaq-100",
}
PLACEHOLDER_CODES: List[str] = [
    "sp500", "sp100", "dow30", "ndx100", "value", "growth", "top50", "bottom50", "all"
]

# ───────── in-memory snapshot ───────── #
FIRM: Dict[str, pd.DataFrame] = {}
SUMM: Dict[str, Dict]        = {}

# ───────── helpers ───────── #
def _seed_placeholders() -> None:
    """Ensure /api/portfolios is never empty."""
    today = dt.date.today().isoformat()
    for code in PLACEHOLDER_CODES:
        if code not in SUMM:
            SUMM[code] = {"code": code, "label": INDEX_CODES.get(code, code.upper()), "date": today, "n": 0}
            FIRM[code] = pd.DataFrame()

def stats(sub: pd.DataFrame) -> Dict:
    if sub.empty:
        return dict(n=0)
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

def build_ports(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ports: Dict[str, pd.DataFrame] = {"all": df}

    def size_fallback(code: str, base: pd.DataFrame) -> pd.DataFrame:
        target_n = {"sp500": 500, "sp100": 100, "dow30": 30, "ndx100": 100}.get(code, 0)
        sub = base.dropna(subset=["mktcap"])
        return sub.nlargest(target_n, "mktcap") if target_n and not sub.empty else base.iloc[0:0]

    # Index buckets (official list → fallback to size ranking)
    for code in INDEX_CODES:
        try:
            idx = live._index_tickers(code) or []  # may return [] on failure
            subset = df[df.ticker.isin(idx)]
            ports[code] = subset if not subset.empty else size_fallback(code, df)
        except Exception as exc:
            logging.warning("index %s skipped (%s); using fallback", code, exc)
            ports[code] = size_fallback(code, df)

    # Sector buckets
    for sec in df.sector.dropna().unique():
        s = str(sec).strip()
        if s:
            ports[f"sec_{s.split()[0].lower()}"] = df[df.sector == sec]

    # Size buckets
    sub = df.dropna(subset=["mktcap"])
    if not sub.empty:
        ports["top50"]    = sub.nlargest(50, "mktcap")
        ports["bottom50"] = sub.nsmallest(50, "mktcap")

    # Value/Growth buckets
    val_df = df.dropna(subset=["bm"])
    if not val_df.empty:
        q70, q30 = val_df.bm.quantile([0.7, 0.3])
        ports["value"]  = val_df[val_df.bm >= q70]
        ports["growth"] = val_df[val_df.bm <= q30]

    return ports

def _safe_index(code: str) -> List[str]:
    try:
        return live._index_tickers(code) or []
    except Exception as exc:
        logging.warning("safe_index(%s) failed: %s", code, exc)
        return []

# ───────── background refresh ───────── #
async def refresh() -> None:
    """Build a fresh snapshot in the background; keep old data on any failure."""
    try:
        staged_universes = [
            _safe_index("sp500"),
            _safe_index("sp100"),
            _safe_index("ndx100"),
            ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","JPM","AVGO"],  # last resort
        ]

        panel = pd.DataFrame()
        for uni in staged_universes:
            if not uni:
                continue
            try:
                panel = await asyncio.to_thread(live.get_live_panel, uni)
                if not panel.empty:
                    break
            except Exception as exc:
                logging.warning("stage failed (%s): %s", (str(uni)[:40] + "..."), exc)

        if panel.empty:
            logging.warning("refresh skipped — no panel; keeping previous snapshot")
            _seed_placeholders()
            return

        # Recompute into new dicts, then atomic swap
        new_SUMM, new_FIRM = {}, {}
        for code, sub in build_ports(panel).items():
            label = INDEX_CODES.get(code, code)
            new_SUMM[code] = {"code": code, "label": label, "date": dt.date.today().isoformat(), **stats(sub)}
            new_FIRM[code] = sub

        # Atomic-ish pointer swap (avoids partial reads)
        global SUMM, FIRM
        SUMM, FIRM = new_SUMM, new_FIRM

        _seed_placeholders()  # ensure expected codes exist even if empty
        total_firms = sum(df.shape[0] for df in FIRM.values())
        logging.info("snapshot updated — %d portfolios, %d firms", len(SUMM), total_firms)

    except Exception as exc:
        logging.exception("refresh crashed; keeping previous snapshot: %s", exc)
        _seed_placeholders()

# ───────── startup: don't await refresh ───────── #
@app.on_event("startup")
async def on_startup() -> None:
    _seed_placeholders()                 # instant non-empty API
    asyncio.create_task(refresh())       # fire-and-forget; UI won't block
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(refresh, "interval", minutes=30)
    scheduler.start()

# ───────── API routes ───────── #
@app.get("/api/portfolios")
def api_portfolios() -> List[Dict]:
    return list(SUMM.values())

@app.get("/api/market/{code}")
def api_market(code: str):
    obj = SUMM.get(code)
    return obj if obj else JSONResponse({"error": "unknown code"}, 404)

@app.get("/api/firms/{code}")
def api_firms(code: str, limit: int = Query(0, ge=0, le=10000)):
    df = FIRM.get(code)
    if df is None:
        return JSONResponse({"error": "unknown code"}, 404)
    clean = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    recs = clean.head(limit).to_dict("records") if limit else clean.to_dict("records")
    return JSONResponse(recs)

@app.get("/health")
def health():
    need = ["sp500", "sp100", "dow30", "ndx100"]
    have = {k: SUMM.get(k, {"n": 0}) for k in need}
    return {
        "status": "ok" if all((have[k].get("n") or 0) > 0 for k in need) else "degraded",
        "tot_portfolios": len(SUMM),
        "tot_firms": sum((FIRM.get(k, pd.DataFrame()).shape[0] for k in FIRM)),
        "indexes": {k: {"n": int(have[k].get("n") or 0)} for k in need},
    }

# ───────── dev runner ───────── #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
