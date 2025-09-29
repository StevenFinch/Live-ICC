/* icc.js — resilient loader for API (local/deployed) and snapshot (GitHub Pages)
   - Uses API at same-origin when available
   - On GitHub Pages (or file://), loads snapshot.json
   - Caches last good data in localStorage so the UI never blanks
*/

const IS_GHPAGES =
  location.hostname.endsWith(".github.io") || location.protocol === "file:";

const API = IS_GHPAGES ? null : window.location.origin;

/* Portfolio buckets → which section they belong to */
const GROUP = {
  IDX: ["sp500", "sp100", "dow30", "ndx100"],      // major indexes
  BM:  ["value", "growth"],                        // book-to-market buckets
  SEC: ["sec_"],                                   // sector buckets (prefix)
  OTH: ["top50", "bottom50", "all"]                // everything else here last
};

/* Helpers */
const pct = v => (isFinite(v) ? (v * 100).toFixed(2) + " %" : "—");
const cls = v => (isFinite(v) ? (v >= 0 ? "pos" : "neg") : "");

/* Placeholder portfolios shown until first success */
function placeholders() {
  const codes = ["sp500","sp100","dow30","ndx100","value","growth","top50","bottom50","all"];
  return codes.map(code => ({ code, label: code.toUpperCase(), n: 0 }));
}

/* Card HTML */
function card(d) {
  return `
    <div class="card">
      <div class="code">${d.label ?? d.code}</div>
      <div class="n">n=${d.n ?? "—"}</div>
      <div class="kv"><span>Market ICC</span><span class="${cls(d.market_icc)}">${pct(d.market_icc)}</span></div>
      <div class="kv"><span>Value ICC</span><span class="${cls(d.icc_value)}">${pct(d.icc_value)}</span></div>
      <div class="kv"><span>Growth ICC</span><span class="${cls(d.icc_growth)}">${pct(d.icc_growth)}</span></div>
      <div class="kv"><span>IVP</span><span class="${cls(d.ivp)}">${pct(d.ivp)}</span></div>
    </div>`;
}

/* Append rendered card HTML to a section’s grid */
function appendTo(sectionId, html) {
  document.querySelector(`#${sectionId} .grid`).insertAdjacentHTML("beforeend", html);
}

/* Render into sections */
function render(arr) {
  ["sec-ind", "sec-bm", "sec-sector", "sec-others"].forEach(
    id => (document.querySelector(`#${id} .grid`).innerHTML = "")
  );
  arr.forEach(d => {
    const code = String(d.code || "").toLowerCase();
    if (GROUP.IDX.includes(code)) {
      appendTo("sec-ind", card(d));
    } else if (GROUP.BM.includes(code)) {
      appendTo("sec-bm", card(d));
    } else if (GROUP.SEC.some(prefix => code.startsWith(prefix))) {
      appendTo("sec-sector", card(d));
    } else {
      appendTo("sec-others", card(d));
    }
  });
}

function setStamp(msg) {
  const el = document.getElementById("updated");
  if (el) el.textContent = msg;
}

/* State: keep last successful payload so the UI never blanks */
let last = null;

/* First paint immediately with placeholders */
render(placeholders());
setStamp(`(starting up — ${new Date().toLocaleTimeString()})`);

/* Try API (if present), else snapshot.json on GitHub Pages/file:// */
async function load() {
  try {
    // 1) Prefer live API when not on GitHub Pages
    if (API) {
      const r = await fetch(`${API}/api/portfolios`, { cache: "no-store" });
      if (r.ok) {
        const data = await r.json();
        if (Array.isArray(data) && data.length) {
          last = data;
          localStorage.setItem("icc_last", JSON.stringify(data));
          render(data);
          setStamp(`(updated ${new Date().toLocaleTimeString()})`);
          return;
        }
      }
    }

    // 2) Snapshot mode (GitHub Pages or file://)
    if (IS_GHPAGES || !API) {
      try {
        const snap = await fetch("snapshot.json", { cache: "reload" });
        if (snap.ok) {
          const data = await snap.json();
          if (Array.isArray(data) && data.length) {
            last = data;
            localStorage.setItem("icc_last", JSON.stringify(data));
            render(data);
            setStamp(`(snapshot — ${new Date().toLocaleTimeString()})`);
            return;
          }
        }
      } catch { /* fall through to cache */ }
    }

    // 3) No fresh data → use cached last good payload
    const cached = localStorage.getItem("icc_last");
    if (cached) {
      last = JSON.parse(cached);
      render(last);
      setStamp(`(offline cache — ${new Date().toLocaleTimeString()})`);
    } else {
      render(placeholders());
      setStamp(`(no data yet — ${new Date().toLocaleTimeString()})`);
    }
  } catch {
    // Network/parse error → keep what we have or placeholders
    if (last) {
      render(last);
      setStamp(`(retrying… kept previous — ${new Date().toLocaleTimeString()})`);
    } else {
      const cached = localStorage.getItem("icc_last");
      if (cached) {
        last = JSON.parse(cached);
        render(last);
        setStamp(`(offline cache — ${new Date().toLocaleTimeString()})`);
      } else {
        render(placeholders());
        setStamp(`(backend unreachable — ${new Date().toLocaleTimeString()})`);
      }
    }
  }
}

/* First load + refresh every 60 s */
load();
setInterval(load, 60_000);
