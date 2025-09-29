// frontend/public/sw.js
const APP_CACHE  = "icc-app-v3";
const DATA_CACHE = "icc-data-v3";

// Use relative paths so this works at /static/ (FastAPI) and at /<repo>/ (GitHub Pages)
const PRECACHE_URLS = [
  "./",              // directory index
  "index.html",
  "icc.js",
  "snapshot.json",   // last built snapshot for offline/data fallback
];

// -------- install: precache app shell --------
self.addEventListener("install", (evt) => {
  evt.waitUntil(
    caches.open(APP_CACHE).then((cache) => cache.addAll(PRECACHE_URLS)).catch(() => {})
  );
  self.skipWaiting();
});

// -------- activate: clean old caches --------
self.addEventListener("activate", (evt) => {
  evt.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((k) => {
          if (k !== APP_CACHE && k !== DATA_CACHE) return caches.delete(k);
        })
      )
    )
  );
  self.clients.claim();
});

// Utility: respond with cached index.html for navigations (offline support)
async function handleNavigate(request) {
  // Try network first for navigations, fall back to cached index.html
  try {
    const res = await fetch(request);
    // Optionally cache fresh HTML
    const copy = res.clone();
    caches.open(APP_CACHE).then((c) => c.put("index.html", copy)).catch(() => {});
    return res;
  } catch {
    const cached = await caches.match("index.html");
    return cached || new Response("<h1>Offline</h1>", { headers: { "Content-Type": "text/html" } });
  }
}

// -------- fetch: routing --------
self.addEventListener("fetch", (evt) => {
  const req = evt.request;

  // Only GET requests are cacheable
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const isSameOrigin = url.origin === self.location.origin;

  // 1) Page navigations: network-first, fall back to cached index.html
  if (req.mode === "navigate") {
    evt.respondWith(handleNavigate(req));
    return;
  }

  // 2) API: /api/portfolios — network-first, then cached API, then snapshot.json, then []
  if (isSameOrigin && url.pathname === "/api/portfolios") {
    evt.respondWith((async () => {
      try {
        const res = await fetch(req, { cache: "no-store" });
        const copy = res.clone();
        caches.open(DATA_CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      } catch {
        // cached API?
        const cachedApi = await caches.match(req);
        if (cachedApi) return cachedApi;
        // fallback to cached snapshot.json
        const snap = await caches.match("snapshot.json");
        if (snap) {
          try {
            const arr = await snap.clone().json();
            return new Response(JSON.stringify(arr), {
              headers: { "Content-Type": "application/json" }
            });
          } catch {}
        }
        // final fallback
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
    })());
    return;
  }

  // 3) Same-origin static assets (scripts, styles, images, fonts…) — cache-first
  if (isSameOrigin) {
    evt.respondWith((async () => {
      const cached = await caches.match(req);
      if (cached) return cached;
      try {
        const res = await fetch(req);
        // Cache successful static responses
        const copy = res.clone();
        caches.open(APP_CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      } catch {
        // If fetch fails and not in cache, just let it error
        return caches.match("index.html") || Response.error();
      }
    })());
    return;
  }

  // 4) Cross-origin requests (e.g., fonts/CDN) — try network, fall back to cache if present
  evt.respondWith((async () => {
    try {
      return await fetch(req);
    } catch {
      const cached = await caches.match(req);
      return cached || Response.error();
    }
  })());
});

// Optional: allow immediate activation on new version
self.addEventListener("message", (evt) => {
  if (evt.data === "SKIP_WAITING") self.skipWaiting();
});
