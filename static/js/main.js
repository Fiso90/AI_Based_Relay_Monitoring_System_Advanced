// Shared helpers used across dashboard pages.

// Minimal fetch() polyfill for older WebKit engines (e.g. wkhtmltopdf/wkhtmltoimage,
// used to generate documentation screenshots) that predate the Fetch API.
// All modern browsers use their native fetch; this only activates as a fallback.
if (typeof window.fetch !== "function") {
  window.fetch = function (url, opts) {
    opts = opts || {};
    return new Promise(function (resolve, reject) {
      const xhr = new XMLHttpRequest();
      xhr.open(opts.method || "GET", url, true);
      if (opts.headers) {
        Object.keys(opts.headers).forEach((k) => xhr.setRequestHeader(k, opts.headers[k]));
      }
      xhr.onload = function () {
        resolve({
          ok: xhr.status >= 200 && xhr.status < 300,
          status: xhr.status,
          json: function () { return Promise.resolve(JSON.parse(xhr.responseText)); },
          text: function () { return Promise.resolve(xhr.responseText); },
        });
      };
      xhr.onerror = function () { reject(new Error("Network error")); };
      xhr.send(opts.body || null);
    });
  };
}


// Auth integration: every state-changing request carries the session's CSRF
// token (from <meta name="csrf-token">), and any 401 (session expired) sends
// the user back to the login page instead of failing silently.
(function () {
  var meta = document.querySelector('meta[name="csrf-token"]');
  var token = meta ? meta.getAttribute("content") : "";
  var rawFetch = window.fetch;
  window.fetch = function (url, opts) {
    opts = opts || {};
    var method = (opts.method || "GET").toUpperCase();
    if (method !== "GET" && method !== "HEAD" && token) {
      var h = {};
      var src = opts.headers || {};
      Object.keys(src).forEach(function (k) { h[k] = src[k]; });
      h["X-CSRF-Token"] = token;
      opts.headers = h;
    }
    return rawFetch(url, opts).then(function (res) {
      if (res.status === 401 && window.location.pathname !== "/login") {
        window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname);
      }
      return res;
    });
  };
})();

// RISK_COLORS is populated at runtime from /api/eda/meta so the dashboard
// renders sensibly for any uploaded dataset's class names, not just the
// three this project shipped with. Falls back to the known defaults until
// that fetch resolves so charts requested before it loads don't crash.
var RISK_COLORS = {
  NORMAL: "#4ecb71",
  WARNING: "#f2a93b",
  "HIGH TRIP RISK": "#f2555a",
};
var DASHBOARD_CLASSES = ["NORMAL", "WARNING", "HIGH TRIP RISK"];

function refreshClassMeta() {
  return getJSON("/api/eda/meta").then(function (meta) {
    RISK_COLORS = meta.colors;
    DASHBOARD_CLASSES = meta.classes;
    return meta;
  }).catch(function () { return null; });
}

function badgeClass(risk) {
  var upper = (risk || "").toUpperCase();
  if (upper.indexOf("NORMAL") !== -1) return "badge-normal";
  if (upper.indexOf("WARN") !== -1) return "badge-warning";
  return "badge-high";
}

// NOTE: written as plain Promise chains (no async/await) rather than for any
// technical necessity in modern browsers -- it also happens to let the same
// markup render correctly in older rendering engines (e.g. the WebKit build
// used by wkhtmltopdf/wkhtmltoimage to generate documentation screenshots),
// since those predate async/await support.
function getJSON(url) {
  return fetch(url).then(function (res) {
    if (!res.ok) throw new Error("Request failed: " + url);
    return res.json();
  });
}

function postJSON(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then(function (res) { return res.json(); });
}

// Simple tab controller: any element with [data-tabs] wrapping
// .tab-btn[data-tab] buttons and .tab-panel[data-panel] targets.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-tabs]").forEach((wrapper) => {
    const buttons = wrapper.querySelectorAll(".tab-btn");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        wrapper.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        wrapper.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        wrapper.querySelector(`.tab-panel[data-panel="${target}"]`).classList.add("active");
        // fire a custom event so pages can lazy-load chart data per tab
        wrapper.dispatchEvent(new CustomEvent("tabchange", { detail: { tab: target } }));
      });
    });
  });
});
