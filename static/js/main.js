// Shared helpers used across dashboard pages.

const RISK_COLORS = {
  NORMAL: "#4ecb71",
  WARNING: "#f2a93b",
  "HIGH TRIP RISK": "#f2555a",
};

function badgeClass(risk) {
  if (risk === "NORMAL") return "badge-normal";
  if (risk === "WARNING") return "badge-warning";
  return "badge-high";
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
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
