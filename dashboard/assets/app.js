/* Dashboard shell: tabs, the status bar, and the live training view.
   Re-reads data/live.js every few seconds. Script tags work from file://, so no server
   is needed; scripts/monitor.py keeps the file current while a kernel runs. */
(function () {
  "use strict";
  const { el, tile, hero, meter, lineChart } = window.Charts;
  const { renderPerformance, renderDataset, trainingCharts, runSummaryTable, compareChart, bestRow, humanRun, duration, num, section } = window.Sections;

  const POLL_MS = 5000;
  const STALE_AFTER_S = 120;
  const TERMINAL = new Set(["complete", "error", "cancel_acknowledged", "cancel_requested"]);

  const ui = { tab: "training", run: null };
  try {
    const stored = localStorage.getItem("blood-dashboard-tab");
    if (stored) ui.tab = stored;
  } catch (e) { /* storage may be unavailable; the default tab is fine */ }
  if (location.hash) ui.tab = location.hash.slice(1); // deep links such as #performance

  /* --- theme --- */
  const themeBtn = document.getElementById("theme-btn");
  function currentTheme() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit) return explicit;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function paintThemeBtn() { themeBtn.textContent = currentTheme() === "dark" ? "Light theme" : "Dark theme"; }
  try {
    const saved = localStorage.getItem("blood-dashboard-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* ignore */ }
  themeBtn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("blood-dashboard-theme", next); } catch (e) { /* ignore */ }
    paintThemeBtn();
    renderAll(true);
  });
  paintThemeBtn();

  /* --- tabs --- */
  const panels = { training: document.getElementById("tab-training"), performance: document.getElementById("tab-performance"), dataset: document.getElementById("tab-dataset") };
  function selectTab(name) {
    ui.tab = name;
    try { localStorage.setItem("blood-dashboard-tab", name); } catch (e) { /* ignore */ }
    for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== name;
    document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    renderAll(true);
  }
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));

  /* --- status chip --- */
  function statusInfo(live) {
    if (!live) return { cls: "idle", icon: "–", text: "No training run" };
    if (live.status === "complete") return { cls: "good", icon: "✓", text: "Complete" };
    if (live.status === "error") return { cls: "critical", icon: "✕", text: "Failed" };
    if (live.status === "cancel_acknowledged" || live.status === "cancel_requested") return { cls: "warning", icon: "!", text: "Cancelled" };
    if (live.status === "running") return { cls: "running", icon: "▶", text: "Running" };
    if (live.status === "queued") return { cls: "idle", icon: "…", text: "Queued" };
    return { cls: "idle", icon: "?", text: "Status unknown" };
  }
  function renderChip(live) {
    const info = statusInfo(live);
    const chip = document.getElementById("status-chip");
    chip.className = `chip ${info.cls}`;
    chip.replaceChildren(el("span", { class: "dot", "aria-hidden": "true", text: info.icon }), el("span", { text: info.text }));
    const updated = document.getElementById("updated");
    updated.textContent = live && live.monitor_updated ? `Updated ${duration(Date.now() / 1000 - live.monitor_updated)} ago` : "";
  }

  /* --- training tab --- */
  function isStale(live) {
    return live && live.monitor_updated && !TERMINAL.has(live.status) && Date.now() / 1000 - live.monitor_updated > STALE_AFTER_S;
  }
  function runOrderOf(live) {
    const planned = (live.config && live.config.candidates ? live.config.candidates.map((c) => c.name) : []);
    for (const name of Object.keys(live.runs || {})) if (!planned.includes(name)) planned.push(name);
    return planned;
  }
  function currentRunName(live, order) {
    const running = order.filter((n) => live.runs[n] && live.runs[n].status === "training");
    if (running.length) return running[running.length - 1];
    const started = order.filter((n) => live.runs[n]);
    return started.length ? started[started.length - 1] : null;
  }
  function etaSeconds(live, order, activeName) {
    const cfgEpochs = live.config && live.config.train_args ? live.config.train_args.epochs : null;
    const run = live.runs[activeName];
    if (!run || !run.rows.length) return null;
    const recent = run.rows.slice(-10).map((r) => r.epoch_time).filter(Boolean);
    const perEpoch = recent.reduce((a, b) => a + b, 0) / (recent.length || 1);
    const remainThis = Math.max(0, (run.epochs || cfgEpochs || 0) - run.rows.length) * perEpoch;
    const pending = order.filter((n) => !live.runs[n]).length;
    // Rough, because later candidates differ in size from the one running now.
    const allRuns = remainThis + pending * (cfgEpochs || 0) * perEpoch;
    return { thisRun: remainThis, pendingRuns: pending, perEpoch, allRuns };
  }

  function renderTraining(live) {
    const root = panels.training;
    root.replaceChildren();
    if (!live || !live.status) {
      root.append(el("div", { class: "empty" },
        el("strong", { text: "No training run yet" }),
        "Push a kernel with scripts/kaggle_jobs.py, then run scripts/monitor.py. This page updates on its own."));
      return;
    }
    if (isStale(live)) {
      root.append(el("div", { class: "banner" }, el("div", {},
        el("strong", { text: `This view is ${duration(Date.now() / 1000 - live.monitor_updated)} out of date` }),
        "The local monitor has stopped writing updates, so the numbers below are the last ones received. Start it again with ", el("code", { text: "python scripts/monitor.py" }), ".")));
    }
    if (live.status === "error" || live.error) {
      const message = (live.error && live.error.message) || live.failure || "The kernel reported an error.";
      root.append(el("div", { class: "banner critical" }, el("div", {}, el("strong", { text: "Training failed" }), message)));
    }

    if (live.recovered_error && !live.error) {
      root.append(el("div", { class: "banner info" }, el("div", {},
        el("strong", { text: "The final evaluation was repeated from the saved model" }),
        "Training finished normally, but the first evaluation attempt ran out of GPU memory. The selected model's saved weights were evaluated again in a separate kernel, so nothing was retrained.")));
    }
    const order = runOrderOf(live);
    const active = currentRunName(live, order);
    const run = active ? live.runs[active] : null;
    const finished = order.filter((n) => live.runs[n] && live.runs[n].status === "finished").length;

    // Overall progress in epochs, across all planned candidates.
    const totalEpochs = (live.config && live.config.train_args ? live.config.train_args.epochs : 0) * order.length;
    const doneEpochs = order.reduce((sum, n) => sum + (live.runs[n] ? live.runs[n].rows.length : 0), 0);
    const eta = active ? etaSeconds(live, order, active) : null;

    const rows = run ? run.rows : [];
    const last = rows[rows.length - 1];
    const best = rows.length ? bestRow(rows) : null;

    const overallBest = order
      .filter((n) => live.runs[n] && live.runs[n].rows.length)
      .map((n) => ({ n, row: bestRow(live.runs[n].rows) }))
      .sort((a, b) => b.row["metrics/mAP50-95(B)"] - a.row["metrics/mAP50-95(B)"])[0];

    const selected = live.selection && live.runs[live.selection.winner] ? live.runs[live.selection.winner].result : null;
    root.append(
      el("div", { class: "hero-row" },
        selected
          ? hero("Selected model, validation mAP50-95", num(selected.val_map50_95), `${humanRun(live.selection.winner)}, best saved weights`)
          : hero("Best validation mAP50-95 so far", overallBest ? num(overallBest.row["metrics/mAP50-95(B)"]) : "–", overallBest ? `${humanRun(overallBest.n)}, epoch ${overallBest.row.epoch}` : "Waiting for the first epoch"),
        el("div", { class: "tiles" },
          tile("Candidate", active ? `${finished + (run && run.status === "training" ? 1 : 0)} of ${order.length}` : "–", active ? humanRun(active) : (live.stage_message || "Starting")),
          tile("Epoch", run && last ? `${last.epoch} of ${run.epochs}` : "–", last && last.gpu_mem_gb ? `${last.gpu_mem_gb} GB GPU memory` : null),
          tile("Time per epoch", eta ? duration(eta.perEpoch) : "–", best ? `Best so far: epoch ${best.epoch}` : null),
          tile("Time left", eta && live.status === "running" ? `~${duration(eta.allRuns)}` : "–", eta && live.status === "running" ? `rough estimate; this run ~${duration(eta.thisRun)}` : null),
          tile("GPU", live.env && live.env.gpu_name ? live.env.gpu_name.replace("NVIDIA ", "").replace("Tesla ", "") : "–", live.env ? `PyTorch ${live.env.torch}` : null),
        )),
    );

    const meters = el("div", { class: "card" });
    if (totalEpochs) {
      meters.append(meter("All candidates", `${doneEpochs} of ${totalEpochs} epochs`, doneEpochs / totalEpochs));
    }
    if (run && run.status === "training") {
      const p = run.progress;
      const within = p ? (p.batch / p.of) : 0;
      const frac = run.epochs ? (rows.length + (p && p.epoch > rows.length ? within : 0)) / run.epochs : 0;
      meters.append(meter(humanRun(active), p && p.epoch > rows.length ? `epoch ${p.epoch}, batch ${p.batch} of ${p.of}` : `${rows.length} of ${run.epochs} epochs`, frac));
    }
    meters.append(el("div", { class: "small", style: "margin-top:8px" },
      el("span", { text: live.stage_message || "" }),
      el("span", { text: live.url ? "  ·  " : "" }),
      live.url ? el("a", { href: live.url, target: "_blank", rel: "noopener", text: "Open the kernel on Kaggle" }) : null));
    root.append(meters);

    if (order.length) root.append(...section("Candidates", "Each starts from COCO-pretrained weights and is compared on validation only. The test split stays untouched until one is selected."), runSummaryTable(live.runs, live.selection, order));
    if (order.filter((n) => live.runs[n] && live.runs[n].rows.length).length > 1) root.append(el("div", { style: "margin-top:14px" }, compareChart(live.runs, order)));

    const startedRuns = order.filter((n) => live.runs[n]);
    if (startedRuns.length) {
      const chosen = ui.run && startedRuns.includes(ui.run) ? ui.run : active;
      const select = el("select", {
        "aria-label": "Candidate shown in the charts",
        onchange: (e) => { ui.run = e.target.value; renderTraining(window.__LIVE__); },
      }, startedRuns.map((n) => el("option", { value: n, text: humanRun(n), ...(n === chosen ? { selected: "selected" } : {}) })));
      root.append(...section("Training curves"), el("div", { class: "filters" }, el("span", { text: "Showing" }), select));
      root.append(trainingCharts(live.runs[chosen].rows, chosen, "live", live.runs[chosen].epochs));
    } else {
      root.append(el("div", { class: "empty" }, el("strong", { text: "Waiting for the first epoch" }), live.stage_message || "The kernel is starting up."));
    }

    const log = el("pre", { class: "logbox", tabindex: "0" });
    log.textContent = (live.log_tail || []).slice(-120).join("\n") || "No log output yet.";
    const details = el("details", { open: ui.logOpen ? "open" : null, ontoggle: (e) => { ui.logOpen = e.target.open; } },
      el("summary", { class: "small", style: "cursor:pointer;margin-top:18px", text: "Kernel log (latest lines)" }), log);
    root.append(details);
    if (ui.logOpen) log.scrollTop = log.scrollHeight;
  }

  /* --- render loop: only touches the DOM when the data changed --- */
  let lastLive = null, lastStaticTab = null;
  function renderAll(force) {
    const live = window.__LIVE__ || null;
    const staticData = window.__STATIC__ || null;
    renderChip(live);
    const signature = live ? String(live.monitor_updated) : "";
    if (ui.tab === "training" && (force || signature !== lastLive || isStale(live))) {
      renderTraining(live);
      lastLive = signature;
    }
    if (ui.tab === "performance" && (force || lastStaticTab !== "performance")) {
      renderPerformance(panels.performance, staticData ? staticData.performance : null);
      lastStaticTab = "performance";
    }
    if (ui.tab === "dataset" && (force || lastStaticTab !== "dataset")) {
      renderDataset(panels.dataset, staticData ? staticData.dataset : null);
      lastStaticTab = "dataset";
    }
  }
  function reload(src, then) {
    const s = document.createElement("script");
    s.src = `${src}?t=${Date.now()}`;
    s.onload = () => { s.remove(); then && then(); };
    s.onerror = () => { s.remove(); then && then(); };
    document.head.appendChild(s);
  }
  let lastStaticStamp = null, tickCount = 0;
  function tick() {
    reload("data/live.js", () => renderAll(false));
    // Results appear when the run finishes, so an open page should notice them by itself.
    if (++tickCount % 6 === 0) {
      reload("data/static.js", () => {
        const stamp = window.__STATIC__ ? window.__STATIC__.generated_at : null;
        if (stamp !== lastStaticStamp) {
          lastStaticStamp = stamp;
          lastStaticTab = null;
          renderAll(false);
        }
      });
    }
  }
  lastStaticStamp = window.__STATIC__ ? window.__STATIC__.generated_at : null;
  selectTab(ui.tab in panels ? ui.tab : "training");
  tick();
  setInterval(tick, POLL_MS);
})();
