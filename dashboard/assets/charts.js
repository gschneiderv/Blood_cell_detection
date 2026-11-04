/* Small SVG chart kit with no dependencies, so the dashboard works offline from disk.
   Shared by the dashboard and the generated reports. Labels always go through
   textContent because run and class names originate outside this page. */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === undefined || v === null) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "style") node.style.cssText = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid);
    }
    return node;
  }
  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  /* --- one shared tooltip --- */
  let tip;
  function tooltip() {
    if (!tip) {
      tip = el("div", { class: "tooltip", role: "status" });
      document.body.appendChild(tip);
    }
    return tip;
  }
  function placeTip(x, y) {
    const t = tooltip();
    t.style.display = "block";
    const r = t.getBoundingClientRect();
    const px = Math.min(Math.max(8, x + 14), window.innerWidth - r.width - 8);
    const py = Math.min(Math.max(8, y + 14), window.innerHeight - r.height - 8);
    t.style.left = px + "px";
    t.style.top = py + "px";
  }
  function hideTip() {
    if (tip) tip.style.display = "none";
  }
  function fillTip(head, rows) {
    const t = tooltip();
    t.replaceChildren();
    if (head !== null && head !== undefined) t.append(el("div", { class: "tt-head", text: head }));
    for (const r of rows) {
      t.append(
        el(
          "div",
          { class: "tt-row" },
          r.color ? el("span", { class: "tt-key", style: `background:${r.color}` }) : null,
          el("span", { class: "tt-val", text: r.value }),
          el("span", { class: "tt-name", text: r.name })
        )
      );
    }
  }

  /* --- chart frame: title, table toggle, legend, plot, table view --- */
  const tableOpen = {};
  function frame(opts) {
    const id = opts.id || opts.title;
    const plot = el("div", { class: "plot" });
    const tableWrap = el("div", { class: "table-wrap", hidden: "hidden" });
    const button = el("button", {
      class: "ghost-btn", type: "button", text: "Table", "aria-pressed": "false",
      onclick: () => setOpen(!tableOpen[id]),
    });
    function setOpen(open) {
      tableOpen[id] = open;
      tableWrap.hidden = !open;
      button.setAttribute("aria-pressed", String(open));
    }
    const caption = el(
      "figcaption", {},
      el("div", {}, el("h3", { text: opts.title }), opts.subtitle ? el("div", { class: "sub", text: opts.subtitle }) : null),
      opts.table ? button : null
    );
    const legend = el("div", { class: "legend" });
    const figure = el("figure", { class: "chart" }, caption, legend, plot, tableWrap);
    if (tableOpen[id]) setOpen(true);
    return { figure, plot, legend, tableWrap };
  }
  function renderLegend(node, items, kind) {
    node.replaceChildren();
    if (items.length < 2) {
      node.style.display = "none";
      return;
    }
    for (const it of items) {
      node.append(
        el("span", { class: "key" },
          el("span", { class: kind === "rect" ? "swatch-rect" : "swatch-line", style: `background:${it.color}` }),
          el("span", { text: it.name }))
      );
    }
  }
  function dataTable(headers, rows, numericFrom = 1) {
    return el("table", { class: "data" },
      el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i >= numericFrom ? "n" : null, text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i >= numericFrom ? "n" : null, text: c })))))
    );
  }

  /* --- scales --- */
  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const norm = raw / mag;
    return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  }
  function ticks(min, max, count) {
    if (!(max > min)) return [min];
    const step = niceStep(max - min, count);
    const out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmtTick(v, step) {
    const digits = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
    if (Math.abs(v) >= 10000) return (v / 1000).toLocaleString("en-US", { maximumFractionDigits: 1 }) + "K";
    return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  /* --- line chart --- */
  function lineChart(opts) {
    const f = frame({ ...opts, table: true });
    const series = opts.series.filter((s) => s.points && s.points.length);
    renderLegend(f.legend, series, "line");
    const fmt = opts.format || ((v) => v.toFixed(3));
    const fmtX = opts.formatX || ((v) => String(v));
    const xName = opts.xLabel || "x";

    const allX = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
    f.tableWrap.append(
      dataTable([xName, ...series.map((s) => s.name)],
        (opts.tableRows ? opts.tableRows : allX).map((x) => [
          fmtX(x),
          ...series.map((s) => {
            const hit = s.points.find((p) => p[0] === x);
            return hit && hit[1] !== null ? fmt(hit[1]) : "";
          }),
        ]))
    );

    function draw() {
      const width = Math.max(240, f.plot.clientWidth || 480);
      const height = opts.height || 210;
      const hasDirect = opts.directLabels && series.length && series.length <= 4;
      const m = { top: 8, right: hasDirect ? 76 : 14, bottom: opts.xLabel ? 38 : 24, left: 44 };
      const iw = width - m.left - m.right;
      const ih = height - m.top - m.bottom;
      f.plot.replaceChildren();
      const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.title });
      f.plot.append(svg);
      if (!series.length) {
        const t = svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, svg);
        t.textContent = opts.emptyText || "Waiting for data";
        return;
      }
      const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v !== null && Number.isFinite(v));
      let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...ys);
      let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...ys);
      if (opts.yMin === undefined && Math.min(...ys) > 0 && opts.zeroBase !== false) yMin = 0;
      if (yMax === yMin) yMax = yMin + (yMin === 0 ? 1 : Math.abs(yMin) * 0.1);
      const yt = ticks(yMin, yMax, 4);
      if (opts.yMax === undefined) yMax = Math.max(yMax, yt[yt.length - 1]);
      const xMin = opts.xMin !== undefined ? opts.xMin : allX[0];
      const xMax = opts.xMax !== undefined ? opts.xMax : Math.max(allX[allX.length - 1], xMin + 1);
      const X = (v) => m.left + ((v - xMin) / (xMax - xMin || 1)) * iw;
      const Y = (v) => m.top + ih - ((v - yMin) / (yMax - yMin || 1)) * ih;

      const stepY = yt.length > 1 ? yt[1] - yt[0] : 1;
      for (const t of yt) {
        svgEl("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t), style: "stroke:var(--grid);stroke-width:1" }, svg);
        const label = svgEl("text", { x: m.left - 7, y: Y(t) + 4, "text-anchor": "end" }, svg);
        label.textContent = fmtTick(t, stepY);
      }
      svgEl("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih, style: "stroke:var(--axis);stroke-width:1" }, svg);
      const xt = ticks(xMin, xMax, Math.max(3, Math.floor(iw / 70))).filter((t) => !opts.integerX || Number.isInteger(t));
      const stepX = opts.integerX ? 1 : xt.length > 1 ? xt[1] - xt[0] : 1;
      for (const t of xt) {
        const label = svgEl("text", { x: X(t), y: m.top + ih + 15, "text-anchor": "middle" }, svg);
        label.textContent = fmtTick(t, stepX);
      }
      if (opts.xLabel) {
        const label = svgEl("text", { x: m.left + iw / 2, y: height - 5, "text-anchor": "middle" }, svg);
        label.textContent = opts.xLabel;
      }

      for (const mark of opts.marks || []) {
        if (mark.x < xMin || mark.x > xMax) continue;
        svgEl("line", { x1: X(mark.x), x2: X(mark.x), y1: m.top, y2: m.top + ih, style: "stroke:var(--axis);stroke-width:1" }, svg);
        const nearRight = X(mark.x) > m.left + iw - 80;
        // Near the baseline, because metric lines tend to crowd the top of these charts.
        const label = svgEl("text", { x: X(mark.x) + (nearRight ? -5 : 5), y: m.top + ih - 6, "text-anchor": nearRight ? "end" : "start" }, svg);
        label.textContent = mark.label;
      }

      series.forEach((s) => {
        const pts = s.points.filter((p) => p[1] !== null && Number.isFinite(p[1]));
        const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join("");
        svgEl("path", { d, style: `fill:none;stroke:${s.color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round` }, svg);
        if (pts.length) {
          const last = pts[pts.length - 1];
          svgEl("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2` }, svg);
        }
      });

      if (hasDirect) {
        const ends = series
          .map((s) => {
            const pts = s.points.filter((p) => p[1] !== null && Number.isFinite(p[1]));
            const last = pts[pts.length - 1];
            return last ? { name: s.name, value: last[1], y: Y(last[1]), x: X(last[0]) } : null;
          })
          .filter(Boolean)
          .sort((a, b) => a.y - b.y);
        const collides = ends.some((e, i) => i && e.y - ends[i - 1].y < 13);
        if (!collides) {
          for (const e of ends) {
            const t = svgEl("text", { x: e.x + 8, y: e.y + 4, class: "direct" }, svg);
            t.textContent = fmt(e.value);
          }
        }
      }

      const cross = svgEl("line", { y1: m.top, y2: m.top + ih, style: "stroke:var(--axis-text);stroke-width:1;display:none" }, svg);
      const hit = svgEl("rect", { x: m.left, y: m.top, width: iw, height: ih + 16, tabindex: 0, style: "fill:transparent;outline:none" }, svg);
      const dots = series.map((s) => svgEl("circle", { r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2;display:none` }, svg));
      function nearestX(px) {
        const value = xMin + ((px - m.left) / (iw || 1)) * (xMax - xMin);
        let best = allX[0];
        for (const x of allX) if (Math.abs(x - value) < Math.abs(best - value)) best = x;
        return best;
      }
      function nearestPoint(points, x) {
        let best = null;
        for (const p of points) if (best === null || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
        return best;
      }
      function show(x, cx, cy) {
        cross.setAttribute("x1", X(x));
        cross.setAttribute("x2", X(x));
        cross.style.display = "";
        const rows = [];
        series.forEach((s, i) => {
          const p = opts.snapNearest ? nearestPoint(s.points, x) : s.points.find((q) => q[0] === x);
          if (p && p[1] !== null) {
            dots[i].setAttribute("cx", X(p[0]));
            dots[i].setAttribute("cy", Y(p[1]));
            dots[i].style.display = "";
            rows.push({ color: s.color, value: fmt(p[1]), name: s.name });
          } else {
            dots[i].style.display = "none";
          }
        });
        fillTip(`${xName} ${fmtX(x)}`, rows);
        placeTip(cx, cy);
      }
      function clear() {
        cross.style.display = "none";
        dots.forEach((d) => (d.style.display = "none"));
        hideTip();
      }
      hit.addEventListener("pointermove", (e) => {
        const rect = svg.getBoundingClientRect();
        show(nearestX(e.clientX - rect.left), e.clientX, e.clientY);
      });
      hit.addEventListener("pointerleave", clear);
      let focusIndex = allX.length - 1;
      hit.addEventListener("focus", () => {
        const rect = svg.getBoundingClientRect();
        show(allX[focusIndex], rect.left + X(allX[focusIndex]), rect.top + m.top);
      });
      hit.addEventListener("blur", clear);
      hit.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault();
        focusIndex = Math.min(allX.length - 1, Math.max(0, focusIndex + (e.key === "ArrowRight" ? 1 : -1)));
        const rect = svg.getBoundingClientRect();
        show(allX[focusIndex], rect.left + X(allX[focusIndex]), rect.top + m.top);
      });
    }
    observe(f.plot, draw);
    return f.figure;
  }

  /* Redraw on container resize; drawing happens once the node is attached and sized. */
  function observe(node, draw) {
    let lastWidth = -1;
    const ro = new ResizeObserver(() => {
      const w = node.clientWidth;
      if (w && w !== lastWidth) {
        lastWidth = w;
        requestAnimationFrame(draw); // drawing inside the callback would re-trigger the observer
      }
    });
    ro.observe(node);
  }

  /* --- horizontal bars with the value at the tip --- */
  function barChart(opts) {
    const f = frame({ ...opts, table: true });
    f.legend.style.display = "none";
    const fmt = opts.format || ((v) => v.toLocaleString("en-US"));
    f.tableWrap.append(dataTable([opts.categoryName || "Category", opts.valueName || "Value"], opts.bars.map((b) => [b.name, fmt(b.value)])));
    function draw() {
      const width = Math.max(240, f.plot.clientWidth || 480);
      const rowH = 30;
      const labelW = Math.min(width * 0.36, opts.labelWidth || 130);
      const tipW = 64;
      const height = opts.bars.length * rowH + 4;
      const iw = width - labelW - tipW;
      const max = opts.max !== undefined ? opts.max : Math.max(...opts.bars.map((b) => b.value), 1e-9);
      f.plot.replaceChildren();
      const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.title });
      f.plot.append(svg);
      svgEl("line", { x1: labelW, x2: labelW, y1: 0, y2: height, style: "stroke:var(--axis);stroke-width:1" }, svg);
      opts.bars.forEach((b, i) => {
        const y = i * rowH + 7;
        const w = Math.max(b.value > 0 ? 2 : 0, (b.value / max) * iw);
        const t = svgEl("text", { x: labelW - 8, y: y + 12, "text-anchor": "end", class: "direct" }, svg);
        t.textContent = b.name;
        const r = Math.min(4, w / 2);
        const bh = 16;
        const d = `M${labelW} ${y}h${w - r}a${r} ${r} 0 0 1 ${r} ${r}v${bh - 2 * r}a${r} ${r} 0 0 1 -${r} ${r}h-${w - r}z`;
        const bar = svgEl("path", { d, style: `fill:${b.color || opts.color || "var(--s1)"}` }, svg);
        const v = svgEl("text", { x: labelW + w + 6, y: y + 12, class: "direct" }, svg);
        v.textContent = fmt(b.value);
        const hit = svgEl("rect", { x: 0, y: i * rowH, width, height: rowH, tabindex: 0, style: "fill:transparent;outline:none" }, svg);
        const enter = (cx, cy) => {
          bar.style.opacity = "0.8";
          fillTip(b.name, [{ color: b.color || opts.color || "var(--s1)", value: fmt(b.value), name: b.note || opts.valueName || "" }]);
          placeTip(cx, cy);
        };
        hit.addEventListener("pointermove", (e) => enter(e.clientX, e.clientY));
        hit.addEventListener("pointerleave", () => { bar.style.opacity = ""; hideTip(); });
        hit.addEventListener("focus", () => { const r2 = hit.getBoundingClientRect(); enter(r2.left + labelW, r2.top); });
        hit.addEventListener("blur", () => { bar.style.opacity = ""; hideTip(); });
      });
    }
    observe(f.plot, draw);
    return f.figure;
  }

  /* --- tall segmented bar: one whole split into parts --- */
  function segmentedBar(opts) {
    const f = frame({ ...opts, table: true });
    renderLegend(f.legend, opts.parts, "rect");
    const total = opts.parts.reduce((a, p) => a + p.value, 0) || 1;
    f.tableWrap.append(
      dataTable([opts.partName || "Part", "Count", "Share"], opts.parts.map((p) => [p.name, p.value.toLocaleString("en-US"), ((100 * p.value) / total).toFixed(1) + "%"]))
    );
    const bar = el("div", { class: "segbar", role: "img", "aria-label": opts.parts.map((p) => `${p.name} ${((100 * p.value) / total).toFixed(1)}%`).join(", ") });
    for (const p of opts.parts) {
      const share = (100 * p.value) / total;
      const seg = el("div", { class: "seg", style: `flex:${p.value} 1 0;background:${p.color}`, tabindex: 0 });
      if (share >= 9) seg.textContent = share.toFixed(1) + "%";
      const enter = (cx, cy) => {
        fillTip(p.name, [{ color: p.color, value: p.value.toLocaleString("en-US"), name: `${share.toFixed(1)}%` }]);
        placeTip(cx, cy);
      };
      seg.addEventListener("pointermove", (e) => enter(e.clientX, e.clientY));
      seg.addEventListener("pointerleave", hideTip);
      seg.addEventListener("focus", () => { const r = seg.getBoundingClientRect(); enter(r.left, r.top); });
      seg.addEventListener("blur", hideTip);
      bar.append(seg);
    }
    f.plot.append(bar);
    return f.figure;
  }

  /* --- confusion matrix heatmap (sequential one-hue ramp, row-normalised) --- */
  function confusionMatrix(opts) {
    const f = frame({ ...opts, table: false });
    f.legend.style.display = "none";
    const labels = opts.labels;
    const m = opts.matrix; // m[trueIdx][predIdx]
    const grid = el("div", { class: "heat", style: `grid-template-columns:minmax(70px,auto) repeat(${labels.length},minmax(0,1fr))` });
    grid.append(el("div", { class: "hd rowhd", text: "True ↓  Pred →" }));
    labels.forEach((l) => grid.append(el("div", { class: "hd", text: l })));
    const ramp = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5", "--seq-6", "--seq-7"];
    labels.forEach((rowLabel, i) => {
      grid.append(el("div", { class: "hd rowhd", text: rowLabel }));
      const rowTotal = m[i].reduce((a, b) => a + b, 0);
      labels.forEach((colLabel, j) => {
        const count = m[i][j];
        if (opts.skip && opts.skip(i, j)) {
          grid.append(el("div", { class: "cell", style: "background:transparent" }));
          return;
        }
        const share = rowTotal ? count / rowTotal : 0;
        const step = count === 0 ? 0 : Math.max(1, Math.min(7, Math.ceil(Math.pow(share, 0.5) * 7)));
        const lightMode = getComputedStyle(document.documentElement).colorScheme !== "dark";
        const inkOnDark = lightMode ? step >= 4 : step <= 4 && step > 0;
        const inkColor = step === 0 ? "var(--ink-2)" : inkOnDark ? "#fff" : lightMode ? "#0b0b0b" : "#0b0b0b";
        const cell = el("div", { class: "cell", tabindex: 0, style: `background:var(${ramp[step]});color:${inkColor}` },
          el("span", { class: "v", text: count.toLocaleString("en-US") }),
          rowTotal ? el("span", { class: "p", text: (share * 100).toFixed(share < 0.1 && share > 0 ? 1 : 0) + "%" }) : null);
        const enter = (cx, cy) => {
          fillTip(`True ${rowLabel}, predicted ${colLabel}`, [{ value: count.toLocaleString("en-US"), name: rowTotal ? `${(share * 100).toFixed(1)}% of true ${rowLabel}` : "" }]);
          placeTip(cx, cy);
        };
        cell.addEventListener("pointermove", (e) => enter(e.clientX, e.clientY));
        cell.addEventListener("pointerleave", hideTip);
        cell.addEventListener("focus", () => { const r = cell.getBoundingClientRect(); enter(r.left, r.top); });
        cell.addEventListener("blur", hideTip);
        grid.append(cell);
      });
    });
    f.plot.append(grid);
    f.plot.append(el("div", { class: "scale" }, el("span", { text: "0%" }), el("span", { class: "ramp" }), el("span", { text: "100% of the row" })));
    return f.figure;
  }

  /* --- figures --- */
  function tile(label, value, sub) {
    return el("div", { class: "tile" }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), sub ? el("div", { class: "sub", text: sub }) : null);
  }
  function hero(label, value, sub) {
    return el("div", { class: "hero" }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), sub ? el("div", { class: "sub", text: sub }) : null);
  }
  function meter(label, right, fraction) {
    const pct = Math.max(0, Math.min(1, fraction || 0)) * 100;
    return el("div", { class: "meter", role: "progressbar", "aria-valuenow": pct.toFixed(0), "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": label },
      el("div", { class: "meter-label" }, el("span", { text: label }), el("span", { text: right })),
      el("div", { class: "track" }, el("div", { class: "bar", style: `width:${pct.toFixed(1)}%` })));
  }
  const ICONS = { good: "✓", warn: "!", bad: "✕", muted: "–" };
  function statusCell(kind, label) {
    return el("span", { class: `status-cell ${kind}` }, el("span", { "aria-hidden": "true", text: ICONS[kind] || "" }), el("span", { text: label }));
  }
  function table(headers, rows, opts = {}) {
    const numeric = opts.numeric || [];
    return el("table", { class: "data", style: opts.minWidth ? `min-width:${opts.minWidth}px` : null },
      el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: numeric.includes(i) ? "n" : null, text: h })))),
      el("tbody", {}, rows.map((r, ri) =>
        el("tr", { class: opts.highlight && opts.highlight(ri) ? "highlight" : null },
          r.map((c, i) => {
            const td = el("td", { class: numeric.includes(i) ? "n" : null });
            if (c instanceof Node) td.append(c);
            else td.textContent = c === null || c === undefined ? "" : String(c);
            return td;
          })))));
  }

  global.Charts = { el, lineChart, barChart, segmentedBar, confusionMatrix, tile, hero, meter, statusCell, table };
})(window);
