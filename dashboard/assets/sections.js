/* Section renderers shared by the dashboard tabs and the generated reports. */
(function (global) {
  "use strict";
  const { el, lineChart, barChart, segmentedBar, confusionMatrix, tile, hero, meter, statusCell, table } = global.Charts;

  const CLASS_COLOR = { WBC: "var(--s1)", RBC: "var(--s2)", Platelets: "var(--s3)" };
  const CLASS_NAMES = ["WBC", "RBC", "Platelets"];
  const pct = (v, d = 1) => (v === null || v === undefined ? "–" : (v * 100).toFixed(d) + "%");
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const int = (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString("en-US"));
  const ci = (c, d = 3) => (c ? `95% CI ${num(c[0], d)}–${num(c[1], d)}` : "");

  function duration(seconds) {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
    return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
  }
  function humanRun(name) {
    return String(name)
      .replace(/^yolo26([nsm])_full$/, (_, size) => `YOLO26${size}, full fine-tune`)
      .replace(/^yolo26([nsm])_frozen$/, (_, size) => `YOLO26${size}, frozen backbone`)
      .replace(/^smoke_yolo26n$/, "YOLO26n, smoke test");
  }
  function section(title, note) {
    return note ? [el("h2", { text: title }), el("p", { class: "section-note", text: note })] : [el("h2", { text: title })];
  }
  function finding(lead, text) {
    return el("p", { class: "finding" }, el("strong", { text: lead + " " }), text);
  }

  /* ---------- training ---------- */
  function series(rows, key, name, color) {
    return {
      name,
      color,
      points: rows.filter((r) => r[key] !== undefined).map((r) => [r.epoch, r[key]]),
    };
  }
  function lossChart(rows, kind, title, subtitle, scope, frame) {
    return lineChart({
      id: `${scope}-loss-${kind}`,
      title,
      subtitle,
      xLabel: "Epoch",
      directLabels: true,
      ...frame,
      series: [
        series(rows, `train/${kind}_loss`, "Train", "var(--s1)"),
        series(rows, `val/${kind}_loss`, "Validation", "var(--s2)"),
      ],
    });
  }
  function bestRow(rows) {
    let best = null;
    for (const r of rows) if (!best || r["metrics/mAP50-95(B)"] > best["metrics/mAP50-95(B)"]) best = r;
    return best;
  }
  function trainingCharts(rows, runName, scope = "", totalEpochs = null) {
    const frame = { xMin: 1, xMax: totalEpochs || Math.max(2, rows.length), integerX: true };
    const best = bestRow(rows);
    const marks = best ? [{ x: best.epoch, label: `best: epoch ${best.epoch}` }] : [];
    const lossKinds = rows.length && rows[0]["train/l1_loss"] !== undefined ? "l1" : "dfl";
    const perClass = (metric, title, subtitle) =>
      lineChart({
        id: `${scope}-class-${metric}`,
        title,
        subtitle,
        xLabel: "Epoch",
        yMin: 0,
        yMax: 1,
        directLabels: true,
        ...frame,
        series: CLASS_NAMES.map((c) => ({
          name: c,
          color: CLASS_COLOR[c],
          points: rows.filter((r) => r.per_class && r.per_class[c]).map((r) => [r.epoch, r.per_class[c][metric]]),
        })),
      });
    return el("div", { class: "grid two" },
      lineChart({
        id: `${scope}-map`, title: "Validation mAP", subtitle: "Higher is better. best.pt is the epoch with the highest mAP50-95.",
        xLabel: "Epoch", yMin: 0, yMax: 1, directLabels: true, marks, ...frame,
        series: [series(rows, "metrics/mAP50-95(B)", "mAP50-95", "var(--s1)"), series(rows, "metrics/mAP50(B)", "mAP50", "var(--s2)")],
      }),
      lineChart({
        id: `${scope}-pr`, title: "Validation precision and recall", subtitle: "Mean over classes, at each class's best-F1 confidence.",
        xLabel: "Epoch", yMin: 0, yMax: 1, directLabels: true, ...frame,
        series: [series(rows, "metrics/precision(B)", "Precision", "var(--s1)"), series(rows, "metrics/recall(B)", "Recall", "var(--s2)")],
      }),
      perClass("ap50_95", "Validation AP50-95 by class", "Each class has its own line, so a weak class can't hide inside the mean."),
      perClass("ap50", "Validation AP50 by class", "AP at IoU 0.50: did the model find the cell at all?"),
      lossChart(rows, "box", "Box loss", "IoU-based localisation loss.", scope, frame),
      lossChart(rows, "cls", "Class loss", "Validation rising while training falls would signal overfitting.", scope, frame),
      lossChart(rows, lossKinds, lossKinds === "l1" ? "L1 box-distance loss" : "DFL loss", "Regression loss on box edge distances.", scope, frame),
      lineChart({
        id: `${scope}-lr`, title: "Learning rate", subtitle: "Cosine schedule after warm-up.", xLabel: "Epoch", format: (v) => v.toExponential(2), zeroBase: false, ...frame,
        series: [series(rows, "lr/pg0", "Learning rate", "var(--s1)")],
      }),
    );
  }
  function runSummaryTable(runs, selection, runOrder) {
    const winner = selection ? selection.winner : null;
    const rows = runOrder.map((name) => {
      const run = runs[name] || {};
      const result = run.result;
      const best = run.rows && run.rows.length ? bestRow(run.rows) : null;
      const statusNode = !run.status
        ? statusCell("muted", "Queued")
        : run.status === "finished"
          ? statusCell("good", name === winner ? "Selected" : "Finished")
          : statusCell("warn", "Training");
      return [
        humanRun(name),
        statusNode,
        result ? num(result.val_map50_95) : best ? num(best["metrics/mAP50-95(B)"]) : "–",
        result ? num(result.val[result.best_head].map50) : best ? num(best["metrics/mAP50(B)"]) : "–",
        result ? (result.best_head === "nms" ? "NMS" : "NMS-free") : "–",
        result ? (result.params / 1e6).toFixed(1) + "M" : "–",
        result ? result.gflops.toFixed(1) : "–",
        result ? duration(result.train_seconds) : "–",
      ];
    });
    return el("div", { class: "card table-card" },
      table(["Candidate", "Status", "Val mAP50-95", "Val mAP50", "Head", "Params", "GFLOPs", "Train time"], rows, {
        numeric: [2, 3, 5, 6, 7],
        highlight: (i) => runOrder[i] === winner,
      }));
  }
  function compareChart(runs, runOrder) {
    const started = runOrder.filter((name) => runs[name] && runs[name].rows && runs[name].rows.length);
    // The candidates end within about 0.01 of each other, invisible on a 0 to 1 axis.
    const zoom = started.some((name) => runs[name].rows.length >= 20);
    return lineChart({
      id: "compare",
      title: "Validation mAP50-95 by candidate",
      subtitle: zoom ? "From epoch 10, on a zoomed axis, because the candidates finish close together." : "All candidates on one axis.",
      xLabel: "Epoch",
      integerX: true,
      directLabels: true,
      ...(zoom ? { xMin: 10, yMin: 0.7, yMax: 0.9 } : {}),
      series: started.map((name) => ({
        name: humanRun(name),
        color: `var(--s${runOrder.indexOf(name) + 1})`,
        points: runs[name].rows.filter((r) => !zoom || r.epoch >= 10).map((r) => [r.epoch, r["metrics/mAP50-95(B)"]]),
      })),
    });
  }

  /* ---------- performance ---------- */
  function perClassTable(p) {
    const rows = CLASS_NAMES.map((c) => {
      const m = p.per_class[c];
      return [c, int(m.instances), num(m.ap50), m.ap50_ci ? `${num(m.ap50_ci[0])}–${num(m.ap50_ci[1])}` : "", num(m.ap50_95), m.ap50_95_ci ? `${num(m.ap50_95_ci[0])}–${num(m.ap50_95_ci[1])}` : "", num(m.precision), num(m.recall), num(m.f1), int(m.tp), int(m.fp), int(m.fn)];
    });
    return el("div", { class: "card table-card" },
      table(["Class", "Cells", "AP50", "95% CI", "AP50-95", "95% CI", "Precision", "Recall", "F1", "TP", "FP", "FN"], rows, { numeric: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] }));
  }
  function bySourceTable(p) {
    const rows = p.by_source.map((s) => [
      s.label, int(s.images),
      `${int(s.instances.WBC)} / ${int(s.instances.RBC)} / ${int(s.instances.Platelets)}`,
      num(s.map50), num(s.map50_95),
      num(s.per_class.WBC.ap50_95), num(s.per_class.RBC.ap50_95),
      s.instances.Platelets >= 5 ? num(s.per_class.Platelets.ap50_95) : `too few (${s.instances.Platelets})`,
      num(s.f1_macro),
    ]);
    return el("div", { class: "card table-card" },
      table(["Source (image size)", "Test images", "Cells WBC / RBC / Platelets", "mAP50", "mAP50-95", "WBC AP50-95", "RBC AP50-95", "Platelet AP50-95", "Macro F1"], rows, { numeric: [1, 2, 3, 4, 5, 6, 7, 8] }));
  }
  function gallery(items) {
    return el("div", { class: "gallery" }, items.map((g) =>
      el("figure", {}, el("img", { src: g.src, alt: g.alt || g.title, loading: "lazy" }), el("figcaption", {}, el("strong", { text: g.title }), g.caption ? g.caption : null))));
  }
  function predictionKey() {
    return el("div", { class: "keyline" },
      el("span", {}, el("span", { class: "box", style: "border-color:#2a78d6" }), "Correct WBC"),
      el("span", {}, el("span", { class: "box", style: "border-color:#eb6834" }), "Correct RBC"),
      el("span", {}, el("span", { class: "box", style: "border-color:#1baf7a" }), "Correct platelet"),
      el("span", {}, el("span", { class: "box", style: "border-color:#ffd400" }), "False positive (no matching label)"),
      el("span", {}, el("span", { class: "box", style: "border-color:#ff1fb4" }), "Missed label (false negative)"));
  }

  function renderPerformance(root, p) {
    root.replaceChildren();
    if (!p) {
      root.append(el("div", { class: "empty" },
        el("strong", { text: "No final results yet" }),
        "Test metrics appear here after the selected model finishes its one evaluation on the test split."));
      return;
    }
    root.append(
      el("div", { class: "hero-row" },
        hero("Test mAP50-95", num(p.overall.map50_95), ci(p.overall.map50_95_ci)),
        el("div", { class: "tiles" },
          tile("Test mAP50", num(p.overall.map50), ci(p.overall.map50_ci)),
          tile("Macro F1", num(p.operating.macro_f1), `at confidence ${num(p.operating.conf, 2)}`),
          tile("Macro precision", num(p.operating.macro_precision), "at that threshold"),
          tile("Macro recall", num(p.operating.macro_recall), "at that threshold"),
          tile("Inference", `${num(p.speed.inference_ms, 1)} ms`, `per image on ${p.speed.device}`),
        )),
    );
    (p.notes || []).forEach((n) => root.append(el("div", { class: `banner ${n.kind || "info"}` }, el("div", {}, el("strong", { text: n.title }), n.text))));
    root.append(
      ...(p.border_comparison ? [
        ...section("How the model must be run", "Ultralytics' validation pads each image with a 16-pixel gray border, but its predict() function does not. Without the border, cells cut off by the image edge are detected much worse, so plain predict() performs below the benchmark. All results on this page use the border, which scripts/predict.py adds automatically. The decision rests on the validation rows. Test is shown only to confirm it."),
        el("div", { class: "card table-card" },
          table(["Input", "Split", "mAP50", "mAP50-95", "Threshold", "Macro F1", "RBC false positives", "RBC missed"],
            [["without_border", "Plain predict()"], ["with_border", "With border"]].flatMap(([key, label]) =>
              ["val", "test"].map((split) => {
                const r = p.border_comparison[key][split];
                return [label, split, num(r.map50), num(r.map50_95), num(r.threshold, 2), num(r.macro_f1), int(r.rbc_fp), int(r.rbc_fn)];
              })),
            { numeric: [2, 3, 4, 5, 6, 7], minWidth: 640, highlight: (i) => i >= 2 })),
      ] : []),
      ...section("Results by class", `Test split: ${int(p.images)} images never used for training or for any decision. Intervals come from ${int(p.bootstrap_samples)} bootstrap resamples of test images. Precision, recall and counts use the confidence threshold chosen on validation (${num(p.operating.conf, 2)}), at IoU 0.50. An AP50 of 0.995 is the highest this calculation can return. Cross-check: Ultralytics' own evaluation of this model gives mAP50 ${num(p.ultralytics_reference.map50)} and mAP50-95 ${num(p.ultralytics_reference.map50_95)}.`),
      perClassTable(p),
      ...section("Where the errors go"),
      el("div", { class: "grid two" },
        confusionMatrix({
          id: "cm", title: "Confusion matrix", subtitle: `Counts at confidence ${num(p.operating.conf, 2)} and IoU 0.50. Percentages are shares of each row.`,
          labels: [...CLASS_NAMES, "Background"], matrix: p.confusion_matrix, skip: (i, j) => i === 3 && j === 3,
        }),
        lineChart({
          id: "f1conf", title: "F1 against confidence threshold", subtitle: `Measured on test for reference. The threshold was chosen on validation (${num(p.operating.conf, 2)}).`,
          xLabel: "Confidence", xMin: 0, xMax: 1, yMin: 0, yMax: 1, height: 250, snapNearest: true,
          marks: [{ x: p.operating.conf, label: `chosen: ${num(p.operating.conf, 2)}` }], tableRows: p.curves.conf_table,
          series: CLASS_NAMES.map((c) => ({ name: c, color: CLASS_COLOR[c], points: p.curves.f1_conf[c] })),
        }),
        lineChart({
          id: "prcurve", title: "Precision-recall curve", subtitle: "At IoU 0.50, one line per class.",
          xLabel: "Recall", xMin: 0, xMax: 1, yMin: 0, yMax: 1, height: 250, snapNearest: true, tableRows: p.curves.recall_table,
          series: CLASS_NAMES.map((c) => ({ name: c, color: CLASS_COLOR[c], points: p.curves.pr[c] })),
        }),
        lineChart({
          id: "apiou", title: "AP as the IoU requirement tightens", subtitle: "How precisely the boxes fit, not just whether cells are found.",
          xLabel: "IoU threshold", xMin: 0.5, xMax: 0.95, yMin: 0, yMax: 1, height: 250, directLabels: true, format: (v) => v.toFixed(3), formatX: (v) => v.toFixed(2),
          series: CLASS_NAMES.map((c) => ({ name: c, color: CLASS_COLOR[c], points: p.curves.ap_by_iou[c] })),
        })),
      ...section("Why predictions count as wrong", "Not every counted error is a model failure. This splits false positives and missed labels by where they fall. RBC false positives are never in empty space: they sit in overlapping clusters, near an existing label, or on cells cut off by the image edge, which are not always labelled."),
      el("div", { class: "card table-card" },
        table(["Class", "False positives", "Near a label", "At an image edge", "Elsewhere", "Missed labels", "Missed at an image edge", "Missed elsewhere"],
          CLASS_NAMES.map((c) => {
            const b = p.error_breakdown[c];
            return [c, int(b.fp), int(b.fp_near_label), int(b.fp_at_edge), int(b.fp_elsewhere), int(b.fn), int(b.fn_at_edge), int(b.fn_elsewhere)];
          }),
          { numeric: [1, 2, 3, 4, 5, 6, 7], minWidth: 720 })),
      ...section("Results by image source", "The dataset combines images from four collections, with different microscopes and stains. Each is evaluated separately so that a strong average can't hide a weak source. Means only include classes with at least 5 labels in that source, and sources with few test images have wide uncertainty."),
      bySourceTable(p),
      ...(p.baselines ? [
        ...section("Context: published baselines", p.baselines.note),
        el("div", { class: "card table-card" },
          table(["Model", "Input", "Split", "mAP50", "mAP50-95", "WBC AP50-95", "RBC AP50-95", "Platelet AP50-95"],
            [
              [`This model (${humanRun(p.winner)})`, "640 px", "Leakage corrected", num(p.overall.map50), num(p.overall.map50_95), num(p.per_class.WBC.ap50_95), num(p.per_class.RBC.ap50_95), num(p.per_class.Platelets.ap50_95)],
              ...p.baselines.rows.map((r) => [r.model, r.input, r.split, num(r.map50), num(r.map50_95), num(r.wbc), num(r.rbc), num(r.platelets)]),
            ],
            { numeric: [3, 4, 5, 6, 7], highlight: (i) => i === 0 })),
      ] : []),
      ...section("Counting accuracy", "Blood-smear analysis often comes down to counts, so this measures the error in cells per image at the chosen threshold."),
      el("div", { class: "card table-card" },
        table(["Class", "Mean true count per image", "Mean predicted count", "Mean absolute error", "Bias (predicted minus true)", "Images counted exactly"],
          CLASS_NAMES.map((c) => { const m = p.counts[c]; return [c, num(m.mean_true, 2), num(m.mean_pred, 2), num(m.mae, 2), (m.bias >= 0 ? "+" : "") + num(m.bias, 2), pct(m.exact, 0)]; }),
          { numeric: [1, 2, 3, 4, 5] })),
    );
    if (p.gallery && p.gallery.length) {
      p.gallery.forEach((g) => {
        root.append(...section(g.title, g.note));
        if (g.key !== false) root.append(predictionKey());
        root.append(gallery(g.items.map((it) => ({ ...it, caption: it.caption }))));
      });
    }
    (p.limitations ? [p.limitations] : []).forEach((items) => {
      root.append(...section("Known limitations"), el("ul", { class: "plain" }, items.map((t) => el("li", { text: t }))));
    });
  }

  /* ---------- dataset ---------- */
  function renderDataset(root, d) {
    root.replaceChildren();
    if (!d) {
      root.append(el("div", { class: "empty" }, el("strong", { text: "No dataset summary yet" }), "Run scripts/build_reports.py to generate it."));
      return;
    }
    const a = d.audit;
    const m = d.manifest;
    root.append(
      el("div", { class: "tiles" },
        tile("Images as provided", int(a.totals.images), "TXL-PBC"),
        tile("Annotated cells", int(a.totals.cells), `${int(a.integrity.duplicate_boxes_total)} were duplicate rows`),
        tile("Classes", String(a.names.length), a.names.join(", ")),
        tile("Image sources", String(d.sources.length), "told apart by image size"),
        tile("Cells per image", `${a.image_stats.cells_per_image.median}`, `median, range ${a.image_stats.cells_per_image.min}–${a.image_stats.cells_per_image.max}`),
      ),
      ...section("Which dataset was used", "Three datasets were provided, but only one can train a detector."),
      el("div", { class: "card table-card" }, table(
        ["Dataset", "Contents", "Labels", "Used?", "Why"],
        d.candidates.map((c) => [c.name, c.contents, c.labels, statusCell(c.used ? "good" : "muted", c.used ? "Yes" : "No"), c.why]),
        { minWidth: 780 },
      )),
      ...section("Composition"),
      el("div", { class: "grid two" },
        segmentedBar({
          id: "class-share", title: "Cells by class", subtitle: `All ${int(a.totals.cells)} annotated cells, across all splits.`, partName: "Class",
          parts: CLASS_NAMES.map((c) => ({ name: c, value: a.totals.per_class[c], color: CLASS_COLOR[c] })),
        }),
        barChart({
          id: "platelet-density", title: "Platelet labels per image, by source",
          subtitle: `About a ${Math.round(Math.max(...d.sources.map((s) => s.platelets_per_image)) / Math.min(...d.sources.map((s) => s.platelets_per_image)))}-fold spread between sources. See the label quality section.`,
          categoryName: "Source", valueName: "Platelets per image", color: CLASS_COLOR.Platelets, format: (v) => v.toFixed(2),
          bars: [...d.sources].sort((x, y) => y.platelets_per_image - x.platelets_per_image).map((s) => ({ name: s.size, value: s.platelets_per_image })),
        })),
      ...section("Splits", "The test split is exactly as provided. The training counts are after the leakage correction described below."),
      el("div", { class: "card table-card" }, table(
        ["Split", "Images provided", "Removed (leakage)", "Images used", "WBC", "RBC", "Platelets", "Cells used"],
        ["train", "val", "test"].map((s) => {
          const used = m.per_split[s];
          return [s, int(a.per_split[s].images), int(used.dropped), int(used.images), int(used.per_class[0]), int(used.per_class[1]), int(used.per_class[2]), int(used.cells)];
        }),
        { numeric: [1, 2, 3, 4, 5, 6, 7] })),
      ...section("Image sources", "File names don't record where an image came from, so sources are identified by image size. The resulting counts match the four collections listed in the TXL-PBC paper exactly. Colour and scale differ strongly between them, which is why the model is also evaluated per source."),
      el("div", { class: "card table-card" }, table(
        ["Source (image size)", "Collection", "Images", "Train / val / test", "WBC", "RBC", "Platelets", "Platelets per image"],
        d.sources.map((s) => [s.size, s.origin, int(s.images), `${s.splits.train || 0} / ${s.splits.val || 0} / ${s.splits.test || 0}`, int(s.WBC), int(s.RBC), int(s.Platelets), s.platelets_per_image.toFixed(2)]),
        { numeric: [2, 3, 4, 5, 6, 7] })),
      el("div", { class: "grid two", style: "margin-top:14px" },
        lineChart({
          id: "box-size", title: "Cell size at the 640 px training resolution", subtitle: "Side of each box, as the square root of its area. Three well-separated scales, and almost nothing tiny.",
          xLabel: "Box side (px)", yMin: 0, height: 240, format: (v) => (v * 100).toFixed(1) + "%", directLabels: false,
          series: CLASS_NAMES.map((c) => ({ name: c, color: CLASS_COLOR[c], points: a.box_size_hist.series[c].map((v, i) => [a.box_size_hist.bin_centres[i], v]) })),
        }),
        el("div", { class: "card table-card" },
          el("h3", { text: "Box size summary" }),
          el("p", { class: "small", text: `Side length at 640 px resolution. Only ${pct(a.box_size_stats.Platelets.frac_under_32px_at_640)} of platelets fall below 32 px, and most source images are smaller than 640 px, so a larger input size would mostly just upsample.` }),
          table(["Class", "5th percentile", "Median", "Under 32 px"],
            CLASS_NAMES.map((c) => { const s = a.box_size_stats[c]; return [c, `${s.side_at_640_p05} px`, `${s.side_at_640_median} px`, pct(s.frac_under_32px_at_640)]; }),
            { numeric: [1, 2, 3] }))),
      ...section("Data checks", "Nothing was assumed from the dataset README. Each row below was verified against the files themselves."),
      el("div", { class: "card table-card" }, table(["Check", "Result", "Detail"], d.checks.map((c) => [c.name, statusCell(c.status, c.label), c.detail]), { minWidth: 620 })),
      ...section("Leakage: the same field of view in more than one split", d.leakage.summary),
      el("div", { class: "tiles" },
        tile("Image pairs compared", int(d.leakage.pairs_compared), "every pair, including flipped"),
        tile("Same-field pairs", int(d.leakage.pairs_total), `${int(d.leakage.pairs_across_splits)} cross a split boundary`),
        tile("Test images with a twin", `${int(d.leakage.test_affected)} of ${int(a.per_split.test.images)}`, "their twins left train and val"),
        tile("Images removed", int(d.leakage.removed), `${int(d.leakage.removed_train)} train, ${int(d.leakage.removed_val)} val`),
      ),
      gallery(d.leakage.examples),
      el("div", { class: "grid two", style: "margin-top:14px" },
        barChart({
          id: "twin-source", title: "Same-field pairs by source", subtitle: "Three of the four sources contain them.", categoryName: "Source", valueName: "Pairs", color: "var(--s1)",
          bars: Object.entries(d.leakage.pairs_by_source).sort((x, y) => y[1] - x[1]).map(([name, value]) => ({ name, value })),
        }),
        barChart({
          id: "twin-calibration", title: "How the same-field threshold was set", subtitle: `Same-source pairs by number of geometrically consistent keypoints. Look-alike nuclei stop at 10, and genuine twins start at ${d.leakage.threshold}, which is the threshold.`,
          categoryName: "Matched keypoints", valueName: "Pairs", color: "var(--s1)", labelWidth: 110,
          bars: d.leakage.calibration.map(([name, value]) => ({ name, value })),
        })),
      ...section("Label quality", "These issues can't be fixed without re-annotation, so they are reported here and in the performance results."),
      ...d.label_findings.map((f) => finding(f.lead, f.text)),
      ...section("Ground-truth samples by source", "Boxes as annotated: blue WBC, orange RBC, green platelets."),
      gallery(d.samples),
    );
  }

  /* ---------- training report ---------- */
  function renderTrainingReport(root, t) {
    root.replaceChildren();
    const args = (t.config && t.config.train_args) || {};
    const order = (t.config && t.config.candidates ? t.config.candidates.map((c) => c.name) : Object.keys(t.runs));
    const finished = order.filter((n) => t.runs[n] && t.runs[n].result);
    const totalSeconds = finished.reduce((sum, n) => sum + t.runs[n].result.train_seconds, 0);
    const winner = t.selection ? t.selection.winner : null;
    const winnerResult = winner && t.runs[winner] ? t.runs[winner].result : null;
    root.append(
      el("div", { class: "hero-row" },
        hero("Selected model, validation mAP50-95", winnerResult ? num(winnerResult.val_map50_95) : "–", winner ? humanRun(winner) : "No model selected yet"),
        el("div", { class: "tiles" },
          tile("Candidates trained", String(finished.length), `${args.epochs || "–"} epochs each`),
          tile("Total GPU training time", duration(totalSeconds), t.env && t.env.gpu_name ? t.env.gpu_name : ""),
          tile("Training images", t.dataset && t.dataset.counts ? int(t.dataset.counts.train) : "–", t.dataset && t.dataset.counts ? `${int(t.dataset.counts.val)} for validation` : ""),
          tile("Input size", args.imgsz ? `${args.imgsz} px` : "–", `batch ${args.batch || "–"}`),
        )),
    );
    (t.observations || []).length && root.append(...section("What the training shows"), ...t.observations.map((o) => finding(o.lead, o.text)));
    root.append(
      ...section("Candidates", `All start from COCO-pretrained weights and are compared on the validation split only. ${t.selection ? `Rule: the smallest model within ${t.selection.margin} of the best validation mAP50-95 wins, because smaller gaps are within noise.` : ""}`),
      runSummaryTable(t.runs, t.selection, order),
    );
    if (finished.length > 1) root.append(el("div", { style: "margin-top:14px" }, compareChart(t.runs, order)));
    const withRows = order.filter((n) => t.runs[n] && t.runs[n].rows.length);
    const primary = winner && withRows.includes(winner) ? winner : withRows[0];
    if (primary) root.append(...section(`Training curves: ${humanRun(primary)}${primary === winner ? " (selected)" : ""}`), trainingCharts(t.runs[primary].rows, primary, primary, t.runs[primary].epochs));
    const others = withRows.filter((n) => n !== primary);
    if (others.length) {
      root.append(...section("Training curves: other candidates"));
      for (const name of others) {
        root.append(el("details", { style: "margin:8px 0" },
          el("summary", { style: "cursor:pointer;font-weight:600;font-size:14px;margin-bottom:10px", text: humanRun(name) }),
          trainingCharts(t.runs[name].rows, name, name, t.runs[name].epochs)));
      }
    }
    root.append(...section("Settings"), el("div", { class: "card table-card" },
      table(["Setting", "Value"], Object.entries(args).map(([k, v]) => [k, String(v)]))));
    if (t.env) {
      root.append(...section("Environment"), el("div", { class: "card table-card" },
        table(["Item", "Value"], [
          ["GPU", `${t.env.gpu_name || "–"} (${t.env.gpu_memory_gb || "–"} GB)`],
          ["PyTorch", t.env.torch], ["Ultralytics", t.env.ultralytics], ["Python", t.env.python],
          ["Kaggle kernel (private)", t.kernel],
        ])));
    }
  }

  global.Sections = { renderTrainingReport, renderPerformance, renderDataset, trainingCharts, runSummaryTable, compareChart, bestRow, humanRun, duration, section, num, int, pct, finding };
})(window);
