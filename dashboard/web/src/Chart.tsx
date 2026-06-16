import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import type { AtsEntry, AtsOverlay, Candle } from "./api";

type Tip = { x: number; y: number; e: AtsEntry } | null;

// Per-timeframe value-line styling for the "global view": higher timeframes draw bolder/brighter.
const TF_SECS: Record<string, number> = {
  "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
};
const TF_STYLE: Record<string, { color: string; width: 1 | 2 | 3 | 4 }> = {
  "1m": { color: "#3b6ea5", width: 1 }, "5m": { color: "#58a6ff", width: 2 },
  "15m": { color: "#d29922", width: 2 }, "30m": { color: "#f0883e", width: 3 },
  "1h": { color: "#f0883e", width: 3 }, "4h": { color: "#db61a2", width: 3 },
  "1d": { color: "#e3b341", width: 4 },
};

export default function Chart({
  candles, liveBar, tf, ats, mode,
}: {
  candles: Candle[]; liveBar: Candle | null;
  tf: string; ats: AtsOverlay | null; mode: "live" | "archive" | "deep";
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlay = useRef<ISeriesApi<"Line">[]>([]);   // ATS boxes + forward value lines (one each)
  const entriesRef = useRef<AtsEntry[]>([]);
  const [tip, setTip] = useState<Tip>(null);

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      height: 440,
      layout: { background: { color: "#0e1117" }, textColor: "#c9d1d9" },
      grid: { vertLines: { color: "#1b2230" }, horzLines: { color: "#1b2230" } },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: "#1b2230" },
    });
    chart.current = c;
    series.current = c.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
      wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
    // Hover tooltip: when the crosshair is over an ATS entry bar, show its details.
    c.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) { setTip(null); return; }
      const t = param.time as number;
      const e = entriesRef.current.find((x) => x.bar_epoch === t);
      setTip(e ? { x: param.point.x, y: param.point.y, e } : null);
    });
    const ro = new ResizeObserver(() => c.applyOptions({ width: el.current!.clientWidth }));
    ro.observe(el.current);
    return () => { ro.disconnect(); c.remove(); chart.current = null; series.current = null; overlay.current = []; };
  }, []);

  useEffect(() => {
    if (series.current && candles.length) series.current.setData(candles as any);
  }, [candles]);

  useEffect(() => {
    // The WS feed only streams the forming 1m bar — apply it on the live 1m chart only.
    if (series.current && liveBar && tf === "1m" && mode === "live") series.current.update(liveBar as any);
  }, [liveBar, tf, mode]);

  // ATS overlay (TradeATS "global view"): each swing-pivot contraction = a box (top/bottom over its
  // bars) + a VALUE LINE projected forward. Drawn for the timeframe being viewed (the pivot detector
  // is selective, so these are meaningful swing compressions, not noise). One short line series per
  // element (no left-edge clamp); only elements intersecting the visible window are drawn.
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    for (const s of overlay.current) c.removeSeries(s);
    overlay.current = [];
    if (!candles.length || !ats) return;
    const lo = candles[0].time as number, hi = candles[candles.length - 1].time as number;
    const cur = TF_SECS[tf] ?? 60;
    // value line = a reference level: clamp it to the visible window so higher-TF equilibria span it.
    const line = (color: string, width: 1 | 2 | 3 | 4, t0: number, t1: number, v: number | null) => {
      if (v == null) return;
      const a = Math.max(t0, lo), b = Math.min(t1, hi);
      if (b <= a) return;
      const s = c.addLineSeries({ color, lineWidth: width, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData([{ time: a as Time, value: v }, { time: b as Time, value: v }]);
      overlay.current.push(s);
    };
    // box = an exact rectangle at real times (viewed TF only, to avoid clutter).
    const box = (t0: number, t1: number, v: number | null) => {
      if (v == null || t1 < lo || t0 > hi) return;
      const s = c.addLineSeries({ color: "#3b6ea5", lineWidth: 1, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData([{ time: t0 as Time, value: v }, { time: t1 as Time, value: v }]);
      overlay.current.push(s);
    };
    // The ATS "global view": draw the value line for the VIEWED TF and every HIGHER TF (color-coded,
    // higher = bolder), so e.g. on 1m you see the 1m + 5m + 15m + 1h equilibria stacked together.
    for (const v of ats.value_lines) {
      if ((TF_SECS[v.tf] ?? 60) < cur) continue;
      const st = TF_STYLE[v.tf] ?? { color: "#58a6ff", width: 2 as const };
      line(st.color, st.width, v.box_start, v.line_end, v.value_line);
      if (v.tf === tf) { box(v.box_start, v.box_end, v.box_high); box(v.box_start, v.box_end, v.box_low); }
    }
  }, [ats, tf, candles]);

  // ATS pullback entries (purple arrows) — on the LTF chart, clipped to the visible window.
  useEffect(() => {
    if (!series.current) return;
    const lo = candles.length ? (candles[0].time as number) : -Infinity;
    const hi = candles.length ? (candles[candles.length - 1].time as number) : Infinity;
    const entries = (ats && tf === ats.ltf) ? ats.entries : [];
    entriesRef.current = entries;
    const markers = entries
      .filter((e) => e.bar_epoch >= lo && e.bar_epoch <= hi)
      .map((e) => {
        const up = e.direction === "up";
        return {
          time: e.bar_epoch as Time, position: (up ? "belowBar" : "aboveBar") as any,
          color: "#d2a8ff", shape: (up ? "arrowUp" : "arrowDown") as any, text: "ATS",
        };
      })
      .sort((a, b) => (a.time as number) - (b.time as number));
    series.current.setMarkers(markers as any);
  }, [ats, tf, candles]);

  const e = tip?.e;
  return (
    <div className="chartwrap">
      <div ref={el} style={{ width: "100%" }} />
      <div className="legend">
        <span><i style={{ background: "#3b6ea5" }} />1m</span>
        <span><i style={{ background: "#58a6ff" }} />5m</span>
        <span><i style={{ background: "#d29922" }} />15m</span>
        <span><i style={{ background: "#f0883e" }} />1h value lines (higher TF = bolder)</span>
        <span><i className="box" />contraction box (viewed TF)</span>
        <span><i className="arr up" style={{ borderBottomColor: "#d2a8ff" }} />ATS entry ({ats?.ltf ?? "1m"})</span>
      </div>
      {e && (
        <div className="tip" style={{ left: tip!.x + 14, top: tip!.y + 8 }}>
          <b>ATS entry {e.direction}</b> · {e.tf}<br />
          price {e.price?.toFixed?.(5)}<br />
          value {e.value_line?.toFixed?.(5)} · HTF bias {e.htf_bias ?? "—"}
        </div>
      )}
    </div>
  );
}
