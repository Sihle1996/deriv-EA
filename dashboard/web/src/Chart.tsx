import { useEffect, useRef, useState } from "react";
import {
  createChart, type IChartApi, type ISeriesApi, type Time,
} from "lightweight-charts";
import type { AtsOverlay, Candle, SignalRec } from "./api";

type Tip = { x: number; y: number; sig: SignalRec } | null;

export default function Chart({
  candles, signals, liveBar, tf, ats, mode,
}: {
  candles: Candle[]; signals: SignalRec[]; liveBar: Candle | null;
  tf: string; ats: AtsOverlay | null; mode: "live" | "archive";
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlay = useRef<ISeriesApi<"Line">[]>([]);   // ATS boxes + forward value lines (one series each)
  const sigRef = useRef<SignalRec[]>([]);
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
    c.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) { setTip(null); return; }
      const t = param.time as number;
      const s = sigRef.current.find((x) => x.timeframe === tf && x.bar_epoch === t);
      setTip(s ? { x: param.point.x, y: param.point.y, sig: s } : null);
    });
    const ro = new ResizeObserver(() => c.applyOptions({ width: el.current!.clientWidth }));
    ro.observe(el.current);
    return () => { ro.disconnect(); c.remove(); chart.current = null; series.current = null; overlay.current = []; };
  }, []);

  useEffect(() => {
    if (series.current && candles.length) series.current.setData(candles as any);
  }, [candles]);

  useEffect(() => {
    // The WS feed only streams the forming 1m bar — apply it on the live 1m chart only (never in
    // archive mode, where the chart is a static historical view).
    if (series.current && liveBar && tf === "1m" && mode === "live") series.current.update(liveBar as any);
  }, [liveBar, tf, mode]);

  // ATS overlay drawn the TradeATS way: each contraction = a faint BOX (top/bottom over its bars)
  // with a solid VALUE LINE projected forward from it. One short line series per element (avoids the
  // left-edge clamp of a single connected line). Only elements intersecting the window are drawn.
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    for (const s of overlay.current) c.removeSeries(s);
    overlay.current = [];
    if (!candles.length) return;
    const lo = candles[0].time as number, hi = candles[candles.length - 1].time as number;
    const seg = (color: string, width: 1 | 2, t0: number, t1: number, v: number) => {
      if (t1 < lo || t0 > hi || v == null) return;
      const s = c.addLineSeries({
        color, lineWidth: width, priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData([{ time: t0 as Time, value: v }, { time: t1 as Time, value: v }]);
      overlay.current.push(s);
    };
    for (const v of ats?.value_lines ?? []) {
      if (v.tf !== tf) continue;
      seg("#58a6ff", 2, v.box_start, v.line_end, v.value_line);          // value line (point of origin)
      if (v.box_high != null) seg("#2d4a6b", 1, v.box_start, v.box_end, v.box_high);  // box top
      if (v.box_low != null) seg("#2d4a6b", 1, v.box_start, v.box_end, v.box_low);    // box bottom
    }
  }, [ats, tf, candles]);

  // Markers: Phase-2 C/E/T/R (current tf) + ATS pullback entries (on the LTF, distinct purple).
  // Clipped to the visible candle window (same reason as the value line).
  useEffect(() => {
    sigRef.current = signals;
    if (!series.current) return;
    const lo = candles.length ? (candles[0].time as number) : -Infinity;
    const hi = candles.length ? (candles[candles.length - 1].time as number) : Infinity;
    const inRange = (t: number) => t >= lo && t <= hi;
    const markers = signals
      .filter((s) => s.timeframe === tf && inRange(s.bar_epoch))
      .map((s) => {
        const up = s.direction === "up";
        let position = "aboveBar", color = "#e3b341", shape = "circle", text = "C";
        if (s.phase === "expansion") {
          position = up ? "belowBar" : "aboveBar";
          color = up ? "#26a69a" : "#ef5350"; shape = up ? "arrowUp" : "arrowDown"; text = "E";
        } else if (s.phase === "trend") {
          position = up ? "belowBar" : "aboveBar";
          color = up ? "#26a69a" : "#ef5350"; shape = "square"; text = "T";
        } else if (s.phase === "reversal") {
          position = "aboveBar"; color = "#a371f7"; shape = "square"; text = "R";
        }
        return { time: s.bar_epoch as Time, position: position as any, color, shape: shape as any, text };
      });
    // ATS entries render on the LTF chart (that's where pullback entries are taken).
    if (ats && tf === ats.ltf) {
      for (const e of ats.entries) {
        if (!inRange(e.bar_epoch)) continue;
        const up = e.direction === "up";
        markers.push({
          time: e.bar_epoch as Time,
          position: (up ? "belowBar" : "aboveBar") as any,
          color: "#d2a8ff", shape: (up ? "arrowUp" : "arrowDown") as any, text: "ATS",
        });
      }
    }
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    series.current.setMarkers(markers as any);
  }, [signals, ats, tf, candles]);

  const t = tip?.sig;
  return (
    <div className="chartwrap">
      <div ref={el} style={{ width: "100%" }} />
      <div className="legend">
        <span><i className="dot c" />C — contraction (coils)</span>
        <span><i className="arr up" />E — expansion (breakout)</span>
        <span><i className="sq grn" />T — trend (move continued)</span>
        <span><i className="sq rev" />R — reversal (retraced)</span>
        <span><i className="vline" />ATS value line + entries</span>
      </div>
      {t && (
        <div className="tip" style={{ left: tip!.x + 14, top: tip!.y + 8 }}>
          <b>{t.phase}{t.direction ? ` ${t.direction}` : ""}</b> · {t.timeframe}<br />
          price {t.price_at_signal?.toFixed?.(5)}<br />
          bw %ile {t.bw_percentile != null ? (t.bw_percentile * 100).toFixed(0) : "—"} · z {t.bbw_zscore?.toFixed?.(2)}
        </div>
      )}
    </div>
  );
}
