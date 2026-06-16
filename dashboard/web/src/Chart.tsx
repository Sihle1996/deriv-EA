import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import type { AtsEntry, AtsOverlay, Candle } from "./api";

type Tip = { x: number; y: number; e: AtsEntry } | null;

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
    if (tf !== ats.htf && tf !== ats.ltf) return;
    const lo = candles[0].time as number, hi = candles[candles.length - 1].time as number;
    const seg = (color: string, width: 1 | 2, t0: number, t1: number, v: number | null) => {
      if (v == null || t1 < lo || t0 > hi) return;
      const s = c.addLineSeries({
        color, lineWidth: width, priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData([{ time: t0 as Time, value: v }, { time: t1 as Time, value: v }]);
      overlay.current.push(s);
    };
    for (const v of ats.value_lines) {
      if (v.tf !== tf) continue;                                 // value lines for the viewed timeframe
      seg("#58a6ff", 2, v.box_start, v.line_end, v.value_line);  // value line (point of origin)
      seg("#3b6ea5", 1, v.box_start, v.box_end, v.box_high);     // box top
      seg("#3b6ea5", 1, v.box_start, v.box_end, v.box_low);      // box bottom
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
        <span><i className="vline" />value line (point of origin)</span>
        <span><i className="box" />swing contraction box</span>
        <span><i className="arr up" style={{ borderBottomColor: "#d2a8ff" }} />ATS pullback entry ({ats?.ltf ?? "1m"})</span>
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
