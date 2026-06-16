import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import type { Candle, SignalRec } from "./api";

type Tip = { x: number; y: number; sig: SignalRec } | null;

export default function Chart({
  candles, signals, liveBar, tf,
}: { candles: Candle[]; signals: SignalRec[]; liveBar: Candle | null; tf: string }) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
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
    // Hover tooltip: when the crosshair is over a bar that has a signal, show its details.
    c.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) { setTip(null); return; }
      const t = param.time as number;
      const s = sigRef.current.find((x) => x.timeframe === "1m" && x.bar_epoch === t);
      setTip(s ? { x: param.point.x, y: param.point.y, sig: s } : null);
    });
    const ro = new ResizeObserver(() => c.applyOptions({ width: el.current!.clientWidth }));
    ro.observe(el.current);
    return () => { ro.disconnect(); c.remove(); chart.current = null; series.current = null; };
  }, []);

  useEffect(() => {
    if (series.current && candles.length) series.current.setData(candles as any);
  }, [candles]);

  useEffect(() => {
    // The WS feed only streams the forming 1m bar — only apply it on the 1m chart; higher
    // timeframes refresh via the periodic /api/candles poll instead.
    if (series.current && liveBar && tf === "1m") series.current.update(liveBar as any);
  }, [liveBar, tf]);

  useEffect(() => {
    sigRef.current = signals;
    if (!series.current) return;
    const markers = signals
      .filter((s) => s.timeframe === tf)   // signals exist on 1m/5m; higher TFs show none (context only)
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
      })
      .sort((a, b) => (a.time as number) - (b.time as number));
    series.current.setMarkers(markers as any);
  }, [signals, tf]);

  const t = tip?.sig;
  return (
    <div className="chartwrap">
      <div ref={el} style={{ width: "100%" }} />
      <div className="legend">
        <span><i className="dot c" />C — contraction (coils)</span>
        <span><i className="arr up" />E — expansion (breakout)</span>
        <span><i className="sq grn" />T — trend (move continued)</span>
        <span><i className="sq rev" />R — reversal (retraced)</span>
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
