import { useEffect, useRef } from "react";
import { createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import type { Candle, SignalRec } from "./api";

export default function Chart({
  candles, signals, liveBar,
}: { candles: Candle[]; signals: SignalRec[]; liveBar: Candle | null }) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);

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
    const ro = new ResizeObserver(() => c.applyOptions({ width: el.current!.clientWidth }));
    ro.observe(el.current);
    return () => { ro.disconnect(); c.remove(); chart.current = null; series.current = null; };
  }, []);

  useEffect(() => {
    if (series.current && candles.length) series.current.setData(candles as any);
  }, [candles]);

  useEffect(() => {
    if (series.current && liveBar) series.current.update(liveBar as any);
  }, [liveBar]);

  useEffect(() => {
    if (!series.current) return;
    const markers = signals
      .filter((s) => s.timeframe === "1m")
      .map((s) => {
        const exp = s.phase === "expansion";
        const up = s.direction === "up";
        return {
          time: s.bar_epoch as Time,
          position: (exp ? (up ? "belowBar" : "aboveBar") : "aboveBar") as any,
          color: exp ? (up ? "#26a69a" : "#ef5350") : "#e3b341",
          shape: (exp ? (up ? "arrowUp" : "arrowDown") : "circle") as any,
          text: exp ? "E" : "C",
        };
      })
      .sort((a, b) => (a.time as number) - (b.time as number));
    series.current.setMarkers(markers as any);
  }, [signals]);

  return <div ref={el} style={{ width: "100%" }} />;
}
