import { useEffect, useRef } from "react";

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type SignalRec = {
  timeframe: string; phase: string; direction: string | null; bar_epoch: number;
  price_at_signal: number; bw_percentile: number; bbw_zscore: number; [k: string]: any;
};
export type Health = {
  symbol: string; ticks: number; signals: number; last_tick_age_s: number | null;
  coverage_pct: number | null; gaps: number | null; live: boolean;
};
export type Backtest = {
  error?: string; verdict?: string; caveat?: string; breakeven?: number;
  real?: { win_rate: number; total_pnl: number; roi_pct: number; n: number };
  null?: { win_rate: number; total_pnl: number; roi_pct: number; n: number } | null;
  trend_n?: number; reversal_n?: number; trend_continuation?: number | null;
};

const j = (url: string) => fetch(url).then((r) => r.json());

export const getSymbols = (): Promise<{ symbol: string; live: boolean }[]> => j("/api/symbols");
export const getCandles = (s: string, tf = "1m", count = 500): Promise<Candle[]> =>
  j(`/api/candles?symbol=${s}&tf=${tf}&count=${count}`);
export const getSignals = (s: string, limit = 100): Promise<SignalRec[]> =>
  j(`/api/signals?symbol=${s}&limit=${limit}`);
export const getBacktest = (s: string): Promise<Backtest> => j(`/api/backtest?symbol=${s}`);
export const getHealth = (s: string): Promise<Health> => j(`/api/health?symbol=${s}`);

/** Subscribe to the live WS feed for `symbol`; auto-reconnects on drop. */
export function useLiveFeed(
  symbol: string,
  onTick: (price: number, epoch: number) => void,
  onCandle: (bar: Candle) => void
) {
  const cb = useRef({ onTick, onCandle });
  cb.current = { onTick, onCandle };
  useEffect(() => {
    if (!symbol) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws?symbol=${symbol}`);
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "tick") cb.current.onTick(m.price, m.epoch);
        else if (m.type === "candle") cb.current.onCandle(m.bar);
      };
      ws.onclose = () => { if (!closed) retry = setTimeout(connect, 1500); };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws?.close(); };
  }, [symbol]);
}
