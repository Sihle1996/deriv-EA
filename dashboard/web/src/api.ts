import { useEffect, useRef } from "react";

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type SignalRec = {
  timeframe: string; phase: string; direction: string | null; bar_epoch: number;
  price_at_signal: number; value_line?: number | null; htf_bias?: string | null; [k: string]: any;
};
export type Health = {
  symbol: string; ticks: number; signals: number; last_tick_age_s: number | null;
  coverage_pct: number | null; gaps: number | null; live: boolean;
};
export type Backtest = {
  error?: string; verdict?: string; verdict_class?: string; caveat?: string; breakeven?: number;
  real?: { win_rate: number; total_pnl: number; roi_pct: number; n: number };
  null?: { win_rate: number; total_pnl: number; roi_pct: number; n: number } | null;
};
export type AtsValueLine = {
  epoch: number; value_line: number; tf: string;
  box_start: number; box_end: number; box_high: number | null; box_low: number | null;
  line_end: number;
};
export type AtsEntry = {
  bar_epoch: number; direction: string | null; price: number | null; tf: string;
  value_line: number | null; htf_bias: string | null;
};
export type AtsFunnel = {
  htf_contractions: number; htf_breakouts: number;
  ltf_contractions: number; ltf_breakouts: number;
  pullback_candidates: number; entries: number;
  blocked_no_bias: number; blocked_counter: number;
};
export type AtsOverlay = {
  symbol: string; htf: string; ltf: string;
  value_lines: AtsValueLine[]; entries: AtsEntry[]; funnel: AtsFunnel;
};

const j = (url: string) => fetch(url).then((r) => r.json());

export const getSymbols = (): Promise<{ symbol: string; live: boolean }[]> => j("/api/symbols");
export const getCandles = (s: string, tf = "1m", count = 500): Promise<Candle[]> =>
  j(`/api/candles?symbol=${s}&tf=${tf}&count=${count}`);
export const getArchiveCandles = (s: string, tf = "1m", count = 2000): Promise<Candle[]> =>
  j(`/api/archive_candles?symbol=${s}&tf=${tf}&count=${count}`);
export const getSignals = (s: string, limit = 100): Promise<SignalRec[]> =>
  j(`/api/signals?symbol=${s}&limit=${limit}`);
export const getBacktest = (s: string): Promise<Backtest> => j(`/api/backtest?symbol=${s}`);
export const getHealth = (s: string): Promise<Health> => j(`/api/health?symbol=${s}`);
export const getAts = (s: string): Promise<AtsOverlay> => j(`/api/ats?symbol=${s}`);

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
