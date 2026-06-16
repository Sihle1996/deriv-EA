import { useCallback, useEffect, useState, type ReactNode } from "react";
import Chart from "./Chart";
import {
  type AtsOverlay, type Backtest, type Candle, type Health, type SignalRec,
  getAts, getBacktest, getCandles, getHealth, getSignals, getSymbols, useLiveFeed,
} from "./api";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"];

export default function App() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [tf, setTf] = useState("1m");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [liveBar, setLiveBar] = useState<Candle | null>(null);
  const [price, setPrice] = useState<number | null>(null);
  const [signals, setSignals] = useState<SignalRec[]>([]);
  const [ats, setAts] = useState<AtsOverlay | null>(null);
  const [bt, setBt] = useState<Backtest | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    getSymbols().then((s) => {
      setSymbols(s.map((x) => x.symbol));
      if (s[0]) setSymbol((cur) => cur || s[0].symbol);
    });
  }, []);

  useEffect(() => {
    if (!symbol) return;
    let stop = false;
    setCandles([]); setLiveBar(null);
    const loadCandles = () => getCandles(symbol, tf).then((c) => { if (!stop) setCandles(c); });
    const poll = () => {
      loadCandles();                       // refetch candles so higher TFs stay current
      getSignals(symbol).then(setSignals);
      getAts(symbol).then((a) => { if (!stop) setAts(a); });
      getBacktest(symbol).then(setBt);
      getHealth(symbol).then(setHealth);
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { stop = true; clearInterval(id); };
  }, [symbol, tf]);

  const onTick = useCallback((p: number) => setPrice(p), []);
  const onCandle = useCallback((bar: Candle) => setLiveBar({ ...bar }), []);
  useLiveFeed(symbol, onTick, onCandle);

  return (
    <div className="app">
      <div className="caveat">
        Research harness — Deriv synthetic indices are CSPRNG-generated: there is no predictive edge.
        Demo only · No trading · Read-only viewer.
      </div>
      <header>
        <h1>Deriv Research Dashboard</h1>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={tf} onChange={(e) => setTf(e.target.value)} title="chart timeframe">
          {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <span className="price">{price !== null ? price.toFixed(5) : "—"}</span>
        {health && <span className={"badge " + (health.live ? "ok" : "bad")}>{health.live ? "LIVE" : "STALE"}</span>}
      </header>

      <Chart candles={candles} signals={signals} liveBar={liveBar} tf={tf} ats={ats} />

      <div className="grid">
        <BacktestPanel bt={bt} />
        <HealthPanel health={health} />
      </div>

      <SignalsTable signals={signals} />
    </div>
  );
}

function pct(x: number | undefined) { return x === undefined || isNaN(x) ? "—" : (x * 100).toFixed(1); }

function BacktestPanel({ bt }: { bt: Backtest | null }) {
  if (!bt) return <Panel title="Backtest — would it make money?"><p>loading…</p></Panel>;
  if (bt.error || !bt.real) return <Panel title="Backtest — would it make money?"><p className="small">{bt.error || "no tradeable signals yet"}</p></Panel>;
  const r = bt.real, n = bt.null;
  const vcls = { good: "ok", watch: "watch", weak: "weak", bad: "bad" }[bt.verdict_class || "bad"] || "bad";
  return (
    <Panel title="Backtest — would it make money?">
      <table>
        <thead><tr><th></th><th>win %</th><th>P&amp;L</th><th>ROI</th></tr></thead>
        <tbody>
          <tr><td>Real signals</td><td>{pct(r.win_rate)}</td><td>{r.total_pnl.toFixed(2)}</td><td>{r.roi_pct.toFixed(1)}%</td></tr>
          {n && <tr className="dim"><td>Random (null)</td><td>{pct(n.win_rate)}</td><td>{n.total_pnl.toFixed(2)}</td><td>{n.roi_pct.toFixed(1)}%</td></tr>}
        </tbody>
      </table>
      <p>break-even win rate <b>{pct(bt.breakeven)}%</b> · <span className={vcls}>{bt.verdict}</span></p>
      {(bt.trend_n || bt.reversal_n) ? (
        <p>after expansion: <b>{bt.trend_n}</b> trended / <b>{bt.reversal_n}</b> reversed
          {bt.trend_continuation != null && <> · continuation <b>{(bt.trend_continuation * 100).toFixed(0)}%</b></>}
          <span className="small"> (no momentum on an RNG; ratio also reflects threshold shapes — read the P&amp;L, not this)</span>
        </p>
      ) : null}
      <p className="small">{bt.caveat}</p>
    </Panel>
  );
}

function HealthPanel({ health }: { health: Health | null }) {
  if (!health) return <Panel title="Archive health"><p>loading…</p></Panel>;
  return (
    <Panel title="Archive health">
      <p>ticks <b>{health.ticks.toLocaleString()}</b> · signals <b>{health.signals}</b></p>
      <p>coverage <b>{health.coverage_pct ?? "—"}%</b> · gaps <b>{health.gaps ?? "—"}</b></p>
      <p>last archived tick <b>{health.last_tick_age_s ?? "—"}s</b> ago
        <span className={"badge " + (health.live ? "ok" : "bad")}>{health.live ? "fresh" : "stale"}</span></p>
      <p className="small">Freshness is archive-based (bots flush every ~100 ticks); the live chart above is real-time.</p>
    </Panel>
  );
}

function SignalsTable({ signals }: { signals: SignalRec[] }) {
  return (
    <Panel title={`Recent signals (${signals.length})`}>
      <div className="scroll">
        <table className="sig">
          <thead><tr><th>tf</th><th>phase</th><th>dir</th><th>price</th><th>bw %ile</th><th>z</th><th>bar epoch</th></tr></thead>
          <tbody>
            {signals.slice(0, 60).map((s, i) => (
              <tr key={i}>
                <td>{s.timeframe}</td>
                <td className={s.phase === "expansion" ? "ok" : ""}>{s.phase}</td>
                <td>{s.direction ?? "—"}</td>
                <td>{s.price_at_signal?.toFixed?.(5)}</td>
                <td>{s.bw_percentile != null ? (s.bw_percentile * 100).toFixed(0) : "—"}</td>
                <td>{s.bbw_zscore?.toFixed?.(2)}</td>
                <td>{s.bar_epoch}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return <div className="panel"><h3>{title}</h3>{children}</div>;
}
