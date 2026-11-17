// GTA VI launch-day Twitch viewers, hour by hour (Node 22, standard library only, deterministic).
//
// Idea: the hour-1 print is known, so each later hour h is forecast as print_1 * exp(rho_h), where rho_h is the
// log-ratio "hour h / hour 1" seen on the same UTC release day of past big launches (the case's `comparables`).
// Launches are weighted by how GTA VI-like they are (console-first, single-player, unlock at midnight local time,
// rolling east to west), and the five quantiles are read from the weighted, kernel-smoothed mixture of those
// log-ratios, widened a little. The pinball loss is scored on the log scale, so the median is the target.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const HOUR_MS = 3_600_000;

/** Model constants (log-ratio units). */
const P = {
  wA: 1, // weight of console-first single-player launches (the closest relatives of GTA VI)
  wB: 0.35, // other big launches with rolling midnight unlocks
  wC: 0.08, // simultaneous-global, PC-only, live-service and free-to-play launches
  bw: 0.25, // Gaussian kernel width around each past launch's log-ratio
  widen: 0.9, // leave-one-out over the console-first launches gives 90% coverage here
  sf: 0.5, // width of the similarity window on the pre-release ramp feature (0 = off) // multiplier on the distance of each quantile from the median
};

interface Point { from: string; viewers: number }
interface Comparable { name: string; day_start: string; hourly: Point[] }
interface Case {
  day_start: string;
  history_hourly: Point[];
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

const A = /wolverine|red dead|god of war|spider-man|ghost of|last of us|stellar blade|final fantasy|saros|death stranding|resident evil|zelda|forza|halo|starfield/;
const C = /wukong|silksong|battlefield|marathon|baldur|arc raiders|rivals|helldivers|clair obscur|nightreign/;

const launchWeight = (name: string): number => {
  const n = name.toLowerCase();
  if (n.startsWith('grand theft auto vi')) return 0; // GTA VI's own pre-release category is not a launch
  return A.test(n) ? P.wA : C.test(n) ? P.wC : P.wB;
};

/** The weighted, kernel-smoothed mixture CDF of samples at x. */
const mixCdf = (xs: number[], ws: number[], total: number, x: number): number => {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += ws[i]! * 0.5 * (1 + erf((x - xs[i]!) / (P.bw * Math.SQRT2)));
  return s / total;
};

/** Abramowitz-Stegun 7.1.26 error function (max error 1.5e-7). */
const erf = (x: number): number => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};

/** The tau-quantile of the mixture, by bisection. */
const mixQuantile = (xs: number[], ws: number[], total: number, tau: number): number => {
  let lo = Math.min(...xs) - 6 * P.bw;
  let hi = Math.max(...xs) + 6 * P.bw;
  for (let it = 0; it < 60; it++) {
    const mid = (lo + hi) / 2;
    if (mixCdf(xs, ws, total, mid) < tau) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
};

/** ln(hour-1 level / mean of the hours 6..1 before the day start), or null with fewer than 3 such hours. */
const ramp = (byHour: Map<number, number>, d0: number, y1: number): number | null => {
  const v: number[] = [];
  for (let k = 1; k <= 6; k++) { const y = byHour.get(d0 - k * HOUR_MS); if (y !== undefined) v.push(y); }
  return v.length < 3 ? null : Math.log(Math.max(1, y1) / Math.max(1, v.reduce((a, b) => a + b, 0) / v.length));
};

export const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const p1 = Math.max(1, c.known.find((k) => k.hour === 1)!.viewers);
  const own = new Map<number, number>();
  for (const p of c.history_hourly) own.set(Date.parse(p.from), p.viewers);
  const f0 = ramp(own, Date.parse(c.day_start), p1);
  // per-hour log-ratio samples from every comparable launch day
  const samples: Array<{ x: number[]; w: number[] }> = Array.from({ length: 25 }, () => ({ x: [], w: [] }));
  for (const k of c.comparables) {
    let w = launchWeight(k.name);
    if (w <= 0) continue;
    const d0 = Date.parse(k.day_start);
    const byHour = new Map<number, number>();
    for (const p of k.hourly) byHour.set(Date.parse(p.from), p.viewers);
    const y1 = byHour.get(d0);
    if (y1 === undefined || y1 < 1) continue;
    const fj = ramp(byHour, d0, y1);
    if (P.sf > 0 && f0 !== null && fj !== null) w *= Math.exp(-0.5 * ((f0 - fj) / P.sf) ** 2);
    for (let h = 2; h <= 24; h++) {
      const y = byHour.get(d0 + (h - 1) * HOUR_MS);
      if (y === undefined) continue;
      samples[h]!.x.push(Math.log(Math.max(1, y) / y1));
      samples[h]!.w.push(w);
    }
  }
  return c.forecast.map(({ hour }) => {
    const s = samples[hour]!;
    const total = s.w.reduce((a, b) => a + b, 0);
    const qs = LEVELS.map((t) => mixQuantile(s.x, s.w, total, t));
    const med = qs[2]!;
    const q = qs.map((v) => Math.round(p1 * Math.exp(med + (v - med) * P.widen) * 100) / 100);
    for (let j = 1; j < 5; j++) q[j] = Math.max(q[j]!, q[j - 1]!);
    return { hour, q: q.map((v) => Math.max(v, 0.01)) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file}: unexpected levels`);
  writeFileSync(join(OUTPUT_DIR, `${file.slice(0, -3)}.out`), `${JSON.stringify({ hours: forecastCase(c) })}\n`);
}
