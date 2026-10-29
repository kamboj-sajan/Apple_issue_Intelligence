"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt } from "./ui";

const axisProps = {
  stroke: "var(--axis)",
  tick: { fill: "var(--text-muted)", fontSize: 11 },
  tickLine: false,
};
const gridProps = { stroke: "var(--grid)", vertical: false };

function TooltipBox({ active, payload, label, labelFormat = fmt.date, valueFormat = (v) => fmt.num(v, 1) }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="mb-1 font-medium text-ink">{labelFormat(label)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 text-ink-2">
          <span className="inline-block h-0.5 w-3" style={{ background: p.color }} />
          <span>{p.name}</span>
          <span className="tabular ml-auto font-medium text-ink">{valueFormat(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Weekly observed vs expected question counts (one y-axis); optional highlighted week. */
export function ObservedExpectedChart({ series, highlightWeek, height = 280 }) {
  const data = series.map((p) => ({ ...p, t: new Date(p.week_start).getTime() }));
  const hl = highlightWeek ? data.find((d) => d.week_start.slice(0, 10) === highlightWeek.slice(0, 10)) : null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 16, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridProps} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmt.date} {...axisProps} />
        <YAxis allowDecimals={false} width={36} {...axisProps} />
        <Tooltip content={<TooltipBox />} cursor={{ stroke: "var(--axis)" }} />
        <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
        <Line name="Observed questions" dataKey="observed" stroke="var(--series-1)" strokeWidth={2} dot={false}
          activeDot={{ r: 4 }} isAnimationActive={false} />
        <Line name="Expected (negative binomial)" dataKey="expected_negbin" stroke="var(--reference)" strokeWidth={2}
          strokeDasharray="5 4" dot={false} connectNulls isAnimationActive={false} />
        {hl && (
          <ReferenceDot x={hl.t} y={hl.observed} r={6} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2}
            label={{ value: "signal", position: "top", fill: "var(--text-secondary)", fontSize: 11 }} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** One metric's monthly permutation z-score with its calibrated alert threshold. */
export function DriftZChart({ rows, height = 260 }) {
  const data = rows.map((r) => ({ ...r, t: new Date(`${r.month}-01T00:00:00Z`).getTime() }));
  const threshold = rows.find((r) => r.threshold !== null)?.threshold;
  const alerts = data.filter((d) => d.alert);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 16, right: 110, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridProps} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]}
          tickFormatter={(t) => new Date(t).toISOString().slice(0, 7)} {...axisProps} />
        <YAxis width={36} {...axisProps} />
        <Tooltip content={<TooltipBox labelFormat={(t) => new Date(t).toISOString().slice(0, 7)} valueFormat={(v) => fmt.num(v, 2)} />}
          cursor={{ stroke: "var(--axis)" }} />
        {threshold !== undefined && (
          <ReferenceLine y={threshold} stroke="var(--reference)" strokeDasharray="5 4"
            label={{ value: `threshold z = ${fmt.num(threshold, 1)}`, position: "right", fill: "var(--text-secondary)", fontSize: 11 }} />
        )}
        <Line name="Drift z-score" dataKey="z" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
          isAnimationActive={false} />
        {alerts.map((a) => (
          <ReferenceDot key={a.month} x={a.t} y={a.z} r={5} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal bars with 95% CI whiskers for one metric across methods (single hue; identity is the row label). */
export function MetricBars({ rows, valueLabel, height }) {
  const data = rows.map((r) => ({ ...r, err: [r.mean - r.ci_low, r.ci_high - r.mean] }));
  return (
    <ResponsiveContainer width="100%" height={height || 40 + rows.length * 44}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }} barCategoryGap={10}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" domain={[0, "auto"]} {...axisProps} tickFormatter={(v) => fmt.num(v, 2)} />
        <YAxis type="category" dataKey="label" width={170} {...axisProps} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
        <Tooltip cursor={{ fill: "var(--surface-2)" }}
          content={({ active, payload }) => active && payload?.length ? (
            <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">
              <div className="font-medium">{payload[0].payload.label}</div>
              <div className="tabular text-ink-2">{valueLabel}: {fmt.num(payload[0].payload.mean)} [{fmt.num(payload[0].payload.ci_low)}, {fmt.num(payload[0].payload.ci_high)}]</div>
            </div>
          ) : null} />
        <Bar dataKey="mean" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          <ErrorBar dataKey="err" direction="x" stroke="var(--text-secondary)" width={6} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Simple single-series bars (counts by category). */
export function CountBars({ rows, labelKey, valueKey, height }) {
  return (
    <ResponsiveContainer width="100%" height={height || 40 + rows.length * 30}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }} barCategoryGap={6}>
        <XAxis type="number" hide />
        <YAxis type="category" dataKey={labelKey} width={170} {...axisProps} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
        <Tooltip cursor={{ fill: "var(--surface-2)" }} formatter={(v) => fmt.int(v)} />
        <Bar dataKey={valueKey} name="Questions" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false}
          label={{ position: "right", fill: "var(--text-secondary)", fontSize: 11, formatter: (v) => fmt.int(v) }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Single-series rate over time (e.g. share of novel questions). */
export function RateLine({ rows, xKey, yKey, name, height = 220, yFormat = (v) => fmt.pct(v, 1) }) {
  const data = rows.map((r) => ({ ...r, t: new Date(`${r[xKey]}-01T00:00:00Z`).getTime() }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 12, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridProps} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]}
          tickFormatter={(t) => new Date(t).toISOString().slice(0, 7)} {...axisProps} />
        <YAxis width={44} tickFormatter={yFormat} {...axisProps} />
        <Tooltip content={<TooltipBox labelFormat={(t) => new Date(t).toISOString().slice(0, 7)} valueFormat={yFormat} />}
          cursor={{ stroke: "var(--axis)" }} />
        <Line name={name} dataKey={yKey} stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
          isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
