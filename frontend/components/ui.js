"use client";

export const fmt = {
  int: (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString("en-US")),
  num: (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
  pct: (v, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`),
  signedPct: (v) => (v === null || v === undefined ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}%`),
  date: (v) => (v ? new Date(v).toISOString().slice(0, 10) : "–"),
  dateTime: (v) => (v ? new Date(v).toISOString().replace("T", " ").slice(0, 16) + " UTC" : "–"),
  sci: (v) => (v === null || v === undefined ? "–" : v < 1e-3 ? Number(v).toExponential(1) : Number(v).toFixed(3)),
};

export function PageHeader({ title, subtitle, children }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink-2">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export function Card({ title, subtitle, children, className = "", right }) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-2">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-5">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="tabular mt-2 text-3xl font-semibold">{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-2">{hint}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral", title }) {
  const tones = {
    neutral: "bg-surface-2 text-ink-2",
    accent: "bg-accent/15 text-accent",
    critical: "bg-critical/15 text-critical",
    good: "bg-good/15 text-good",
    warning: "bg-warning/20 text-ink",
  };
  return (
    <span title={title} className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Loading({ label = "Loading…" }) {
  return <div className="py-8 text-center text-sm text-muted">{label}</div>;
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div role="alert" className="rounded-lg border border-critical/40 bg-critical/10 p-4 text-sm">
      <span className="font-semibold text-critical">Error</span>
      {error.code && <span className="ml-2 font-mono text-xs text-ink-2">{error.code}</span>}
      <div className="mt-1 text-ink">{error.message}</div>
    </div>
  );
}

export function Empty({ children }) {
  return <div className="py-8 text-center text-sm text-muted">{children}</div>;
}

export function QuestionLink({ id, title, link }) {
  return (
    <a href={link || `https://apple.stackexchange.com/questions/${id}`} target="_blank" rel="noreferrer"
      className="text-accent hover:underline">
      {title || `Question ${id}`}
      <span className="sr-only"> (opens Ask Different)</span>
      <span aria-hidden className="ml-1 text-xs">↗</span>
    </a>
  );
}

export function Pagination({ total, limit, offset, onChange }) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  return (
    <div className="mt-4 flex items-center justify-between text-sm text-ink-2">
      <span className="tabular">
        {total === 0 ? "No results" : `${fmt.int(offset + 1)}–${fmt.int(Math.min(offset + limit, total))} of ${fmt.int(total)}`}
      </span>
      <div className="flex gap-2">
        <button className="rounded-md border border-line px-3 py-1 disabled:opacity-40" disabled={page <= 1}
          onClick={() => onChange(Math.max(0, offset - limit))}>Previous</button>
        <button className="rounded-md border border-line px-3 py-1 disabled:opacity-40" disabled={page >= pages}
          onClick={() => onChange(offset + limit)}>Next</button>
      </div>
    </div>
  );
}

export function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink-2">
      {label}
      {children}
    </label>
  );
}

export const inputClass =
  "rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-ink outline-none focus:border-accent";

export function Table({ columns, rows, rowKey, empty = "No rows" }) {
  if (!rows?.length) return <Empty>{empty}</Empty>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-line text-xs uppercase tracking-wide text-muted">
            {columns.map((c) => (
              <th key={c.key} className={`px-2 py-2 font-medium ${c.align === "right" ? "text-right" : ""}`}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey ? rowKey(r) : i} className="border-b border-line last:border-0 hover:bg-surface-2">
              {columns.map((c) => (
                <td key={c.key} className={`px-2 py-2 align-top ${c.align === "right" ? "tabular text-right" : ""}`}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
