"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useApi } from "@/lib/api";
import { ObservedExpectedChart } from "@/components/charts";
import { Badge, Card, ErrorBox, Field, Loading, PageHeader, Pagination, QuestionLink, Table, fmt, inputClass } from "@/components/ui";

const LIMIT = 15;

function AnomalyPanel({ id, onClose }) {
  const { data, error, loading } = useApi(`/anomalies/${id}`);
  const a = data?.data;
  return (
    <Card title={a ? `${a.group_type}: ${a.group_value}` : "Signal"} subtitle={a ? `week of ${fmt.date(a.week_start)}` : ""}
      right={<button className="text-sm text-ink-2 hover:text-ink" onClick={onClose}>Close ✕</button>}>
      <ErrorBox error={error} />
      {loading && <Loading />}
      {a && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div><div className="text-xs text-muted">Observed</div><div className="tabular text-xl font-semibold">{a.observed}</div></div>
            <div><div className="text-xs text-muted">Expected</div><div className="tabular text-xl font-semibold">{fmt.num(a.expected, 1)}</div></div>
            <div><div className="text-xs text-muted">Change</div><div className="tabular text-xl font-semibold">{fmt.signedPct(a.rel_change)}</div></div>
            <div><div className="text-xs text-muted">FDR q-value</div><div className="tabular text-xl font-semibold">{fmt.sci(a.q_value)}</div></div>
          </div>
          <ObservedExpectedChart series={a.series} highlightWeek={a.week_start} />
          <dl className="grid grid-cols-1 gap-2 text-sm md:grid-cols-[180px_1fr]">
            {Object.entries(a.explanation).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-ink-2">{k.replaceAll("_", " ")}</dt>
                <dd>{Array.isArray(v) ? v.join(", ") : v}</dd>
              </div>
            ))}
            <dt className="text-ink-2">co-occurring</dt>
            <dd>{a.co_occurring.join(" · ") || "–"}</dd>
          </dl>
          <div>
            <h3 className="mb-2 text-sm font-semibold">Questions from that week (most viewed)</h3>
            <ul className="space-y-1 text-sm">
              {a.evidence_questions.map((q) => (
                <li key={q.question_id}><QuestionLink id={q.question_id} title={q.title} link={q.link} /> <span className="text-xs text-muted">{fmt.date(q.created_at)}</span></li>
              ))}
            </ul>
          </div>
          <p className="rounded-md bg-surface-2 p-3 text-xs text-ink-2"><strong>Limitations.</strong> {a.limitations}</p>
        </div>
      )}
    </Card>
  );
}

function GroupChart() {
  const [groupType, setGroupType] = useState("os");
  const [group, setGroup] = useState(null);
  const groups = useApi("/issues/groups", { group_type: groupType, limit: 40 });
  const options = groups.data?.data || [];
  const selected = group && options.some((o) => o.group_value === group) ? group : options[0]?.group_value;
  const series = useApi(selected ? "/issues/series" : null, { group_type: groupType, group_value: selected });
  return (
    <Card title="Weekly issue volume" subtitle="Observed questions vs the expected count from the group's share of site volume over the previous 26 weeks">
      <div className="mb-3 flex flex-wrap gap-3">
        <Field label="Group type">
          <select className={inputClass} value={groupType} onChange={(e) => { setGroupType(e.target.value); setGroup(null); }}>
            {["os", "device", "tag", "category", "error", "tech", "cluster", "tagpair"].map((g) => <option key={g}>{g}</option>)}
          </select>
        </Field>
        <Field label="Group">
          <select className={inputClass} value={selected || ""} onChange={(e) => setGroup(e.target.value)}>
            {options.map((o) => <option key={o.group_value} value={o.group_value}>{o.group_value} ({fmt.int(o.total)})</option>)}
          </select>
        </Field>
      </div>
      <ErrorBox error={series.error} />
      {series.loading && <Loading />}
      {series.data && <ObservedExpectedChart series={series.data.data} />}
    </Card>
  );
}

function IssuesInner() {
  const router = useRouter();
  const params = useSearchParams();
  const selected = params.get("anomaly");
  const [filters, setFilters] = useState({ group_type: "", device: "", os: "", tag: "", start_date: "", end_date: "", signal_type: "", min_score: "" });
  const [offset, setOffset] = useState(0);
  const [sort, setSort] = useState("score");
  useEffect(() => setOffset(0), [filters, sort]);
  const list = useApi("/anomalies", { ...filters, sort, limit: LIMIT, offset });
  const set = (k) => (e) => setFilters((f) => ({ ...f, [k]: e.target.value }));
  const open = (id) => router.replace(id ? `/issues?anomaly=${id}` : "/issues", { scroll: false });

  return (
    <>
      <PageHeader title="Emerging issues"
        subtitle="Weeks where an issue group's questions rose significantly above expectation (negative-binomial test, Benjamini–Hochberg FDR across groups). These are candidate discussion patterns — not proof of a software bug or a causal relationship." />
      <GroupChart />
      {selected && <div className="mt-6"><AnomalyPanel id={selected} onClose={() => open(null)} /></div>}
      <Card className="mt-6" title="Ranked signals" subtitle="Rank score = −log10(q) × log2(1 + relative increase)">
        <div className="mb-4 flex flex-wrap items-end gap-3">
          <Field label="Group type"><select className={inputClass} value={filters.group_type} onChange={set("group_type")}>
            <option value="">all</option>{["os", "device", "tag", "tagpair", "category", "error", "tech", "cluster"].map((g) => <option key={g}>{g}</option>)}</select></Field>
          <Field label="Device"><input className={inputClass} placeholder="e.g. iphone" value={filters.device} onChange={set("device")} /></Field>
          <Field label="OS version"><input className={inputClass} placeholder="e.g. macos:10.15" value={filters.os} onChange={set("os")} /></Field>
          <Field label="Tag"><input className={inputClass} placeholder="e.g. wifi" value={filters.tag} onChange={set("tag")} /></Field>
          <Field label="From"><input type="date" className={inputClass} value={filters.start_date} onChange={set("start_date")} /></Field>
          <Field label="To"><input type="date" className={inputClass} value={filters.end_date} onChange={set("end_date")} /></Field>
          <Field label="Type"><select className={inputClass} value={filters.signal_type} onChange={set("signal_type")}>
            <option value="">all</option><option>spike</option><option>sustained</option><option>too_recent</option></select></Field>
          <Field label="Min score"><input type="number" min="0" step="0.5" className={`${inputClass} w-24`} value={filters.min_score} onChange={set("min_score")} /></Field>
          <Field label="Sort"><select className={inputClass} value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="score">score</option><option value="newest">newest</option><option value="oldest">oldest</option></select></Field>
        </div>
        <ErrorBox error={list.error} />
        {list.loading && <Loading />}
        {list.data && (
          <>
            <Table rows={list.data.items} rowKey={(r) => r.anomaly_id} empty="No signals match these filters" columns={[
              { key: "week", label: "Week", render: (r) => fmt.date(r.week_start) },
              { key: "group", label: "Issue group", render: (r) => <><Badge>{r.group_type}</Badge> <span className="ml-1">{r.group_value}</span></> },
              { key: "obs", label: "Observed", align: "right", render: (r) => r.observed },
              { key: "exp", label: "Expected", align: "right", render: (r) => fmt.num(r.expected, 1) },
              { key: "chg", label: "Change", align: "right", render: (r) => fmt.signedPct(r.rel_change) },
              { key: "q", label: "q", align: "right", render: (r) => fmt.sci(r.q_value) },
              { key: "score", label: "Score", align: "right", render: (r) => fmt.num(r.rank_score, 2) },
              { key: "type", label: "Type", render: (r) => <Badge tone={r.signal_type === "sustained" ? "accent" : "neutral"}>{r.signal_type}</Badge> },
              { key: "flags", label: "Checks", render: (r) => <span className="text-xs text-ink-2">{r.rolling_agrees ? "rolling ✓" : "rolling –"}{r.partial_week ? " · partial week" : ""}{r.low_site_volume ? " · low volume" : ""}</span> },
              { key: "go", label: "", render: (r) => <button className="text-accent hover:underline" onClick={() => open(r.anomaly_id)}>Explain</button> },
            ]} />
            <Pagination total={list.data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
          </>
        )}
      </Card>
    </>
  );
}

export default function IssuesPage() {
  return <Suspense fallback={<Loading />}><IssuesInner /></Suspense>;
}
