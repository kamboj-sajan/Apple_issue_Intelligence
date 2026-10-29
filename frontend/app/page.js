"use client";

import Link from "next/link";
import { useApi } from "@/lib/api";
import { CountBars } from "@/components/charts";
import { Badge, Card, ErrorBox, Loading, PageHeader, Stat, Table, fmt } from "@/components/ui";

export default function Overview() {
  const ov = useApi("/overview");
  const trending = useApi("/issues/trending", { weeks: 12, limit: 6 });
  const d = ov.data?.data;
  const runs = d?.collection?.last_runs || [];
  const lastRun = runs[runs.length - 1];

  return (
    <>
      <PageHeader title="Overview"
        subtitle="Signals from publicly posted Ask Different questions. Unusual increases are candidate discussion patterns, not evidence of defects or of how many users are affected." />
      <ErrorBox error={ov.error} />
      {ov.loading && <Loading />}
      {d && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Indexed questions" value={fmt.int(d.total_questions)}
              hint={`${fmt.date(d.created_range[0])} → ${fmt.date(d.created_range[1])}`} />
            <Stat label="Anomaly signals" value={fmt.int(d.anomaly_signals)} hint="negative binomial, FDR q ≤ 0.05" />
            <Stat label="Latest analysis" value={fmt.date(d.latest_analysis || d.loaded_at)} hint={`pipeline ${d.pipeline_version}`} />
            <Stat label="Newest question" value={fmt.date(d.latest_question_at)} hint={`loaded ${fmt.dateTime(d.loaded_at)}`} />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Most active issue categories" subtitle="Primary rule-based category, last 12 months of data">
              <CountBars rows={d.most_active_categories_last_12_months} labelKey="category" valueKey="questions" />
            </Card>
            <Card title="Data freshness & collection" subtitle="From the ingestion checkpoint of the loaded snapshot">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                <dt className="text-ink-2">Questions incl. linked context</dt>
                <dd className="tabular">{fmt.int(d.questions_including_context)}</dd>
                <dt className="text-ink-2">Last collection run</dt>
                <dd>{lastRun ? fmt.dateTime(lastRun.finished_at) : "–"}</dd>
                <dt className="text-ink-2">Last run mode / result</dt>
                <dd>{lastRun ? <><Badge>{lastRun.mode}</Badge> <Badge tone={lastRun.stopped_reason === "completed" ? "good" : "warning"}>{lastRun.stopped_reason}</Badge></> : "–"}</dd>
                <dt className="text-ink-2">Activity watermark</dt>
                <dd>{d.collection?.update_watermark ? fmt.dateTime(d.collection.update_watermark * 1000) : "–"}</dd>
                <dt className="text-ink-2">Closed as duplicate</dt>
                <dd className="tabular">{fmt.int(d.quality?.closed_as_duplicate)}</dd>
              </dl>
            </Card>
          </div>
        </>
      )}

      <Card className="mt-6" title="Recent emerging-issue signals"
        right={<Link href="/issues" className="text-sm text-accent hover:underline">All signals →</Link>}>
        <ErrorBox error={trending.error} />
        {trending.data && (
          <Table rows={trending.data.data.recent_signals} rowKey={(r) => r.anomaly_id} columns={[
            { key: "week", label: "Week", render: (r) => fmt.date(r.week_start) },
            { key: "group", label: "Issue group", render: (r) => <><Badge>{r.group_type}</Badge> <span className="ml-1">{r.group_value}</span></> },
            { key: "obs", label: "Observed / expected", align: "right", render: (r) => `${r.observed} / ${fmt.num(r.expected, 1)}` },
            { key: "type", label: "Type", render: (r) => <Badge tone={r.signal_type === "sustained" ? "accent" : "neutral"}>{r.signal_type}</Badge> },
            { key: "go", label: "", render: (r) => <Link className="text-accent hover:underline" href={`/issues?anomaly=${r.anomaly_id}`}>Details</Link> },
          ]} />
        )}
      </Card>
    </>
  );
}
