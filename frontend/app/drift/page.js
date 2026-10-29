"use client";

import { useMemo, useState } from "react";
import { useApi } from "@/lib/api";
import { DriftZChart, RateLine } from "@/components/charts";
import { Badge, Card, ErrorBox, Field, Loading, PageHeader, Table, fmt, inputClass } from "@/components/ui";

const METRICS = {
  mmd_rbf: "Embedding distribution (MMD)",
  centroid_cosine: "Embedding centroid movement",
  clusters: "Semantic-cluster mix (JSD)",
  categories: "Issue-category mix (JSD)",
  tags: "Tag mix (JSD)",
  os: "OS-version mix (JSD)",
  devices: "Device mix (JSD)",
  vocabulary: "Title vocabulary (JSD)",
};

export default function DriftPage() {
  const { data, error, loading } = useApi("/drift");
  const [metric, setMetric] = useState("mmd_rbf");
  const d = data?.data;
  const rows = useMemo(() => (d?.series || []).filter((r) => r.metric === metric && r.status === "ok"), [d, metric]);
  const alerts = (d?.series || []).filter((r) => r.alert);
  const movers = (d?.cluster_movements || []).filter((m) => m.share_ratio >= 2 || m.share_ratio <= 0.5);
  const fa = d?.evaluation?.eval_no_drift;
  const power = d?.evaluation?.eval_injected || [];

  return (
    <>
      <PageHeader title="Drift monitoring"
        subtitle="Has what people ask about changed? Each month is compared with the three months before it, using equal-size samples and permutation z-scores. Alert thresholds are calibrated on 2019–2022 history. This is separate from volume anomalies." />
      <ErrorBox error={error} />
      {loading && <Loading />}
      {d && (
        <>
          <Card title="Monthly drift score" subtitle="Permutation z-score; dots mark alerts (z above threshold and p < 0.01)"
            right={<Field label="Metric"><select className={inputClass} value={metric} onChange={(e) => setMetric(e.target.value)}>
              {Object.entries(METRICS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></Field>}>
            <DriftZChart rows={rows} />
          </Card>

          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Historical drift alerts" subtitle={`Alert rate: ${fmt.pct(d.summary?.alert_rate_calibration, 1)} in calibration months, ${fmt.pct(d.summary?.alert_rate_after_calibration, 1)} afterwards`}>
              <Table rows={alerts} rowKey={(r) => `${r.month}${r.metric}`} empty="No alerts" columns={[
                { key: "month", label: "Month" },
                { key: "metric", label: "Metric", render: (r) => METRICS[r.metric] || r.metric },
                { key: "z", label: "z", align: "right", render: (r) => fmt.num(r.z, 1) },
                { key: "thr", label: "Threshold", align: "right", render: (r) => fmt.num(r.threshold, 1) },
                { key: "n", label: "n", align: "right", render: (r) => fmt.int(r.n) },
              ]} />
            </Card>
            <Card title="New vocabulary" subtitle="Title terms in ≥5 questions of a month and ≤1 in the previous 12 months">
              <div className="flex flex-wrap gap-2">
                {d.new_terms.map((t) => (
                  <span key={`${t.month}${t.term}`} className="rounded-md border border-line px-2 py-1 text-sm">
                    <span className="font-medium">{t.term}</span> <span className="text-xs text-muted">{t.month} · {t.docs}</span>
                  </span>
                ))}
              </div>
            </Card>
          </div>

          <Card className="mt-6" title="Issue-cluster share changes" subtitle="Clusters whose share at least doubled or halved vs the reference months (≥10 questions)">
            <Table rows={movers.slice(-40).reverse()} rowKey={(r) => `${r.month}${r.cluster}`} columns={[
              { key: "month", label: "Month" },
              { key: "cluster", label: "Cluster", render: (r) => <><span className="font-mono text-xs">{r.cluster}</span> <span className="text-ink-2">{r.top_terms.slice(0, 5).join(", ")}</span></> },
              { key: "share", label: "Share", align: "right", render: (r) => fmt.pct(r.share, 1) },
              { key: "ref", label: "Reference", align: "right", render: (r) => fmt.pct(r.reference_share, 1) },
              { key: "ratio", label: "Ratio", align: "right", render: (r) => `×${fmt.num(r.share_ratio, 1)}` },
              { key: "dir", label: "", render: (r) => <Badge tone={r.direction === "growing" ? "accent" : "neutral"}>{r.direction}</Badge> },
            ]} />
          </Card>

          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Share of novel questions" subtitle={`Far from every training-period cluster centroid (cosine < ${fmt.num(d.summary?.novelty_threshold_cosine, 2)})`}>
              <RateLine rows={d.novelty_rate.filter((r) => r.n >= 50)} xKey="month" yKey="novel_rate" name="Novel share" />
              <p className="mt-2 text-xs text-ink-2">
                {d.new_cluster_candidates.length} candidate new clusters found
                {d.new_cluster_candidates.length > 0 && `: ${d.new_cluster_candidates.map((c) => `${c.quarter} (${c.top_terms.slice(0, 2).join(", ")})`).join("; ")}`}.
              </p>
            </Card>
            <Card title="Controlled evaluation" subtitle="False alarms on same-month random splits; detection of an injected topic surge">
              {fa && <p className="mb-3 text-sm">False-alarm rate (no drift): {Object.entries(fa).map(([k, v]) => `${k} ${fmt.pct(v)}`).join(" · ")}</p>}
              <Table rows={power} rowKey={(r) => r.fraction} columns={[
                { key: "fraction", label: "Surge share", render: (r) => fmt.pct(r.fraction) },
                ...["centroid_cosine", "mmd_rbf", "clusters", "tags", "categories"].map((m) => ({
                  key: m, label: m.replace("_rbf", "").replace("_cosine", ""), align: "right", render: (r) => fmt.pct(r[m]),
                })),
              ]} />
              {d.summary?.volume_vs_semantic_drift && (
                <p className="mt-3 text-xs text-ink-2">
                  Semantic drift vs volume anomalies: Spearman ρ = {fmt.num(d.summary.volume_vs_semantic_drift.spearman_rho, 2)} (p = {fmt.num(d.summary.volume_vs_semantic_drift.p_value, 2)}, {d.summary.volume_vs_semantic_drift.n_months} months), so the two are largely distinct signals.
                </p>
              )}
            </Card>
          </div>
        </>
      )}
    </>
  );
}
