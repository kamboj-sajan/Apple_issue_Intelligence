"use client";

import { useState } from "react";
import { useApi } from "@/lib/api";
import { MetricBars } from "@/components/charts";
import { Badge, Card, ErrorBox, Field, Loading, PageHeader, Table, fmt, inputClass } from "@/components/ui";

const LABELS = {
  bm25: "BM25",
  jaccard_entities: "Jaccard (entities)",
  embedding: "Embedding cosine",
  no_message_passing_seed0: "No message passing (ablation)",
  graphsage_mean_over_seeds: "GraphSAGE (3 seeds)",
};
const ORDER = Object.keys(LABELS);

function LinkPrediction({ lp }) {
  const [stratum, setStratum] = useState("all_queries");
  const [metric, setMetric] = useState("mrr");
  const r = lp.report;
  const rows = ORDER.map((m) => lp.ranking.find((x) => x.method === m && x.stratum === stratum && x.metric === metric))
    .filter(Boolean).map((x) => ({ ...x, label: LABELS[x.method] }));
  const paired = lp.paired.filter((p) => p.a === "graphsage_mean_over_seeds" && p.stratum === stratum);
  const cls = lp.classification.filter((c) => c.stratum === (stratum === "all_queries" ? "all_pairs" : stratum));
  return (
    <>
      <Card title="Related-question prediction (test split)"
        subtitle={`${fmt.int(r.counts.test_queries)} test queries (${fmt.int(r.counts.test_queries_with_duplicate_label)} with a duplicate label); candidates = every earlier question; evaluated ${fmt.dateTime(r.created_at)}`}
        right={
          <div className="flex gap-2">
            <Field label="Queries"><select className={inputClass} value={stratum} onChange={(e) => setStratum(e.target.value)}>
              <option value="all_queries">all</option><option value="duplicate_label_queries">duplicate-labelled</option></select></Field>
            <Field label="Metric"><select className={inputClass} value={metric} onChange={(e) => setMetric(e.target.value)}>
              {["mrr", "hits@1", "hits@5", "hits@10", "recall@5", "recall@10", "precision@5", "precision@10"].map((m) => <option key={m}>{m}</option>)}</select></Field>
          </div>
        }>
        <MetricBars rows={rows} valueLabel={metric} />
        <Table rows={rows} rowKey={(x) => x.method} columns={[
          { key: "label", label: "Method" },
          { key: "mean", label: metric, align: "right", render: (x) => fmt.num(x.mean) },
          { key: "ci", label: "95% bootstrap CI", align: "right", render: (x) => `[${fmt.num(x.ci_low)}, ${fmt.num(x.ci_high)}]` },
          { key: "n", label: "Queries", align: "right", render: (x) => fmt.int(x.n_queries) },
        ]} />
      </Card>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Is GraphSAGE better? (paired MRR differences)" subtitle="Same queries; 2,000 bootstrap resamples">
          <Table rows={paired} rowKey={(p) => p.b} columns={[
            { key: "b", label: "GraphSAGE minus", render: (p) => LABELS[p.b] || p.b },
            { key: "d", label: "Δ MRR", align: "right", render: (p) => `${p.mean_diff >= 0 ? "+" : ""}${fmt.num(p.mean_diff)}` },
            { key: "ci", label: "95% CI", align: "right", render: (p) => `[${fmt.num(p.ci_low)}, ${fmt.num(p.ci_high)}]` },
            { key: "sig", label: "", render: (p) => (p.ci_low > 0 ? <Badge tone="good">significant</Badge> : <Badge>not significant</Badge>) },
          ]} />
        </Card>
        <Card title="Pair classification" subtitle="Positives vs the same 5 random + 5 hard sampled negatives per query (lower bounds)">
          <Table rows={cls} rowKey={(c) => `${c.method}${c.metric}`} columns={[
            { key: "method", label: "Method", render: (c) => (LABELS[c.method] || c.method).replace("graphsage_seed", "GraphSAGE seed ") },
            { key: "metric", label: "Metric", render: (c) => c.metric.replace("_", " ") },
            { key: "mean", label: "Value", align: "right", render: (c) => fmt.num(c.mean) },
            { key: "ci", label: "95% CI", align: "right", render: (c) => `[${fmt.num(c.ci_low)}, ${fmt.num(c.ci_high)}]` },
          ]} />
        </Card>
      </div>

      <Card className="mt-6" title="Model selection on validation" subtitle={`Seed 0 · selected: ${r.selected_config.decoder} decoder, ${r.selected_config.layers} SAGE layers, text residual ${r.selected_config.text_residual ? "on" : "off"}`}>
        <Table rows={r.selection} rowKey={(s) => s.name} columns={[
          { key: "name", label: "Configuration", render: (s) => <span className="font-mono text-xs">{s.name}</span> },
          { key: "layers", label: "Message passing", render: (s) => (s.layers === 0 ? <Badge>none (ablation)</Badge> : `${s.layers} layer${s.layers === 1 ? "" : "s"}`) },
          { key: "val", label: "Val MRR", align: "right", render: (s) => fmt.num(s.best_val_mrr_weekly) },
          { key: "ep", label: "Epochs", align: "right", render: (s) => s.epochs_run },
        ]} />
        <p className="mt-3 text-xs text-ink-2">
          Training data: {fmt.int(r.counts.train_positive_pairs)} positive pairs ({fmt.int(r.counts.train_positive_duplicate)} duplicates), {fmt.int(r.counts.val_queries)} validation queries;
          graph {fmt.int(r.counts.graph_nodes)} nodes, {fmt.int(r.counts.train_graph_edges_directed)} directed feature edges before the validation boundary. Seeds {r.seeds.join(", ")}; torch {r.environment.torch}, PyG {r.environment.torch_geometric}.
        </p>
      </Card>
    </>
  );
}

export default function ModelsPage() {
  const { data, error, loading } = useApi("/metrics/model");
  const m = data?.data;
  const ppi = m?.ppi_reproduction;
  const an = m?.anomaly;
  const sm = m?.splits?.split_manifest;
  return (
    <>
      <PageHeader title="Model evaluation" subtitle="Every number on this page was computed by the pipeline for the loaded snapshot; nothing is hard-coded." />
      <ErrorBox error={error} />
      {loading && <Loading />}
      {m && !m.link_prediction && <Card><p className="text-sm text-ink-2">No model results for this snapshot{m.splits?.demo ? " (demo dataset)" : ""}.</p></Card>}
      {m?.link_prediction && <LinkPrediction lp={m.link_prediction} />}

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {ppi && (
          <Card title="GraphSAGE benchmark reproduction (PPI)" subtitle={ppi.paper}>
            <Table rows={Object.entries(ppi.summary).map(([k, v]) => ({ k, ...v }))} rowKey={(r) => r.k} columns={[
              { key: "k", label: "Configuration", render: (r) => r.k.replaceAll("_", " ") },
              { key: "mean", label: "Test micro-F1", align: "right", render: (r) => `${fmt.num(r.mean)} ± ${fmt.num(r.std)}` },
              { key: "n", label: "Seeds", align: "right", render: (r) => r.n_seeds },
            ]} />
            <p className="mt-3 text-xs text-ink-2">Paper (Table 1, GraphSAGE-mean, supervised): <strong>{ppi.paper_micro_f1["GraphSAGE-mean"]}</strong>. This is a sanity check of the implementation, not evidence about Ask Different.</p>
          </Card>
        )}
        {an && (
          <Card title="Anomaly detectors" subtitle="Recall on injected anomalies in real series; false-positive rate on simulated null series">
            <Table rows={an.injection} rowKey={(r) => `${r.kind}${r.factor}`} columns={[
              { key: "kind", label: "Injection", render: (r) => `${r.kind} ×${r.factor}` },
              { key: "nb", label: "Neg. binomial", align: "right", render: (r) => fmt.pct(r.negbin_recall) },
              { key: "roll", label: "Rolling z", align: "right", render: (r) => fmt.pct(r.rolling_recall) },
              { key: "cp", label: "Change point", align: "right", render: (r) => fmt.pct(r.changepoint_recall) },
            ]} />
            <p className="mt-3 text-xs text-ink-2">
              Null false-positive rate at nominal 5%: {an.null_calibration.filter((c) => c.expected_band === "all").map((c) => `${c.method} ${fmt.pct(c.fpr, 1)}`).join(" · ")}.
            </p>
          </Card>
        )}
      </div>

      {sm && (
        <Card className="mt-6" title="Data splits & leakage prevention">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Table rows={["train", "val", "test"].map((s) => ({ s, n: sm.counts[s], r: sm.ranges[s] }))} rowKey={(r) => r.s} columns={[
              { key: "s", label: "Split" },
              { key: "n", label: "Questions", align: "right", render: (r) => fmt.int(r.n) },
              { key: "r", label: "Created", render: (r) => `${fmt.date(r.r[0])} → ${fmt.date(r.r[1])}` },
            ]} />
            <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
              {(data.leakage_controls || []).map((c) => <li key={c}>{c}</li>)}
            </ul>
          </div>
        </Card>
      )}
    </>
  );
}
