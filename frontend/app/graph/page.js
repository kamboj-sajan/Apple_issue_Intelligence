"use client";

import { Suspense, useCallback, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Background, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { apiGet, useApi } from "@/lib/api";
import { Badge, Card, ErrorBox, Field, Loading, PageHeader, fmt, inputClass } from "@/components/ui";

const NODE_TYPES = ["question", "tag", "device", "os", "error", "tech", "category"];
const EDGE_TYPES = [
  "QUESTION_HAS_TAG", "QUESTION_MENTIONS_DEVICE", "QUESTION_MENTIONS_VERSION", "QUESTION_CONTAINS_ERROR",
  "QUESTION_MENTIONS_TECH", "QUESTION_IN_CATEGORY", "QUESTION_SEMANTICALLY_SIMILAR",
  "QUESTION_TEMPORALLY_NEAR_QUESTION", "QUESTION_DUPLICATE_OF", "QUESTION_LINKS_TO", "ISSUE_SPIKED_DURING_WINDOW",
];
// Identity is always carried by the type label inside the node; colour is a secondary cue (4 validated slots).
const TYPE_COLOR = { question: "var(--series-1)", tag: "var(--series-2)", device: "var(--series-3)", os: "var(--series-4)" };
const MAX_RENDERED = 300;

function layout(graph, centerId) {
  // Radial layout: centre, then rings by BFS distance; children placed near their parent's angle.
  const adj = new Map();
  graph.edges.forEach((e) => {
    [[e.source, e.target], [e.target, e.source]].forEach(([a, b]) => {
      if (!adj.has(a)) adj.set(a, []);
      adj.get(a).push(b);
    });
  });
  const dist = new Map([[centerId, 0]]);
  const parent = new Map();
  const queue = [centerId];
  while (queue.length) {
    const u = queue.shift();
    (adj.get(u) || []).forEach((v) => {
      if (!dist.has(v)) { dist.set(v, dist.get(u) + 1); parent.set(v, u); queue.push(v); }
    });
  }
  const rings = new Map();
  graph.nodes.forEach((n) => {
    const d = dist.has(n.node_id) ? dist.get(n.node_id) : 3;
    if (!rings.has(d)) rings.set(d, []);
    rings.get(d).push(n.node_id);
  });
  const pos = new Map([[centerId, { x: 0, y: 0, a: 0 }]]);
  [...rings.keys()].sort().forEach((d) => {
    if (d === 0) return;
    const ids = rings.get(d).sort((a, b) => (pos.get(parent.get(a))?.a || 0) - (pos.get(parent.get(b))?.a || 0));
    const r = Math.max(300 * d, (ids.length * 200) / (2 * Math.PI)); // >= ~200px of arc per 170px-wide node
    ids.forEach((id, i) => {
      const a = (2 * Math.PI * i) / ids.length;
      pos.set(id, { x: r * Math.cos(a), y: r * Math.sin(a), a });
    });
  });
  return pos;
}

function GraphInner() {
  const params = useSearchParams();
  const [center, setCenter] = useState(params.get("node") || "tag:wifi");
  const [input, setInput] = useState(center);
  // Optional URL parameters make a view shareable: ?node=…&depth=1|2&limit=5..100&max=10..300
  const intParam = (k, d, lo, hi) => Math.min(hi, Math.max(lo, parseInt(params.get(k) || "", 10) || d));
  const [depth, setDepth] = useState(intParam("depth", 1, 1, 2));
  const [perNode, setPerNode] = useState(intParam("limit", 20, 5, 100));
  const [maxNodes, setMaxNodes] = useState(intParam("max", 80, 10, 300));
  const [nodeTypes, setNodeTypes] = useState(new Set(NODE_TYPES));
  const [edgeTypes, setEdgeTypes] = useState(new Set(EDGE_TYPES));
  const [extra, setExtra] = useState({ nodes: [], edges: [] });
  const [selected, setSelected] = useState(null);
  const [expandError, setExpandError] = useState(null);

  const query = {
    depth, limit: perNode, max_nodes: maxNodes,
    node_types: [...nodeTypes].join(","), edge_types: [...edgeTypes].join(","),
  };
  const nb = useApi(`/graph/neighbors/${encodeURIComponent(center)}`, query);
  const search = useApi(input && input !== center && input.length > 1 ? "/graph/search" : null, { q: input, limit: 8 });

  const merged = useMemo(() => {
    const base = nb.data?.data;
    if (!base) return null;
    const nodes = new Map([...base.nodes, ...extra.nodes].map((n) => [n.node_id, n]));
    const edges = new Map([...base.edges, ...extra.edges].map((e) => [`${e.source}|${e.target}|${e.edge_type}`, e]));
    const keep = new Set([...nodes.keys()].slice(0, MAX_RENDERED));
    return {
      nodes: [...nodes.values()].filter((n) => keep.has(n.node_id)),
      edges: [...edges.values()].filter((e) => keep.has(e.source) && keep.has(e.target)),
      truncated: base.truncated || nodes.size > MAX_RENDERED,
      center: base.center,
    };
  }, [nb.data, extra]);

  const flow = useMemo(() => {
    if (!merged) return { nodes: [], edges: [] };
    const pos = layout(merged, merged.center.node_id);
    return {
      nodes: merged.nodes.map((n) => {
        const p = pos.get(n.node_id) || { x: 0, y: 0 };
        const color = TYPE_COLOR[n.node_type];
        return {
          id: n.node_id, position: { x: p.x, y: p.y }, data: {
            label: (
              <div className="text-left">
                <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>{n.node_type}</div>
                <div className="line-clamp-2 text-[11px] leading-tight">{n.label || n.node_id}</div>
              </div>
            ),
          },
          style: {
            width: 170, padding: 6, borderRadius: 8, background: "var(--surface)", color: "var(--text-primary)",
            border: `2px solid ${color || "var(--axis)"}`,
            boxShadow: n.node_id === merged.center.node_id ? "0 0 0 3px var(--accent)" : "none",
          },
        };
      }),
      edges: merged.edges.map((e) => ({
        id: `${e.source}|${e.target}|${e.edge_type}`, source: e.source, target: e.target,
        label: e.is_label_edge ? (e.edge_type === "QUESTION_DUPLICATE_OF" ? "duplicate" : "linked") : undefined,
        style: { stroke: e.is_label_edge ? "var(--text-primary)" : "var(--axis)", strokeWidth: e.is_label_edge ? 2 : 1,
          strokeDasharray: e.is_label_edge ? "6 4" : undefined },
        labelStyle: { fontSize: 10, fill: "var(--text-primary)" },
        labelBgStyle: { fill: "var(--surface)" },
        labelBgPadding: [4, 2],
      })),
    };
  }, [merged]);

  const expand = useCallback(async (nodeId) => {
    setExpandError(null);
    try {
      const r = await apiGet(`/graph/neighbors/${encodeURIComponent(nodeId)}`, { ...query, depth: 1 });
      setExtra((x) => ({ nodes: [...x.nodes, ...r.data.nodes], edges: [...x.edges, ...r.data.edges] }));
    } catch (err) {
      setExpandError(err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth, perNode, maxNodes, nodeTypes, edgeTypes]);

  const go = (id) => { setCenter(id); setInput(id); setExtra({ nodes: [], edges: [] }); setSelected(null); };
  const toggle = (set, setter, v) => { const s = new Set(set); s.has(v) ? s.delete(v) : s.add(v); setter(s); setExtra({ nodes: [], edges: [] }); };
  const selNode = merged?.nodes.find((n) => n.node_id === selected);
  const selEdges = merged?.edges.filter((e) => e.source === selected || e.target === selected) || [];

  return (
    <>
      <PageHeader title="Knowledge graph"
        subtitle="Questions linked to tags, devices, OS versions, error codes, categories and to earlier similar questions. Dashed dark edges are known duplicate/link relations (prediction labels, never used as model input). Views are capped to keep rendering responsive." />
      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Node id or search">
            <input className={`${inputClass} w-72`} value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && go(input.trim())} placeholder="tag:wifi, os:macos:14, question:485393" />
          </Field>
          <button className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white" onClick={() => go(input.trim())}>Show</button>
          <Field label="Depth"><select className={inputClass} value={depth} onChange={(e) => setDepth(+e.target.value)}><option value={1}>1 hop</option><option value={2}>2 hops</option></select></Field>
          <Field label={`Edges per node (${perNode})`}><input type="range" min="5" max="100" value={perNode} onChange={(e) => setPerNode(+e.target.value)} /></Field>
          <Field label={`Max nodes (${maxNodes})`}><input type="range" min="10" max="300" step="10" value={maxNodes} onChange={(e) => setMaxNodes(+e.target.value)} /></Field>
        </div>
        {search.data?.data?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            {search.data.data.map((n) => (
              <button key={n.node_id} onClick={() => go(n.node_id)} className="rounded-md border border-line px-2 py-1 hover:bg-surface-2">
                <span className="text-muted">{n.node_type}</span> {n.label?.slice(0, 50)} <span className="text-muted">({n.degree})</span>
              </button>
            ))}
          </div>
        )}
        <details className="mt-3 text-xs text-ink-2">
          <summary className="cursor-pointer">Filter node and edge types</summary>
          <div className="mt-2 flex flex-wrap gap-3">
            {NODE_TYPES.map((t) => (
              <label key={t} className="flex items-center gap-1"><input type="checkbox" checked={nodeTypes.has(t)} onChange={() => toggle(nodeTypes, setNodeTypes, t)} />{t}</label>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-3">
            {EDGE_TYPES.map((t) => (
              <label key={t} className="flex items-center gap-1"><input type="checkbox" checked={edgeTypes.has(t)} onChange={() => toggle(edgeTypes, setEdgeTypes, t)} />{t.toLowerCase()}</label>
            ))}
          </div>
        </details>
      </Card>

      <ErrorBox error={nb.error || expandError} />
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <div className="h-[620px] rounded-xl border border-line bg-surface">
          {nb.loading && <Loading />}
          {merged && (
            <ReactFlow nodes={flow.nodes} edges={flow.edges} fitView minZoom={0.1} nodesConnectable={false} colorMode="system"
              onNodeClick={(_, n) => setSelected(n.id)} onNodeDoubleClick={(_, n) => expand(n.id)} proOptions={{ hideAttribution: false }}>
              <Background color="var(--grid)" gap={24} />
              <Controls showInteractive={false} />
            </ReactFlow>
          )}
        </div>
        <Card title="Selection" subtitle={merged ? `${merged.nodes.length} nodes · ${merged.edges.length} edges${merged.truncated ? " · truncated by limits" : ""}` : ""}>
          <div className="mb-3 flex flex-wrap gap-2 text-xs">
            {Object.entries(TYPE_COLOR).map(([t, c]) => (
              <span key={t} className="inline-flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm border-2" style={{ borderColor: c }} />{t}</span>
            ))}
            <span className="inline-flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm border-2" style={{ borderColor: "var(--axis)" }} />other</span>
          </div>
          {!selNode && <p className="text-sm text-ink-2">Click a node to inspect it; double-click to expand its neighbours.</p>}
          {selNode && (
            <div className="space-y-3 text-sm">
              <div><Badge>{selNode.node_type}</Badge> <span className="font-mono text-xs">{selNode.node_id}</span></div>
              <div>{selNode.label}</div>
              <div className="text-xs text-ink-2">degree {fmt.int(selNode.degree)}</div>
              <div className="flex flex-wrap gap-2">
                <button className="rounded-md border border-line px-2 py-1 text-xs" onClick={() => expand(selNode.node_id)}>Expand</button>
                <button className="rounded-md border border-line px-2 py-1 text-xs" onClick={() => go(selNode.node_id)}>Centre here</button>
                {selNode.link && <a className="rounded-md border border-line px-2 py-1 text-xs text-accent" href={selNode.link} target="_blank" rel="noreferrer">Open on Ask Different ↗</a>}
              </div>
              <ul className="max-h-72 space-y-1 overflow-y-auto text-xs">
                {selEdges.map((e) => (
                  <li key={`${e.source}|${e.target}|${e.edge_type}`} className="border-b border-line pb-1">
                    <span className="font-medium">{e.edge_type.toLowerCase()}</span> · w={fmt.num(e.weight, 2)}
                    <div className="text-ink-2">{e.source === selected ? e.target : e.source}</div>
                    {e.evidence && <div className="text-muted">{e.evidence}</div>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

export default function GraphPage() {
  return <Suspense fallback={<Loading />}><GraphInner /></Suspense>;
}
