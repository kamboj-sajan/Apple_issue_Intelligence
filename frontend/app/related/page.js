"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useApi } from "@/lib/api";
import { Badge, Card, ErrorBox, Field, Loading, PageHeader, QuestionLink, fmt, inputClass } from "@/components/ui";

const METHOD_NAMES = { bm25: "BM25 (lexical)", embedding: "Embedding cosine", graphsage: "GraphSAGE" };

function Explanation({ ex }) {
  const ents = Object.entries(ex.shared_entities || {});
  return (
    <dl className="mt-2 grid grid-cols-[96px_1fr] gap-x-2 gap-y-1 text-xs text-ink-2">
      <dt>Shared tags</dt><dd>{ex.shared_tags.length ? ex.shared_tags.join(", ") : "none"}</dd>
      <dt>Shared entities</dt><dd>{ents.length ? ents.map(([t, v]) => `${t}: ${v.join(", ")}`).join(" · ") : "none"}</dd>
      <dt>Shared terms</dt><dd>{ex.lexical_overlap.length ? ex.lexical_overlap.map((t) => t.term).join(", ") : "none"}</dd>
      <dt>Embedding cos</dt><dd className="tabular">{fmt.num(ex.embedding_cosine)}</dd>
      <dt>Graph</dt>
      <dd>{ex.graph_relations.length ? ex.graph_relations.map((r) => `${r.edge_type.toLowerCase()} (${fmt.num(r.weight, 2)})`).join(" · ") : "no direct edge"}</dd>
    </dl>
  );
}

function RelatedView({ id }) {
  const { data, error, loading } = useApi(`/related-questions/${id}`, { k: 5 });
  const d = data?.data;
  return (
    <>
      <ErrorBox error={error} />
      {loading && <Loading />}
      {d && (
        <>
          <Card className="mt-6" title={d.question.title}
            subtitle={`#${d.question.question_id} · ${fmt.date(d.question.created_at)} · ${d.question.tags.join(", ")}`}
            right={<QuestionLink id={d.question.question_id} link={d.question.link} title="Open" />}>
            <p className="text-xs text-ink-2">{d.note}</p>
          </Card>
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
            {d.methods.map((m) => (
              <Card key={m.method} title={METHOD_NAMES[m.method]} subtitle={m.score_description}>
                {m.items.length === 0 && <p className="text-sm text-muted">No earlier candidates.</p>}
                <ol className="space-y-4">
                  {m.items.map((it) => (
                    <li key={it.question_id} className="border-b border-line pb-3 last:border-0">
                      <div className="flex items-start gap-2 text-sm">
                        <span className="tabular w-5 shrink-0 text-muted">{it.rank}.</span>
                        <div className="min-w-0">
                          <QuestionLink id={it.question_id} title={it.title} link={it.link} />
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
                            <span>{fmt.date(it.created_at)}</span>
                            <span className="tabular">score {fmt.num(it.score, 3)}</span>
                            {it.explanation.labelled_relation && <Badge tone="good" title="A known relation between the two questions; not used by any method">known {it.explanation.labelled_relation}</Badge>}
                          </div>
                          <Explanation ex={it.explanation} />
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              </Card>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function RelatedInner() {
  const router = useRouter();
  const params = useSearchParams();
  const id = params.get("id");
  const [q, setQ] = useState("");
  const results = useApi(q.trim().length > 1 ? "/questions" : null, { q: q.trim(), limit: 8 });
  const pick = (qid) => { router.replace(`/related?id=${qid}`, { scroll: false }); setQ(""); };
  return (
    <>
      <PageHeader title="Related questions"
        subtitle="Earlier questions that each method considers related, with the evidence behind each recommendation. GraphSAGE is served as a re-ranking of the 100 most similar earlier questions by embedding." />
      <Card>
        <Field label="Search by title or question id">
          <input className={`${inputClass} w-full max-w-xl`} value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && /^\d+$/.test(q.trim()) && pick(q.trim())} placeholder="e.g. Time Machine backup fails, or 485393" />
        </Field>
        <ErrorBox error={results.error} />
        {results.data?.items?.length > 0 && (
          <ul className="mt-3 divide-y divide-line text-sm">
            {results.data.items.map((r) => (
              <li key={r.question_id}>
                <button className="w-full py-2 text-left hover:bg-surface-2" onClick={() => pick(r.question_id)}>
                  {r.title} <span className="text-xs text-muted">#{r.question_id} · {fmt.date(r.created_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>
      {id ? <RelatedView id={id} /> : <p className="mt-6 text-sm text-ink-2">Pick a question to compare BM25, embedding and GraphSAGE recommendations.</p>}
    </>
  );
}

export default function RelatedPage() {
  return <Suspense fallback={<Loading />}><RelatedInner /></Suspense>;
}
