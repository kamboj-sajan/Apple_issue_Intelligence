"""Read-side queries used by the API routes."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select, text
from sqlalchemy.orm import Session

from backend.app.core.errors import APIError, not_found
from backend.app.db.models import (
    Anomaly,
    AnomalySeries,
    DatasetSnapshot,
    GraphEdge,
    GraphNode,
    ModelMetric,
    Question,
    QuestionEntity,
    RelatedQuestion,
    TermIdf,
)
from backend.app.schemas.common import Meta
from ml.retrieval.bm25 import tokenize

LABEL_EDGE_TYPES = {"QUESTION_DUPLICATE_OF", "QUESTION_LINKS_TO"}
METHOD_DESCRIPTIONS = {
    "bm25": "BM25 lexical relevance (k1=1.5, b=0.75) between question texts; higher = more shared distinctive words.",
    "embedding": "Cosine similarity of all-MiniLM-L6-v2 sentence embeddings (title + body), in [-1, 1].",
    "graphsage": "GraphSAGE link score (MLP decoder + learned text-similarity residual), unbounded. Served as a "
    "re-ranking of the 100 most similar earlier questions by embedding, using the graph as of the snapshot.",
}


def meta(session: Session) -> Meta:
    snap = session.scalars(select(DatasetSnapshot).limit(1)).first()
    if snap is None:
        return Meta(data_source="empty")
    return Meta(data_source="demo" if snap.is_demo else "real", snapshot_id=snap.snapshot_id, loaded_at=snap.loaded_at)


def snapshot(session: Session) -> DatasetSnapshot | None:
    return session.scalars(select(DatasetSnapshot).limit(1)).first()


# --------------------------------------------------------------------------------- questions


def question_filters(
    q: str | None,
    start_date: date | None,
    end_date: date | None,
    tag: str | None,
    device: str | None,
    os: str | None,
    category: str | None,
) -> list:
    conds = []
    if q:
        if q.strip().isdigit():
            conds.append(Question.question_id == int(q.strip()))
        else:
            conds.append(Question.title.ilike(f"%{q.strip()}%"))
    if start_date:
        conds.append(Question.created_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        conds.append(Question.created_at < datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
    if tag:
        conds.append(Question.tags.any(tag))
    if device:
        conds.append(Question.devices.any(device))
    if os:
        conds.append(Question.os_versions.any(os))
    if category:
        conds.append(Question.categories.any(category))
    return conds


def list_questions(session: Session, conds: list, sort: str, limit: int, offset: int) -> tuple[list[Question], int]:
    order = {
        "newest": Question.created_at.desc(),
        "oldest": Question.created_at.asc(),
        "score": Question.score.desc().nulls_last(),
        "views": Question.view_count.desc().nulls_last(),
    }[sort]
    base = select(Question).where(Question.in_research_window, *conds)
    total = session.scalar(select(func.count()).select_from(base.subquery()))
    rows = session.scalars(base.order_by(order, Question.question_id).limit(limit).offset(offset)).all()
    return list(rows), int(total or 0)


def get_question(session: Session, question_id: int) -> Question:
    row = session.get(Question, question_id)
    if row is None:
        raise not_found("question", question_id)
    return row


def question_entities(session: Session, question_id: int) -> list[QuestionEntity]:
    return list(session.scalars(select(QuestionEntity).where(QuestionEntity.question_id == question_id)).all())


# --------------------------------------------------------------------------------- anomalies


def anomaly_filters(
    group_type: str | None,
    group_value: str | None,
    device: str | None,
    os: str | None,
    tag: str | None,
    start_date: date | None,
    end_date: date | None,
    min_score: float | None,
    signal_type: str | None,
) -> list:
    conds = []
    if group_type:
        conds.append(Anomaly.group_type == group_type)
    if group_value:
        conds.append(Anomaly.group_value == group_value)
    for gt, val in (("device", device), ("os", os), ("tag", tag)):
        if val:
            conds.append(and_(Anomaly.group_type == gt, Anomaly.group_value == val))
    if start_date:
        conds.append(Anomaly.week_start >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        conds.append(Anomaly.week_start <= datetime.combine(end_date, datetime.min.time()))
    if min_score is not None:
        conds.append(Anomaly.rank_score >= min_score)
    if signal_type:
        conds.append(Anomaly.signal_type == signal_type)
    return conds


def list_anomalies(session: Session, conds: list, sort: str, limit: int, offset: int):
    order = {
        "score": Anomaly.rank_score.desc(),
        "newest": Anomaly.week_start.desc(),
        "oldest": Anomaly.week_start.asc(),
    }
    base = select(Anomaly).where(*conds)
    total = session.scalar(select(func.count()).select_from(base.subquery()))
    rows = session.scalars(base.order_by(order[sort], Anomaly.anomaly_id).limit(limit).offset(offset)).all()
    return list(rows), int(total or 0)


def series(session: Session, group_type: str, group_value: str, start=None, end=None) -> list[AnomalySeries]:
    stmt = select(AnomalySeries).where(AnomalySeries.group_type == group_type, AnomalySeries.group_value == group_value)
    if start is not None:
        stmt = stmt.where(AnomalySeries.week_start >= start)
    if end is not None:
        stmt = stmt.where(AnomalySeries.week_start <= end)
    return list(session.scalars(stmt.order_by(AnomalySeries.week_start)).all())


def anomaly_detail(session: Session, anomaly_id: str) -> dict[str, Any]:
    a = session.get(Anomaly, anomaly_id)
    if a is None:
        raise not_found("anomaly", anomaly_id)
    pts = series(
        session, a.group_type, a.group_value, a.week_start - timedelta(weeks=52), a.week_start + timedelta(weeks=26)
    )
    ev = session.scalars(select(Question).where(Question.question_id.in_(a.evidence_question_ids or [-1]))).all()
    explanation = {
        "triggered_window": f"week starting {a.week_start.date()} (Monday, UTC)",
        "baseline_period": f"{a.baseline_start.date()} to {a.baseline_end.date()} (26 preceding weeks)",
        "observed_vs_expected": f"{a.observed} questions observed vs {a.expected:.1f} expected "
        f"({a.rel_change * 100:+.0f}%), given this group's share of {a.site_total} site questions that week",
        "statistical_test": "Negative-binomial upper-tail test on the group's count with a site-volume offset; "
        "Benjamini-Hochberg FDR across all groups tested that week.",
        "signal_type": {
            "spike": "short-lived increase; no sustained level change detected",
            "sustained": "an increasing change point starts near this week (PELT, Poisson cost)",
            "too_recent": "fewer than 3 weeks of follow-up data; cannot yet tell spike from shift",
        }.get(a.signal_type, a.signal_type),
        "rolling_detector": "agrees (robust z >= 3)" if a.rolling_agrees else "does not flag this week",
        "data_quality": [
            f for f, on in (("partial week", a.partial_week), ("unusually low site volume", a.low_site_volume)) if on
        ]
        or ["no data-quality flags"],
    }
    return {"anomaly": a, "series": pts, "evidence": list(ev), "explanation": explanation}


def trending(session: Session, weeks: int, limit: int) -> dict[str, Any]:
    end = session.scalar(select(func.max(Question.created_at)).where(Question.in_research_window))
    if end is None:
        return {"window_end": None, "categories": [], "signals": []}
    recent_start, prev_start = end - timedelta(weeks=weeks), end - timedelta(weeks=2 * weeks)
    rows = session.execute(
        text(
            """
            SELECT cat,
                   count(*) FILTER (WHERE created_at > :recent_start) AS recent,
                   count(*) FILTER (WHERE created_at <= :recent_start) AS previous
            FROM questions, unnest(categories) AS cat
            WHERE in_research_window AND created_at > :prev_start
            GROUP BY cat
            """
        ),
        {"recent_start": recent_start, "prev_start": prev_start},
    ).all()
    totals = session.execute(
        text(
            """SELECT count(*) FILTER (WHERE created_at > :rs), count(*) FILTER (WHERE created_at <= :rs)
               FROM questions WHERE in_research_window AND created_at > :ps"""
        ),
        {"rs": recent_start, "ps": prev_start},
    ).one()
    n_recent, n_prev = max(totals[0], 1), max(totals[1], 1)
    cats = [
        {
            "category": c,
            "recent": r,
            "previous": p,
            "recent_share": r / n_recent,
            "previous_share": p / n_prev,
            "change": r / n_recent - p / n_prev,
        }
        for c, r, p in rows
    ]
    cats.sort(key=lambda d: -d["recent"])
    sig = session.scalars(
        select(Anomaly).order_by(Anomaly.week_start.desc(), Anomaly.rank_score.desc()).limit(limit)
    ).all()
    return {"window_end": end, "categories": cats[:limit], "signals": list(sig)}


# --------------------------------------------------------------------------------- graph


def neighborhood(
    session: Session,
    node_id: str,
    depth: int,
    limit: int,
    max_nodes: int,
    node_types: set[str] | None,
    edge_types: set[str] | None,
) -> dict[str, Any]:
    center = session.get(GraphNode, node_id)
    if center is None:
        raise not_found("graph node", node_id)
    seen = {node_id}
    frontier = [node_id]
    edges_out: list[dict] = []
    truncated = False
    for _ in range(depth):
        nxt = []
        for u in frontier:
            stmt = select(GraphEdge).where(or_(GraphEdge.src == u, GraphEdge.dst == u))
            if edge_types:
                stmt = stmt.where(GraphEdge.edge_type.in_(edge_types))
            if node_types:
                other = func.split_part(case((GraphEdge.src == u, GraphEdge.dst), else_=GraphEdge.src), ":", 1)
                stmt = stmt.where(other.in_(node_types))
            rows = session.scalars(stmt.order_by(GraphEdge.weight.desc()).limit(limit + 1)).all()
            if len(rows) > limit:
                truncated, rows = True, rows[:limit]
            for e in rows:
                v = e.dst if e.src == u else e.src
                if v not in seen and len(seen) >= max_nodes:
                    truncated = True
                    continue
                edges_out.append(
                    {
                        "source": e.src,
                        "target": e.dst,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                        "evidence": e.evidence,
                        "is_label_edge": e.edge_type in LABEL_EDGE_TYPES,
                    }
                )
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    nodes = session.scalars(select(GraphNode).where(GraphNode.node_id.in_(seen))).all()
    links = _question_links(session, [n.node_id for n in nodes])
    uniq = {(e["source"], e["target"], e["edge_type"]): e for e in edges_out}
    return {
        "center": center,
        "nodes": list(nodes),
        "edges": list(uniq.values()),
        "truncated": truncated,
        "links": links,
    }


def _question_links(session: Session, node_ids: list[str]) -> dict[str, str | None]:
    qids = [int(n.split(":", 1)[1]) for n in node_ids if n.startswith("question:")]
    if not qids:
        return {}
    rows = session.execute(select(Question.question_id, Question.link).where(Question.question_id.in_(qids))).all()
    return {f"question:{q}": link for q, link in rows}


def search_nodes(session: Session, q: str, node_type: str | None, limit: int) -> list[GraphNode]:
    stmt = select(GraphNode).where(or_(GraphNode.label.ilike(f"%{q}%"), GraphNode.node_id == q))
    if node_type:
        stmt = stmt.where(GraphNode.node_type == node_type)
    return list(session.scalars(stmt.order_by(GraphNode.degree.desc()).limit(limit)).all())


# --------------------------------------------------------------------------------- related questions


def related(session: Session, question_id: int, methods: list[str], k: int) -> dict[str, Any]:
    q = get_question(session, question_id)
    out = []
    for m in methods:
        rows = session.execute(
            select(RelatedQuestion.rank, RelatedQuestion.score, Question)
            .join(Question, Question.question_id == RelatedQuestion.related_id)
            .where(RelatedQuestion.question_id == question_id, RelatedQuestion.method == m, RelatedQuestion.rank <= k)
            .order_by(RelatedQuestion.rank)
        ).all()
        items = [
            {"rank": r, "score": s, "question": other, "explanation": explain_pair(session, q, other)}
            for r, s, other in rows
        ]
        out.append({"method": m, "score_description": METHOD_DESCRIPTIONS[m], "items": items})
    return {"question": q, "methods": out}


def explain_pair(session: Session, a: Question, b: Question) -> dict[str, Any]:
    ents = session.execute(
        select(QuestionEntity.question_id, QuestionEntity.entity_type, QuestionEntity.value).where(
            QuestionEntity.question_id.in_([a.question_id, b.question_id]),
            QuestionEntity.entity_type.in_(["device", "os", "error", "tech", "category"]),
        )
    ).all()
    ea = {(t, v) for qid, t, v in ents if qid == a.question_id}
    eb = {(t, v) for qid, t, v in ents if qid == b.question_id}
    shared: dict[str, list[str]] = {}
    for t, v in sorted(ea & eb):
        shared.setdefault(t, []).append(v)

    ta = set(tokenize(f"{a.title} {a.body_excerpt}"))
    tb = set(tokenize(f"{b.title} {b.body_excerpt}"))
    common = list(ta & tb)
    idf = dict(session.execute(select(TermIdf.term, TermIdf.idf).where(TermIdf.term.in_(common or [""]))).all())
    lexical = sorted(({"term": t, "idf": round(idf[t], 2)} for t in common if t in idf), key=lambda d: -d["idf"])[:8]

    cos = session.scalar(
        text(
            "SELECT 1 - (x.embedding <=> y.embedding) FROM questions x, questions y "
            "WHERE x.question_id = :a AND y.question_id = :b AND x.embedding IS NOT NULL AND y.embedding IS NOT NULL"
        ),
        {"a": a.question_id, "b": b.question_id},
    )
    na, nb = f"question:{a.question_id}", f"question:{b.question_id}"
    direct = session.scalars(
        select(GraphEdge).where(
            or_(and_(GraphEdge.src == na, GraphEdge.dst == nb), and_(GraphEdge.src == nb, GraphEdge.dst == na))
        )
    ).all()
    relations = [{"edge_type": e.edge_type, "weight": round(e.weight, 3), "evidence": e.evidence} for e in direct]
    labelled = next(
        (
            {"QUESTION_DUPLICATE_OF": "duplicate", "QUESTION_LINKS_TO": "body_link"}[e.edge_type]
            for e in direct
            if e.edge_type in LABEL_EDGE_TYPES
        ),
        None,
    )
    return {
        "shared_tags": sorted(set(a.tags or []) & set(b.tags or [])),
        "shared_entities": shared,
        "lexical_overlap": lexical,
        "embedding_cosine": float(cos) if cos is not None else None,
        "graph_relations": relations,
        "labelled_relation": labelled,
    }


# --------------------------------------------------------------------------------- metrics / drift


def model_metrics(session: Session) -> dict[str, Any]:
    """Metric payloads by kind; ``loaded_at`` is when the row was written to the database."""
    return {m.kind: {**m.payload, "loaded_at": m.created_at} for m in session.scalars(select(ModelMetric)).all()}


# Where each artifact records when it was computed (inside the metric payloads).
ANALYSIS_TIME_PATHS = {
    "anomaly": ("summary", "created_at"),
    "drift": ("summary", "created_at"),
    "link_prediction": ("report", "created_at"),
    "retrieval": ("manifest", "created_at"),
    "ppi_reproduction": ("created_at",),
}


def latest_analysis(metrics: dict[str, Any]) -> datetime | None:
    """Most recent analysis timestamp across artifacts (ISO strings parsed to aware datetimes)."""
    times = []
    for kind, path in ANALYSIS_TIME_PATHS.items():
        v: Any = metrics.get(kind)
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
        if isinstance(v, str):
            times.append(datetime.fromisoformat(v))
        elif isinstance(v, datetime):
            times.append(v)
    return max(times, default=None)


def require(value, what: str):
    if value is None:
        raise APIError(404, "not_available", f"{what} has not been computed for the loaded snapshot")
    return value
