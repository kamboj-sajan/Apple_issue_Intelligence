"""HTTP routes. Every response includes ``meta.data_source`` ('real' | 'demo' | 'empty')."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query
from sqlalchemy import func, select, text

from backend.app.api.deps import JobsDep, Limit, Offset, SessionDep, csv_set
from backend.app.core.errors import APIError, not_found
from backend.app.db.models import Anomaly, AnomalySeries, DriftClusterMovement, DriftMetric, DriftNewTerm, Job, Question
from backend.app.schemas.common import Envelope, Page
from backend.app.schemas.domain import (
    AnomalyDetail,
    AnomalySummary,
    Attribution,
    EntityOut,
    EvidenceQuestion,
    GraphEdgeOut,
    GraphNodeOut,
    JobAnalyze,
    JobIngest,
    JobOut,
    Neighborhood,
    QuestionDetail,
    QuestionSummary,
    RelatedResponse,
    SeriesPoint,
    Trending,
)
from backend.app.services import jobs as J
from backend.app.services import queries as Q

router = APIRouter()
NODE_TYPES = {"question", "tag", "device", "os", "error", "tech", "category", "week"}
METHODS = ("bm25", "embedding", "graphsage")
SIGNAL_NOTE = (
    "Signals are statistically unusual increases in posted questions. They are candidate discussion patterns, "
    "not confirmed defects, and do not measure the number of affected users."
)


def _q_summary(q: Question) -> QuestionSummary:
    return QuestionSummary.model_validate(q, from_attributes=True)


@router.get("/health", tags=["system"])
def health(session: SessionDep):
    session.execute(text("SELECT 1"))
    m = Q.meta(session)
    return {"status": "ok", "database": "ok", "meta": m}


@router.get("/overview", tags=["system"])
def overview(session: SessionDep):
    snap = Q.snapshot(session)
    m = Q.meta(session)
    if snap is None:
        return {"data": None, "meta": m}
    n_signals = session.scalar(select(func.count()).select_from(Anomaly))
    latest_q = session.scalar(select(func.max(Question.created_at)).where(Question.in_research_window))
    cats = session.execute(
        text(
            "SELECT primary_category, count(*) FROM questions WHERE in_research_window AND created_at > :s "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 8"
        ),
        {"s": latest_q.replace(year=latest_q.year - 1) if latest_q else None},
    ).all()
    metrics = Q.model_metrics(session)
    return {
        "data": {
            "total_questions": snap.n_questions,
            "questions_including_context": session.scalar(select(func.count()).select_from(Question)),
            "anomaly_signals": n_signals,
            "created_range": [snap.created_range_start, snap.created_range_end],
            "latest_question_at": latest_q,
            "loaded_at": snap.loaded_at,
            "pipeline_version": snap.pipeline_version,
            "latest_analysis": Q.latest_analysis(metrics),
            "most_active_categories_last_12_months": [{"category": c, "questions": n} for c, n in cats],
            "collection": snap.collection,
            "quality": (snap.manifest or {}).get("quality", {}),
        },
        "meta": m,
    }


# ------------------------------------------------------------------------------------ questions


@router.get("/questions", response_model=Page[QuestionSummary], tags=["questions"])
def list_questions(
    session: SessionDep,
    limit: Limit = 20,
    offset: Offset = 0,
    q: Annotated[str | None, Query(max_length=200, description="title substring or question id")] = None,
    start_date: date | None = None,
    end_date: date | None = None,
    tag: str | None = None,
    device: str | None = None,
    os: str | None = None,
    category: str | None = None,
    sort: Literal["newest", "oldest", "score", "views"] = "newest",
):
    if start_date and end_date and start_date > end_date:
        raise APIError(422, "validation_error", "start_date must be on or before end_date")
    rows, total = Q.list_questions(
        session, Q.question_filters(q, start_date, end_date, tag, device, os, category), sort, limit, offset
    )
    return Page(items=[_q_summary(r) for r in rows], total=total, limit=limit, offset=offset, meta=Q.meta(session))


@router.get("/questions/{question_id}", response_model=Envelope[QuestionDetail], tags=["questions"])
def get_question(session: SessionDep, question_id: Annotated[int, Path(ge=1)]):
    q = Q.get_question(session, question_id)
    data = QuestionDetail(
        **_q_summary(q).model_dump(),
        body_excerpt=q.body_excerpt,
        error_codes=q.error_codes or [],
        categories=q.categories or [],
        duplicate_of=q.duplicate_of or [],
        split=q.split,
        in_research_window=q.in_research_window,
        entities=[EntityOut.model_validate(e, from_attributes=True) for e in Q.question_entities(session, question_id)],
        attribution=Attribution(
            author_display_name=q.author_display_name, author_link=q.author_link, content_license=q.content_license
        ),
    )
    return Envelope(data=data, meta=Q.meta(session))


# ------------------------------------------------------------------------------------ issues & anomalies


@router.get("/issues/trending", response_model=Envelope[Trending], tags=["anomalies"])
def issues_trending(session: SessionDep, weeks: Annotated[int, Query(ge=1, le=104)] = 12, limit: Limit = 10):
    t = Q.trending(session, weeks, limit)
    data = Trending(
        window_weeks=weeks,
        window_end=t["window_end"],
        categories=t["categories"],
        recent_signals=[AnomalySummary.model_validate(a, from_attributes=True) for a in t["signals"]],
        note=SIGNAL_NOTE,
    )
    return Envelope(data=data, meta=Q.meta(session))


@router.get("/issues/series", response_model=Envelope[list[SeriesPoint]], tags=["anomalies"])
def issue_series(
    session: SessionDep,
    group_type: Annotated[str, Query(max_length=16)],
    group_value: Annotated[str, Query(max_length=200)],
    start_date: date | None = None,
    end_date: date | None = None,
):
    pts = Q.series(session, group_type, group_value, start_date, end_date)
    if not pts:
        raise not_found("series for", f"{group_type}:{group_value}")
    return Envelope(data=[SeriesPoint.model_validate(p, from_attributes=True) for p in pts], meta=Q.meta(session))


@router.get("/issues/groups", tags=["anomalies"])
def issue_groups(
    session: SessionDep,
    group_type: str | None = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
    limit: Limit = 50,
):
    """Issue groups that have a stored weekly series (for chart pickers), largest first."""
    total = func.sum(AnomalySeries.observed).label("total")
    stmt = select(AnomalySeries.group_type, AnomalySeries.group_value, total).group_by(
        AnomalySeries.group_type, AnomalySeries.group_value
    )
    if group_type:
        stmt = stmt.where(AnomalySeries.group_type == group_type)
    if q:
        stmt = stmt.where(AnomalySeries.group_value.ilike(f"%{q}%"))
    rows = session.execute(stmt.order_by(total.desc()).limit(limit)).all()
    return {"data": [{"group_type": a, "group_value": b, "total": int(c)} for a, b, c in rows], "meta": Q.meta(session)}


@router.get("/anomalies", response_model=Page[AnomalySummary], tags=["anomalies"])
def list_anomalies(
    session: SessionDep,
    limit: Limit = 20,
    offset: Offset = 0,
    group_type: str | None = None,
    group_value: str | None = None,
    device: str | None = None,
    os: str | None = None,
    tag: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_score: Annotated[float | None, Query(ge=0)] = None,
    signal_type: Literal["spike", "sustained", "too_recent"] | None = None,
    sort: Literal["score", "newest", "oldest"] = "score",
):
    conds = Q.anomaly_filters(group_type, group_value, device, os, tag, start_date, end_date, min_score, signal_type)
    rows, total = Q.list_anomalies(session, conds, sort, limit, offset)
    return Page(
        items=[AnomalySummary.model_validate(r, from_attributes=True) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
        meta=Q.meta(session),
    )


@router.get("/anomalies/{anomaly_id}", response_model=Envelope[AnomalyDetail], tags=["anomalies"])
def get_anomaly(session: SessionDep, anomaly_id: Annotated[str, Path(max_length=32)]):
    d = Q.anomaly_detail(session, anomaly_id)
    a = d["anomaly"]
    data = AnomalyDetail(
        **AnomalySummary.model_validate(a, from_attributes=True).model_dump(),
        p_value=a.p_value,
        nb_score=a.nb_score,
        rolling_z=a.rolling_z,
        overdispersion_alpha=a.overdispersion_alpha,
        site_total=a.site_total,
        baseline_start=a.baseline_start,
        baseline_end=a.baseline_end,
        co_occurring=a.co_occurring or [],
        evidence_questions=[EvidenceQuestion.model_validate(q, from_attributes=True) for q in d["evidence"]],
        series=[SeriesPoint.model_validate(p, from_attributes=True) for p in d["series"]],
        explanation=d["explanation"],
        limitations=a.limitations,
    )
    return Envelope(data=data, meta=Q.meta(session))


# ------------------------------------------------------------------------------------ graph & related


@router.get("/graph/neighbors/{node_id}", response_model=Envelope[Neighborhood], tags=["graph"])
def graph_neighbors(
    session: SessionDep,
    node_id: Annotated[str, Path(max_length=200, description="e.g. question:485393, tag:wifi")],
    depth: Annotated[int, Query(ge=1, le=2)] = 1,
    limit: Annotated[int, Query(ge=1, le=100, description="max edges expanded per node")] = 25,
    max_nodes: Annotated[int, Query(ge=2, le=300)] = 120,
    node_types: Annotated[str | None, Query(description="comma-separated node types to include")] = None,
    edge_types: Annotated[str | None, Query(description="comma-separated edge types to include")] = None,
):
    nt = csv_set(node_types)
    if nt and not nt <= NODE_TYPES:
        raise APIError(422, "validation_error", f"unknown node types: {sorted(nt - NODE_TYPES)}")
    r = Q.neighborhood(session, node_id, depth, limit, max_nodes, nt, csv_set(edge_types))
    links = r["links"]

    def node(n):
        return GraphNodeOut(
            node_id=n.node_id, node_type=n.node_type, label=n.label, degree=n.degree, link=links.get(n.node_id)
        )

    data = Neighborhood(
        center=node(r["center"]),
        nodes=[node(n) for n in r["nodes"]],
        edges=[GraphEdgeOut(**e) for e in r["edges"]],
        truncated=r["truncated"],
        limits={"depth": depth, "edges_per_node": limit, "max_nodes": max_nodes},
    )
    return Envelope(data=data, meta=Q.meta(session))


@router.get("/graph/search", tags=["graph"])
def graph_search(
    session: SessionDep,
    q: Annotated[str, Query(min_length=1, max_length=100)],
    node_type: str | None = None,
    limit: Limit = 20,
):
    rows = Q.search_nodes(session, q, node_type, limit)
    return {
        "data": [GraphNodeOut(node_id=n.node_id, node_type=n.node_type, label=n.label, degree=n.degree) for n in rows],
        "meta": Q.meta(session),
    }


@router.get("/related-questions/{question_id}", response_model=Envelope[RelatedResponse], tags=["related"])
def related_questions(
    session: SessionDep,
    question_id: Annotated[int, Path(ge=1)],
    methods: Annotated[
        str, Query(description="comma-separated: bm25,embedding,graphsage")
    ] = "bm25,embedding,graphsage",
    k: Annotated[int, Query(ge=1, le=10)] = 5,
):
    ms = [m.strip() for m in methods.split(",") if m.strip()]
    bad = [m for m in ms if m not in METHODS]
    if bad or not ms:
        raise APIError(422, "validation_error", f"methods must be a subset of {list(METHODS)}", {"invalid": bad})
    r = Q.related(session, question_id, ms, k)
    data = RelatedResponse(
        question=_q_summary(r["question"]),
        methods=[
            {
                "method": m["method"],
                "score_description": m["score_description"],
                "items": [
                    {
                        "rank": i["rank"],
                        "question_id": i["question"].question_id,
                        "title": i["question"].title,
                        "link": i["question"].link,
                        "created_at": i["question"].created_at,
                        "score": i["score"],
                        "explanation": i["explanation"],
                    }
                    for i in m["items"]
                ],
            }
            for m in r["methods"]
        ],
        note="Recommendations are earlier questions only (the evaluated task). 'labelled_relation' shows a known "
        "duplicate/link between the pair, which none of the methods used as input.",
    )
    return Envelope(data=data, meta=Q.meta(session))


# ------------------------------------------------------------------------------------ metrics & drift


@router.get("/metrics/model", tags=["metrics"])
def metrics_model(session: SessionDep):
    m = Q.model_metrics(session)
    return {
        "data": m,
        "leakage_controls": [
            "Time-based splits; boundaries snapped to Monday 00:00 UTC (see split_manifest).",
            "Candidates for every query are only questions created before it.",
            "Label edges (duplicate / body link) are never used for message passing.",
            "Each test query is embedded with the graph as of its own creation time.",
            "Anchor text of links to other questions is removed from question text (pipeline 0.2.0).",
            "BM25 IDF is fit only on questions before the evaluation boundary.",
        ],
        "meta": Q.meta(session),
    }


@router.get("/drift", tags=["drift"])
def drift(
    session: SessionDep,
    metric: str | None = None,
    start_month: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}$")] = None,
    end_month: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}$")] = None,
    alerts_only: bool = False,
):
    stmt = select(DriftMetric)
    if metric:
        stmt = stmt.where(DriftMetric.metric == metric)
    if start_month:
        stmt = stmt.where(DriftMetric.month >= start_month)
    if end_month:
        stmt = stmt.where(DriftMetric.month <= end_month)
    if alerts_only:
        stmt = stmt.where(DriftMetric.alert)
    rows = session.scalars(stmt.order_by(DriftMetric.month, DriftMetric.metric)).all()
    moves = session.scalars(
        select(DriftClusterMovement)
        .where(DriftClusterMovement.count >= 10)
        .order_by(DriftClusterMovement.month, DriftClusterMovement.share_ratio.desc())
    ).all()
    terms = session.scalars(select(DriftNewTerm).order_by(DriftNewTerm.month, DriftNewTerm.docs.desc())).all()
    metrics = Q.model_metrics(session).get("drift", {})
    cols = (
        "month",
        "metric",
        "status",
        "value",
        "z",
        "p_value",
        "threshold",
        "alert",
        "calibration",
        "n",
        "n_current",
        "reference",
    )
    return {
        "data": {
            "series": [{c: getattr(r, c) for c in cols} for r in rows],
            "cluster_movements": [
                {
                    c: getattr(r, c)
                    for c in (
                        "month",
                        "cluster",
                        "count",
                        "share",
                        "reference_share",
                        "share_ratio",
                        "direction",
                        "top_terms",
                    )
                }
                for r in moves
            ],
            "new_terms": [{"month": t.month, "term": t.term, "docs": t.docs} for t in terms],
            "summary": metrics.get("summary"),
            "novelty_rate": metrics.get("novelty_rate", []),
            "new_cluster_candidates": metrics.get("new_cluster_candidates", []),
            "evaluation": metrics.get("evaluation", {}),
        },
        "meta": Q.meta(session),
    }


# ------------------------------------------------------------------------------------ jobs


@router.post("/jobs/ingest", response_model=JobOut, status_code=202, tags=["jobs"])
def jobs_ingest(body: JobIngest, jobs: JobsDep):
    job = jobs.submit("ingest", body.model_dump(), J.ingest_steps(body.mode, body.max_requests))
    return JobOut.model_validate(job, from_attributes=True)


@router.post("/jobs/analyze", response_model=JobOut, status_code=202, tags=["jobs"])
def jobs_analyze(body: JobAnalyze, jobs: JobsDep):
    job = jobs.submit("analyze", body.model_dump(), J.analyze_steps(body.steps, body.load_database))
    return JobOut.model_validate(job, from_attributes=True)


@router.get("/jobs", response_model=list[JobOut], tags=["jobs"])
def list_jobs(session: SessionDep, limit: Limit = 20):
    rows = session.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()
    return [JobOut.model_validate(j, from_attributes=True) for j in rows]


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def get_job(session: SessionDep, job_id: Annotated[str, Path(max_length=36)]):
    job = session.get(Job, job_id)
    if job is None:
        raise not_found("job", job_id)
    return JobOut.model_validate(job, from_attributes=True)
