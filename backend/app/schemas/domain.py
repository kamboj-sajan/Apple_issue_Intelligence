"""Response and request models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class QuestionSummary(BaseModel):
    question_id: int
    title: str
    link: str | None
    created_at: datetime
    tags: list[str]
    primary_category: str | None
    devices: list[str]
    os_versions: list[str]
    score: int | None
    view_count: int | None
    answer_count: int | None
    is_closed_duplicate: bool


class Attribution(BaseModel):
    author_display_name: str | None
    author_link: str | None
    content_license: str | None
    source: str = "Ask Different (apple.stackexchange.com), via the Stack Exchange API"


class EntityOut(BaseModel):
    entity_type: str
    value: str
    surface: str | None
    field: str | None


class QuestionDetail(QuestionSummary):
    body_excerpt: str
    error_codes: list[str]
    categories: list[str]
    duplicate_of: list[int]
    split: str | None
    in_research_window: bool
    entities: list[EntityOut]
    attribution: Attribution


class AnomalySummary(BaseModel):
    anomaly_id: str
    group_type: str
    group_value: str
    week_start: datetime
    observed: int
    expected: float
    abs_change: float
    rel_change: float
    q_value: float
    rank_score: float
    signal_type: str
    rolling_agrees: bool
    partial_week: bool
    low_site_volume: bool


class SeriesPoint(BaseModel):
    week_start: datetime
    observed: int
    expected_negbin: float | None
    expected_rolling: float | None
    site_total: int


class EvidenceQuestion(BaseModel):
    question_id: int
    title: str
    link: str | None
    created_at: datetime


class AnomalyDetail(AnomalySummary):
    p_value: float
    nb_score: float | None
    rolling_z: float | None
    overdispersion_alpha: float | None
    site_total: int
    baseline_start: datetime
    baseline_end: datetime
    co_occurring: list[str]
    evidence_questions: list[EvidenceQuestion]
    series: list[SeriesPoint]
    explanation: dict[str, Any]
    limitations: str


class TrendingCategory(BaseModel):
    category: str
    recent: int
    previous: int
    recent_share: float
    previous_share: float
    change: float


class Trending(BaseModel):
    window_weeks: int
    window_end: datetime | None
    categories: list[TrendingCategory]
    recent_signals: list[AnomalySummary]
    note: str


class GraphNodeOut(BaseModel):
    node_id: str
    node_type: str
    label: str | None
    degree: int
    link: str | None = None


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float
    evidence: str | None
    is_label_edge: bool


class Neighborhood(BaseModel):
    center: GraphNodeOut
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    truncated: bool
    limits: dict[str, int]


class RelatedExplanation(BaseModel):
    shared_tags: list[str]
    shared_entities: dict[str, list[str]]
    lexical_overlap: list[dict[str, Any]]
    embedding_cosine: float | None
    graph_relations: list[dict[str, Any]]
    labelled_relation: str | None


class RelatedItem(BaseModel):
    rank: int
    question_id: int
    title: str
    link: str | None
    created_at: datetime
    score: float
    explanation: RelatedExplanation


class RelatedMethod(BaseModel):
    method: str
    score_description: str
    items: list[RelatedItem]


class RelatedResponse(BaseModel):
    question: QuestionSummary
    methods: list[RelatedMethod]
    note: str


class JobIngest(BaseModel):
    mode: Literal["creation", "update", "fetch-linked"] = "update"
    max_requests: int | None = Field(None, ge=1, le=10000)


class JobAnalyze(BaseModel):
    steps: (
        list[
            Literal[
                "preprocess",
                "split",
                "embed",
                "clusters",
                "anomalies",
                "graph",
                "drift",
                "retrieval-eval",
                "link-predict",
            ]
        ]
        | None
    ) = None
    load_database: bool = True


class JobOut(BaseModel):
    job_id: str
    job_type: str
    status: str
    params: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    current_step: str | None
    log_tail: str
    error: str | None
