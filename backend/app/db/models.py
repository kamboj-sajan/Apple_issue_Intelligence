"""Database schema for persisted analysis results (one active snapshot at a time).

Tables mirror the ML pipeline outputs; the loader (``backend.app.services.loader``) replaces their
contents from the artifacts of one snapshot. ``dataset_snapshots.is_demo`` marks the synthetic demo
dataset so every API response can say whether it is real or demo data.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    snapshot_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    pipeline_version: Mapped[str | None] = mapped_column(String(16))
    created_range_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_range_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    n_questions: Mapped[int] = mapped_column(Integer, default=0)
    manifest: Mapped[dict] = mapped_column(JSONB, default=dict)
    collection: Mapped[dict] = mapped_column(JSONB, default=dict)  # last ingestion runs / checkpoint summary


class Question(Base):
    __tablename__ = "questions"
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text)
    link: Mapped[str | None] = mapped_column(Text)
    body_excerpt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    devices: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    os_versions: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    error_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    primary_category: Mapped[str | None] = mapped_column(String(64), index=True)
    score: Mapped[int | None] = mapped_column(Integer)
    view_count: Mapped[int | None] = mapped_column(Integer)
    answer_count: Mapped[int | None] = mapped_column(Integer)
    is_closed_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_of: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), default=list)
    author_display_name: Mapped[str | None] = mapped_column(Text)
    author_link: Mapped[str | None] = mapped_column(Text)
    content_license: Mapped[str | None] = mapped_column(String(32))
    split: Mapped[str | None] = mapped_column(String(8))
    in_research_window: Mapped[bool] = mapped_column(Boolean, default=True)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        Index("ix_questions_tags", "tags", postgresql_using="gin"),
        Index("ix_questions_devices", "devices", postgresql_using="gin"),
        Index("ix_questions_os_versions", "os_versions", postgresql_using="gin"),
        Index(
            "ix_questions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class QuestionEntity(Base):
    __tablename__ = "question_entities"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(BigInteger, index=True)
    entity_type: Mapped[str] = mapped_column(String(16))
    value: Mapped[str] = mapped_column(Text, index=True)
    surface: Mapped[str | None] = mapped_column(Text)
    field: Mapped[str | None] = mapped_column(String(16))


class Anomaly(Base):
    __tablename__ = "anomalies"
    anomaly_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    group_type: Mapped[str] = mapped_column(String(16), index=True)
    group_value: Mapped[str] = mapped_column(Text, index=True)
    week_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    observed: Mapped[int] = mapped_column(Integer)
    expected: Mapped[float] = mapped_column(Float)
    abs_change: Mapped[float] = mapped_column(Float)
    rel_change: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    q_value: Mapped[float] = mapped_column(Float)
    nb_score: Mapped[float | None] = mapped_column(Float)
    rolling_z: Mapped[float | None] = mapped_column(Float)
    rolling_agrees: Mapped[bool] = mapped_column(Boolean, default=False)
    overdispersion_alpha: Mapped[float | None] = mapped_column(Float)
    signal_type: Mapped[str] = mapped_column(String(16))
    rank_score: Mapped[float] = mapped_column(Float, index=True)
    site_total: Mapped[int] = mapped_column(Integer)
    baseline_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    baseline_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    partial_week: Mapped[bool] = mapped_column(Boolean, default=False)
    low_site_volume: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_question_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), default=list)
    co_occurring: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    limitations: Mapped[str] = mapped_column(Text, default="")


class AnomalySeries(Base):
    __tablename__ = "anomaly_series"
    group_type: Mapped[str] = mapped_column(String(16), primary_key=True)
    group_value: Mapped[str] = mapped_column(Text, primary_key=True)
    week_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    observed: Mapped[int] = mapped_column(Integer)
    expected_negbin: Mapped[float | None] = mapped_column(Float)
    expected_rolling: Mapped[float | None] = mapped_column(Float)
    site_total: Mapped[int] = mapped_column(Integer)


class WeeklyTotal(Base):
    __tablename__ = "weekly_totals"
    week_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    site_total: Mapped[int] = mapped_column(Integer)
    partial_week: Mapped[bool] = mapped_column(Boolean, default=False)
    low_site_volume: Mapped[bool] = mapped_column(Boolean, default=False)


class GraphNode(Base):
    __tablename__ = "graph_nodes"
    node_id: Mapped[str] = mapped_column(Text, primary_key=True)
    node_type: Mapped[str] = mapped_column(String(16), index=True)
    label: Mapped[str | None] = mapped_column(Text)
    known_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    degree: Mapped[int] = mapped_column(Integer, default=0)


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    src: Mapped[str] = mapped_column(Text, index=True)
    dst: Mapped[str] = mapped_column(Text, index=True)
    edge_type: Mapped[str] = mapped_column(String(40), index=True)
    weight: Mapped[float] = mapped_column(Float)
    known_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence: Mapped[str | None] = mapped_column(Text)


class RelatedQuestion(Base):
    __tablename__ = "related_questions"
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    method: Mapped[str] = mapped_column(String(16), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    related_id: Mapped[int] = mapped_column(BigInteger)
    score: Mapped[float] = mapped_column(Float)


class TermIdf(Base):
    __tablename__ = "term_idf"
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    idf: Mapped[float] = mapped_column(Float)


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSONB)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    month: Mapped[str] = mapped_column(String(7), primary_key=True)
    metric: Mapped[str] = mapped_column(String(24), primary_key=True)
    status: Mapped[str] = mapped_column(String(24))
    value: Mapped[float | None] = mapped_column(Float)
    z: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    alert: Mapped[bool] = mapped_column(Boolean, default=False)
    calibration: Mapped[bool] = mapped_column(Boolean, default=False)
    n: Mapped[int | None] = mapped_column(Integer)
    n_current: Mapped[int | None] = mapped_column(Integer)
    reference: Mapped[str | None] = mapped_column(String(24))


class DriftClusterMovement(Base):
    __tablename__ = "drift_cluster_movements"
    month: Mapped[str] = mapped_column(String(7), primary_key=True)
    cluster: Mapped[str] = mapped_column(String(16), primary_key=True)
    count: Mapped[int] = mapped_column(Integer)
    share: Mapped[float] = mapped_column(Float)
    reference_share: Mapped[float] = mapped_column(Float)
    share_ratio: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(12))
    top_terms: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


class DriftNewTerm(Base):
    __tablename__ = "drift_new_terms"
    month: Mapped[str] = mapped_column(String(7), primary_key=True)
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    docs: Mapped[int] = mapped_column(Integer)


class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued | running | succeeded | failed | interrupted
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_step: Mapped[str | None] = mapped_column(Text)
    log_tail: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text)
