"""Transparent, dictionary-driven extraction of devices, OS versions, error codes, technical
entities and issue categories. Every match records the field, surface text and rule that fired,
so downstream explanations can cite exact evidence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class EntityMatch:
    entity_type: str  # device | os | error | tech | category
    value: str  # canonical id, e.g. "macos:14", "iphone:15", "error:-36"
    surface: str  # matched text as written
    field: str  # title | body | code | tags
    rule: str  # pattern or rule identifier that produced the match

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityExtractor:
    def __init__(self, config: dict[str, Any]):
        self.devices = [(d["family"], [re.compile(p, FLAGS) for p in d["patterns"]]) for d in config.get("devices", [])]
        os_cfg = config.get("os_versions", {})
        self.os_named = [(re.compile(o["pattern"], FLAGS), o["value"]) for o in os_cfg.get("named", [])]
        self.os_numeric = [(o["os"], re.compile(o["pattern"], FLAGS)) for o in os_cfg.get("numeric", [])]
        self.errors = [(e["prefix"], re.compile(e["pattern"], FLAGS)) for e in config.get("error_codes", [])]
        self.tech = [
            (name, [re.compile(p, FLAGS) for p in pats]) for name, pats in config.get("technical_entities", {}).items()
        ]
        self.categories = [
            (
                name,
                {t.lower() for t in spec.get("tags", [])},
                [re.compile(p, FLAGS) for p in spec.get("keywords", [])],
            )
            for name, spec in config.get("categories", {}).items()
        ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EntityExtractor:
        return cls(yaml.safe_load(Path(path).read_text()))

    @staticmethod
    def _os_value(os_name: str, major: str, minor: str | None) -> str:
        # macOS 10.x releases are distinct products; from 11 onwards the major is the release.
        if os_name == "macos" and major == "10" and minor:
            return f"macos:10.{int(minor)}"
        return f"{os_name}:{int(major)}"

    def extract(self, fields: dict[str, str]) -> list[EntityMatch]:
        """``fields`` maps field name -> text. Returns de-duplicated matches (per value+field)."""
        out: dict[tuple[str, str, str], EntityMatch] = {}

        def add(m: EntityMatch) -> None:
            out.setdefault((m.entity_type, m.value, m.field), m)

        for fname, text in fields.items():
            if not text:
                continue
            for family, pats in self.devices:
                for pat in pats:
                    for m in pat.finditer(text):
                        model = (m.groupdict().get("model") or "").lower()
                        add(EntityMatch("device", family, m.group(0), fname, pat.pattern))
                        if model:
                            add(EntityMatch("device", f"{family}:{model}", m.group(0), fname, pat.pattern))
            for pat, value in self.os_named:
                for m in pat.finditer(text):
                    add(EntityMatch("os", value, m.group(0), fname, pat.pattern))
            for os_name, pat in self.os_numeric:
                for m in pat.finditer(text):
                    value = self._os_value(os_name, m.group("major"), m.group("minor"))
                    add(EntityMatch("os", value, m.group(0), fname, pat.pattern))
            for prefix, pat in self.errors:
                for m in pat.finditer(text):
                    add(EntityMatch("error", f"{prefix}:{m.group('code').lower()}", m.group(0), fname, pat.pattern))
            for name, pats in self.tech:
                for pat in pats:
                    for m in pat.finditer(text):
                        add(EntityMatch("tech", name, m.group(0), fname, pat.pattern))
        # A bare hex code already captured as "error:0x..." should not also appear as "hex:0x...".
        error_codes = {m.value.split(":", 1)[1] for m in out.values() if m.value.startswith("error:")}
        return [m for m in out.values() if not (m.value.startswith("hex:") and m.value.split(":", 1)[1] in error_codes)]

    def categorize(self, tags: list[str], text: str) -> tuple[list[EntityMatch], str]:
        """Return category matches and the primary category ('uncategorized' if none)."""
        scores: dict[str, int] = {}
        matches: list[EntityMatch] = []
        tagset = {t.lower() for t in tags}
        for name, cat_tags, keywords in self.categories:
            hit_tags = sorted(tagset & cat_tags)
            score = 2 * len(hit_tags)
            evidence = [f"tag:{t}" for t in hit_tags]
            for pat in keywords:
                m = pat.search(text)
                if m:
                    score += 1
                    evidence.append(m.group(0))
            if score:
                scores[name] = score
                matches.append(EntityMatch("category", name, "; ".join(evidence), "rules", f"score={score}"))
        if not scores:
            return [], "uncategorized"
        order = [name for name, _, _ in self.categories]
        primary = max(scores, key=lambda n: (scores[n], -order.index(n)))
        return matches, primary
