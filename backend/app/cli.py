"""Backend command line: ``aii-db migrate | load | load-demo``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from alembic import command
from alembic.config import Config as AlembicConfig

from backend.app.core.config import REPO_ROOT, get_settings
from backend.app.db.session import get_engine


def migrate(url: str | None = None) -> None:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    if url:
        cfg.cmd_opts = argparse.Namespace(x=[f"url={url}"])
    command.upgrade(cfg, "head")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aii-db")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    ld = sub.add_parser("load", help="load the current snapshot's artifacts into PostgreSQL")
    ld.add_argument("--no-related", action="store_true", help="skip related-question precomputation")
    ld.add_argument("--no-graphsage", action="store_true", help="skip GraphSAGE re-ranking")
    sub.add_parser("load-demo", help="load the small SYNTHETIC demo dataset (clearly labelled as demo)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    if args.command == "migrate":
        migrate()
        result = {"migrated": True}
    elif args.command == "load":
        from backend.app.services.loader import load_snapshot

        migrate()
        result = load_snapshot(get_engine(), settings.aii_data_dir, not args.no_related, not args.no_graphsage)
    else:
        from backend.app.services.demo import load_demo

        migrate()
        result = load_demo(get_engine())
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
