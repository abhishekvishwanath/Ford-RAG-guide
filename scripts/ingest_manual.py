"""Run manual ingestion from the command line.

Examples:
  python scripts/ingest_manual.py                          # all stages, whole manual
  python scripts/ingest_manual.py --stages parse structure # selected stages
  python scripts/ingest_manual.py --pages 70 140           # page range (0-based, end exclusive)
  python scripts/ingest_manual.py --dry-run                # validate + inspect only, writes nothing
Interrupt with Ctrl-C or create data/processed/CANCEL; re-running resumes from checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
warnings.filterwarnings("ignore", category=FutureWarning)

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.ingestion.ingestion_service import ALL_STAGES, IngestionService  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", choices=ALL_STAGES)
    ap.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    service = IngestionService(settings)
    if args.dry_run:
        service.validate_source()
        from inspect_manual import inspect  # type: ignore[import-not-found]

        print(json.dumps({"valid": True, "inspection": inspect(settings.manual_pdf_path)}, indent=2))
        return
    results = service.run(args.stages, tuple(args.pages) if args.pages else None)
    print(json.dumps(results, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
