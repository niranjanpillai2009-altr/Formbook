"""Self-contained refresh bundle.

The published page carries everything needed to rebuild itself — the source
of this package, the trained model (XGBoost's own JSON format), and the
historical match table — inside one `<script type="application/json">`
block. A scheduled job can therefore start from nothing but the page's
URL: read the page, unpack the bundle, fetch the latest results, rebuild
the page and republish it. No repository, server or file store required.

    python -m epl_predictor.bundle pack  site/index.html      # embed / update the bundle
    python -m epl_predictor.bundle unpack page.html DEST_DIR  # restore a project
"""
from __future__ import annotations

import base64
import datetime as dt
import gzip
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

BUNDLE_ID = "formbook-bundle"
BUNDLE_VERSION = 1
SOURCE_FILES = [
    "src/epl_predictor/__init__.py", "src/epl_predictor/config.py", "src/epl_predictor/data.py",
    "src/epl_predictor/ratings.py", "src/epl_predictor/features.py", "src/epl_predictor/metrics.py",
    "src/epl_predictor/models.py", "src/epl_predictor/predict.py", "src/epl_predictor/live.py",
    "src/epl_predictor/site.py", "src/epl_predictor/bundle.py", "src/epl_predictor/templates/site.html",
    "refresh_site.py",
]
REQUIREMENTS = "pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\nxgboost>=2.0\nscipy>=1.10\npyarrow>=12.0\njoblib>=1.3\n"


def _b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, compresslevel=9)).decode("ascii")


def _unb64gz(text: str) -> bytes:
    return gzip.decompress(base64.b64decode(text))


def make_bundle() -> dict:
    """Collect source, model and base data from the current project."""
    from .predict import load_bundle
    root = config.PROJECT_ROOT
    files = {rel: (root / rel).read_text() for rel in SOURCE_FILES if (root / rel).exists()}
    files["requirements.txt"] = REQUIREMENTS

    mb = load_bundle()
    model = mb["model"]
    if model.__class__.__name__ != "XGBClassifier":
        raise RuntimeError("the refresh bundle expects the deployed model to be an XGBClassifier")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model.json"
        model.save_model(p)
        model_json = p.read_text()
    meta = {k: mb[k] for k in ("model_name", "features", "classes", "trained_through", "cv_summary", "holdout")}

    matches_csv = (config.PROCESSED_DIR / "matches.csv").read_bytes()
    # only the mirror's own seasons belong in the base table (live results are re-fetched)
    return {
        "version": BUNDLE_VERSION, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": files, "model_json_gz": _b64gz(model_json.encode()), "model_meta": meta,
        "matches_csv_gz": _b64gz(matches_csv),
    }


def embed(html: str, bundle: dict | None = None) -> str:
    """Return `html` with the bundle block appended (replacing any old one)."""
    bundle = bundle or make_bundle()
    payload = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    block = f'<script id="{BUNDLE_ID}" type="application/json">{payload}</script>'
    html = strip(html)
    return html.rstrip() + "\n" + block + "\n"


def strip(html: str) -> str:
    return re.sub(rf'<script id="{BUNDLE_ID}" type="application/json">.*?</script>\s*', "", html, flags=re.S)


def extract(html: str) -> dict:
    m = re.search(rf'<script id="{BUNDLE_ID}" type="application/json">(.*?)</script>', html, flags=re.S)
    if not m:
        raise ValueError("no refresh bundle found in the page")
    return json.loads(m.group(1).replace("<\\/", "</"))


def unpack(bundle: dict, dest: Path) -> Path:
    """Recreate a runnable project under `dest` from a bundle."""
    dest = Path(dest)
    for rel, text in bundle["files"].items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (dest / "models").mkdir(exist_ok=True)
    (dest / "models" / "final_model.json").write_text(_unb64gz(bundle["model_json_gz"]).decode())
    (dest / "models" / "model_meta.json").write_text(json.dumps(bundle["model_meta"]))
    (dest / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (dest / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (dest / "data" / "processed" / "matches.csv").write_bytes(_unb64gz(bundle["matches_csv_gz"]))
    log.info("unpacked bundle v%s (created %s) into %s", bundle["version"], bundle["created"], dest)
    return dest


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack"); a.add_argument("html")
    b = sub.add_parser("unpack"); b.add_argument("html"); b.add_argument("dest")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.cmd == "pack":
        p = Path(args.html)
        p.write_text(embed(p.read_text()))
        log.info("bundle embedded in %s (%.1f KB total)", p, p.stat().st_size / 1024)
    else:
        unpack(extract(Path(args.html).read_text()), Path(args.dest))


if __name__ == "__main__":
    main()
