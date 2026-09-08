"""Rebuild the Formbook page with the latest results and fixtures.

    python refresh_site.py                  # -> site/index.html (artifact fragment, bundle embedded)
    python refresh_site.py --standalone --no-bundle --out docs/index.html   # GitHub Pages build
    python refresh_site.py --offline        # no network: use stored data only

Steps: pull the newest season files + this season's results and fixture list,
recompute every rating and feature, forecast the remaining fixtures, simulate
the season, render the page, and embed the self-refresh bundle so the
published page can rebuild itself again tomorrow.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from epl_predictor import bundle as bundle_mod  # noqa: E402
from epl_predictor.site import SITE_DIR, build_site_data, render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SITE_DIR / "index.html"))
    ap.add_argument("--offline", action="store_true", help="do not fetch anything; use stored data")
    ap.add_argument("--no-bundle", action="store_true", help="skip embedding the self-refresh bundle")
    ap.add_argument("--standalone", action="store_true",
                    help="write a complete HTML document (GitHub Pages / local file) instead of an artifact fragment")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    data = build_site_data(refresh_data=not args.offline)
    html = render(data, standalone=args.standalone)
    if not args.no_bundle:
        html = bundle_mod.embed(html)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    m = data["meta"]
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"season {m['season_label']}: {m['played']}/{m['total']} played, next matchweek {m['next_round']}, "
          f"results through {m['data_through']}")


if __name__ == "__main__":
    main()
