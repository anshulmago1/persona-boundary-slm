"""Download AP World History: Modern free-response PDFs from AP Central.

College Board hosts only the last ~3 exam years as direct PDFs under
``/media/pdf`` (older years 301-redirect to an HTML index). We target 2023-2025,
both scoring-set variants, DBQ + LEQ2/3/4, plus the scoring-guideline PDFs.

Filename drift handled:
  - 2023 uses ``...-modern-...`` ; 2024/2025 drop it. We build the known-correct
    name per year and fall back to probing the other variant, so future years and
    quirks still resolve.

Raw PDFs are copyrighted College Board material -> cached to ``data/rowa/pdfs/``
which is gitignored. Only *derived* labels/metrics ever leave this machine.

    python -m src.rowa.scrape --list          # print resolved URLs, download nothing
    python -m src.rowa.scrape                  # download all (skips cached)
    python -m src.rowa.scrape --years 2024     # subset
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BASE = "https://apcentral.collegeboard.org/media/pdf"
UA = "Mozilla/5.0 (research; AP Row A grader dataset)"
PDF_DIR = Path("data/rowa/pdfs")

YEARS = [2023, 2024, 2025]
SETS = [1, 2]
# Question types with a Row A thesis/claim point: DBQ (Q1) + the three LEQs.
QTYPES = ["dbq", "leq2", "leq3", "leq4"]


@dataclass
class Target:
    year: int
    kind: str  # "samples" | "sg"
    qtype: Optional[str]  # None for scoring guidelines
    set_no: int

    @property
    def stem(self) -> str:
        yy = str(self.year)[-2:]
        if self.kind == "sg":
            return f"ap{yy}-sg-world-history-modern-set-{self.set_no}"
        return f"ap{yy}-apc-world-history-{self.qtype}-set-{self.set_no}"

    @property
    def local(self) -> Path:
        return PDF_DIR / f"{self.stem}.pdf"

    def candidate_names(self) -> List[str]:
        """Correct-first, then the modern/non-modern variant as a fallback."""
        yy = str(self.year)[-2:]
        if self.kind == "sg":
            return [f"ap{yy}-sg-world-history-modern-set-{self.set_no}.pdf"]
        # 2023 -> "-modern-", 2024+ -> no "modern"; try the likely one first.
        with_m = f"ap{yy}-apc-world-history-modern-{self.qtype}-set-{self.set_no}.pdf"
        no_m = f"ap{yy}-apc-world-history-{self.qtype}-set-{self.set_no}.pdf"
        return [with_m, no_m] if self.year <= 2023 else [no_m, with_m]


def all_targets(years: List[int]) -> List[Target]:
    targets: List[Target] = []
    for y in years:
        for s in SETS:
            targets.append(Target(y, "sg", None, s))
            for q in QTYPES:
                targets.append(Target(y, "samples", q, s))
    return targets


def _head_is_pdf(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200 and "application/pdf" in r.headers.get("Content-Type", "")
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def resolve_url(t: Target) -> Optional[str]:
    for name in t.candidate_names():
        url = f"{BASE}/{name}"
        if _head_is_pdf(url):
            return url
    return None


def download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(data)


def run(years: List[int], list_only: bool, throttle: float = 0.5) -> None:
    targets = all_targets(years)
    found, missing, cached = 0, 0, 0
    for t in targets:
        if t.local.exists() and not list_only:
            cached += 1
            print(f"cached   {t.local.name}")
            continue
        url = resolve_url(t)
        if not url:
            missing += 1
            print(f"MISSING  {t.stem}.pdf  (no PDF at any candidate URL)")
            continue
        found += 1
        if list_only:
            print(f"OK       {url}")
            continue
        size = download(url, t.local)
        print(f"saved    {t.local.name}  ({size // 1024} KB)")
        time.sleep(throttle)
    print(
        f"\n{'resolved' if list_only else 'downloaded'}: {found}  "
        f"cached: {cached}  missing: {missing}  (of {len(targets)} targets)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    ap.add_argument("--list", action="store_true", help="resolve URLs, download nothing")
    ap.add_argument("--throttle", type=float, default=0.5)
    args = ap.parse_args()
    run(args.years, list_only=args.list, throttle=args.throttle)


if __name__ == "__main__":
    main()
